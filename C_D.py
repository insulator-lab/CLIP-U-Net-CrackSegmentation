import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

# from models.baseline_unet import unet
# from models.clip import clipunet
from models.clip_ECA import clipunet_eca

# ===================== 数据集类（仅保留通用裂纹数据集，适配C+D） =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256,256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]

        # 读图像
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size) / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)

        # 读掩码（自动处理 jpg/png 后缀，容错防崩溃）
        base, _ = os.path.splitext(name)
        mask = None
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                break
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        mask = cv2.resize(mask, self.img_size) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask

# ===================== 路径配置【仅保留 C 训练 + D 测试】 =====================
# 训练集：仅 Crack500 (C)
C_TRAIN_IMG = "./datasets/Crack500/train/images"
C_TRAIN_MASK = "./datasets/Crack500/train/masks"
C_TEST_IMG = "./datasets/Crack500/test/images"
C_TEST_MASK = "./datasets/Crack500/test/masks"
# 测试集：仅 DeepCrack (D)
D_TEST_IMG = "./datasets/DeepCrack/test_img"
D_TEST_MASK = "./datasets/DeepCrack/test_lab"

# ===================== 构建数据集 & 数据加载器 =====================
# 训练：Crack500
train_C = CrackDataset(C_TRAIN_IMG, C_TRAIN_MASK)
test_C = CrackDataset(C_TEST_IMG, C_TEST_MASK)
# 测试：DeepCrack
test_D = CrackDataset(D_TEST_IMG, D_TEST_MASK)

# 加载器配置
batch_size = 8
train_loader = DataLoader(train_C, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader_C = DataLoader(test_C, batch_size=1, shuffle=False, num_workers=0)
test_loader_D = DataLoader(test_D, batch_size=1, shuffle=False, num_workers=0)

# ===================== 训练配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = unet(n_channels=3, n_classes=1).to(device)
# model = clipunet(n_channels=3, n_classes=1).to(device)
model = clipunet_eca(n_channels=3, n_classes=1,device=device).to(device)

# 加权BCE 解决正负样本不均衡
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0], device=device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ===================== 训练循环 =====================
print("=" * 60)
print("开始训练：Crack500 (C) 单数据集训练")
print("=" * 60)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d} | 平均Loss: {avg_loss:.4f}")

# 保存权重
torch.save(model.state_dict(), "./weights/clip_eca_C2D.pth")

print("\n模型训练完成，权重已保存！")

# ===================== 通用评估函数 =====================
def test_model(model, test_loader, device="cuda"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="测试推理中"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            prob = torch.sigmoid(outputs).cpu().numpy().flatten()
            label = masks.cpu().numpy().flatten()

            all_probs.append(prob)
            all_labels.append(label)

    y_true = np.concatenate(all_labels)
    y_pred_prob = np.concatenate(all_probs)

    # 缺陷像素占比
    crack_ratio = np.sum(y_true) / len(y_true)
    print(f"\n缺陷像素占比: {crack_ratio:.6f}")

    # AUC
    auc = roc_auc_score(y_true, y_pred_prob)

    # 最优F1 + 阈值
    best_f1 = 0.0
    best_threshold = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        y_pred_bin = (y_pred_prob > t).astype(np.int32)
        current_f1 = f1_score(y_true, y_pred_bin)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    # mAP50
    ap_crack = average_precision_score(y_true, y_pred_prob)
    ap_bg = average_precision_score(1 - y_true, 1 - y_pred_prob)
    map50 = (ap_bg + ap_crack) / 2.0

    return auc, best_f1, map50, best_threshold

print("\n" + "=" * 60)
print("           【同域测试】 C 训练 → C 自身测试集指标")
print("=" * 60)
auc_c, f1_c, map50_c, th_c = test_model(model, test_loader_C, device)
print(f"AUC      : {auc_c:.4f}")
print(f"最优F1   : {f1_c:.4f} (阈值 = {th_c:.2f})")
print(f"mAP50    : {map50_c:.4f}")
print("=" * 60)
# ===================== 仅测试：跨域 D (DeepCrack) =====================
print("\n" + "=" * 60)
print("        【跨域测试】C 训练 → D (DeepCrack) 指标")
print("=" * 60)
auc_d, f1_d, map50_d, th_d = test_model(model, test_loader_D, device)
print(f"AUC      : {auc_d:.4f}")
print(f"最优F1   : {f1_d:.4f} (阈值 = {th_d:.2f})")
print(f"mAP50    : {map50_d:.4f}")
print("=" * 60)

# 输出模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型总参数量: {total_params / 1e6:.2f} M")



