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
import random

# ===================== 固定随机种子 保证复现 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# 自动创建权重文件夹
os.makedirs("./weights", exist_ok=True)

# ===================== 数据集类（无改动） =====================
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

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size) / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)

        base, _ = os.path.splitext(name)
        mask = None
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                break
        # 缺失mask打印警告
        if mask is None:
            print(f"警告：未找到标签 {name}，填充全0掩码")
            mask = np.zeros(self.img_size, dtype=np.uint8)

        mask = cv2.resize(mask, self.img_size) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask

# ===================== 仅保留D训练、D测试、C测试 =====================
# 训练集：DeepCrack (D)
D_TRAIN_IMG = "./datasets/DeepCrack/train_img"
D_TRAIN_MASK = "./datasets/DeepCrack/train_lab"
# 同源测试集：DeepCrack test
D_TEST_IMG = "./datasets/DeepCrack/test_img"
D_TEST_MASK = "./datasets/DeepCrack/test_lab"
# 跨域测试集：Crack500 (C)
C_TEST_IMG = "./datasets/Crack500/test/images"
C_TEST_MASK = "./datasets/Crack500/test/masks"

# 构建数据集
train_D = CrackDataset(D_TRAIN_IMG, D_TRAIN_MASK)
test_D = CrackDataset(D_TEST_IMG, D_TEST_MASK)
test_C = CrackDataset(C_TEST_IMG, C_TEST_MASK)

# 加载器配置
batch_size = 8
train_loader = DataLoader(train_D, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader_D = DataLoader(test_D, batch_size=1, shuffle=False, num_workers=0)
test_loader_C = DataLoader(test_C, batch_size=1, shuffle=False, num_workers=0)

# ===================== 模型、训练配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 切换三组消融模型只需注释切换
# from models.baseline_unet import unet
# from models.clip import clipunet
from models.clip_ECA import clipunet_eca
model = clipunet_eca(n_channels=3, n_classes=1,device=device).to(device)

# 损失函数
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], device=device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

print("=" * 60)
print("训练配置：仅 DeepCrack(D) 训练，测试：D同源 / D→C跨域")
print("=" * 60)

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(imgs,is_train=True)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d} | 平均Loss: {avg_loss:.4f}")

# 动态命名权重，区分模型，防止覆盖
model_name = "clip_eca"
save_path = f"./weights/{model_name}_train_D.pth"
torch.save(model.state_dict(), save_path)
print(f"\n模型训练完成，权重保存至：{save_path}")

# ===================== 评估函数：统一阈值搜索区间 0~0.95 =====================
def test_model(model, test_loader, device="cuda"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="测试推理中"):
            imgs = imgs.to(device)
            outputs = model(imgs,is_train=False)
            prob = torch.sigmoid(outputs).cpu().numpy().flatten()
            label = masks.cpu().numpy().flatten()

            all_probs.append(prob)
            all_labels.append(label)

    y_true = np.concatenate(all_labels)
    y_pred_prob = np.concatenate(all_probs)

    crack_ratio = np.sum(y_true) / len(y_true)
    print(f"\n缺陷像素占比: {crack_ratio:.6f}")

    auc = roc_auc_score(y_true, y_pred_prob)

    best_f1 = 0.0
    best_threshold = 0.5
    # 同源、跨域统一搜索范围，对比公平
    for t in np.arange(0.0, 0.95, 0.01):
        y_pred_bin = (y_pred_prob > t).astype(np.int32)
        current_f1 = f1_score(y_true, y_pred_bin)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    ap_crack = average_precision_score(y_true, y_pred_prob)
    ap_bg = average_precision_score(1 - y_true, 1 - y_pred_prob)
    map50 = (ap_bg + ap_crack) / 2.0

    return auc, best_f1, map50, best_threshold

# ===================== 测试：先同源D→D，再跨域D→C =====================
print("\n" + "=" * 60)
print("           【同源测试】 D训练 → D测试集")
print("=" * 60)
auc_d, f1_d, map50_d, th_d = test_model(model, test_loader_D, device)
print(f"AUC      : {auc_d:.4f}")
print(f"最优F1   : {f1_d:.4f} (阈值 = {th_d:.2f})")
print(f"mAP50    : {map50_d:.4f}")
print("=" * 60)

print("\n" + "=" * 60)
print("        【跨域测试】D训练 → C (Crack500)")
print("=" * 60)
auc_c, f1_c, map50_c, th_c = test_model(model, test_loader_C, device)
print(f"AUC      : {auc_c:.4f}")
print(f"最优F1   : {f1_c:.4f} (阈值 = {th_c:.2f})")
print(f"mAP50    : {map50_c:.4f}")
print("=" * 60)

# 输出参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型总参数量: {total_params / 1e6:.2f} M")