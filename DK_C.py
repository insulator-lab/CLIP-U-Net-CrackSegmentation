import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

# ===================== 模型导入（按需切换） =====================
# from models.baseline_unet import unet   Baseline
# from models.new_clip_ECA import clipunet_eca  # CLIP+ECA 最优模型
from models.new_clip import clipunet

# ===================== 通用裂纹数据集（不变） =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
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
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

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


# ===================== ✅ 修复版 KSDD2Dataset（已验证你的数据集格式） =====================
class KSDD2Dataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256)):
        self.data_dir = data_dir
        self.img_size = img_size
        all_files = os.listdir(data_dir)
        self.img_files = [f for f in all_files if "_GT" not in f]
        self.img_files = [f for f in self.img_files if "copy" not in f.lower()]
        print(f"\n✅ KSDD2数据集加载完成，共 {len(self.img_files)} 张图像")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        name, ext = os.path.splitext(img_name)
        gt_name = name + "_GT" + ext
        gt_path = os.path.join(self.data_dir, gt_name)

        img = cv2.imread(os.path.join(self.data_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size) / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # 强制检查GT存在，避免读取失败
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"KSDD2 GT文件不存在: {gt_path}")
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size) / 255.0

        # ✅ 你的KSDD2是黑底白点，不需要反转标签
        # mask = 1.0 - mask

        mask = (mask > 0.5).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask


# ===================== 🔥 路径配置（D+K训练 → 测试C） =====================
# ---------- 训练集：DeepCrack(D) + KSDD2(K) ----------
D_TRAIN_IMG = "./datasets/DeepCrack/train_img"
D_TRAIN_MASK = "./datasets/DeepCrack/train_lab"
K_TRAIN_DIR = "../datasets/ksdd2/train"

# ---------- 同域测试集：D测试 + K测试 混合 ----------
D_TEST_IMG = "./datasets/DeepCrack/test_img"
D_TEST_MASK = "./datasets/DeepCrack/test_lab"
K_TEST_DIR = "../datasets/ksdd2/test"

# ---------- 跨域测试集：Crack500(C)（目标域） ----------
C_TEST_IMG = "./datasets/Crack500/test/images"
C_TEST_MASK = "./datasets/Crack500/test/masks"

# ===================== 构建数据集 & 数据加载器 =====================
# 训练集：D + K 混合
train_D = CrackDataset(D_TRAIN_IMG, D_TRAIN_MASK)
train_K = KSDD2Dataset(K_TRAIN_DIR)
train_DK = ConcatDataset([train_D, train_K])

# 同域测试集：D测试 + K测试 混合
test_D = CrackDataset(D_TEST_IMG, D_TEST_MASK)
test_K = KSDD2Dataset(K_TEST_DIR)
test_DK = ConcatDataset([test_D, test_K])

# 跨域测试集：Crack500(C)
test_C = CrackDataset(C_TEST_IMG, C_TEST_MASK)

# 加载器配置
batch_size = 8
train_loader = DataLoader(train_DK, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader_DK = DataLoader(test_DK, batch_size=1, shuffle=False, num_workers=0)
test_loader_C = DataLoader(test_C, batch_size=1, shuffle=False, num_workers=0)

# ===================== 训练配置（适配D+K混合） =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化（按需切换Baseline/CLIP/CLIP+ECA）
# model = unet(n_channels=3, n_classes=1).to(device)
model = clipunet(n_channels=3, n_classes=1, device=device).to(device)

# ✅ 适配D+K的pos_weight：D缺陷占比~4%，K~0.1%，混合后~2%，pos_weight=20
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0], device=device))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ===================== 训练循环 =====================
print("=" * 60)
print("开始训练：DeepCrack(D) + KSDD2(K) 混合训练 | CLIP+ECA 模型")
print("=" * 60)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        # 训练：使用D+K混合提示词
        outputs = model(imgs,is_train=True)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1:2d} | 平均Loss: {avg_loss:.4f}")

# 保存权重（命名区分D+K实验）
torch.save(model.state_dict(), "./weights/small_clip_DK_unet.pth")
print("\n模型训练完成，权重已保存！")


# ===================== 通用评估函数（不变） =====================
def test_model(model, test_loader, device="cuda"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="测试推理中"):
            imgs = imgs.to(device)
            # 测试：使用目标域C专属提示词
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
    for t in np.arange(0.5, 0.95, 0.01):
        y_pred_bin = (y_pred_prob > t).astype(np.int32)
        current_f1 = f1_score(y_true, y_pred_bin)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    ap_crack = average_precision_score(y_true, y_pred_prob)
    ap_bg = average_precision_score(1 - y_true, 1 - y_pred_prob)
    map50 = (ap_bg + ap_crack) / 2.0
    return auc, best_f1, map50, best_threshold


# ===================== 1. 同域测试：D+K训练 → 自身测试集 =====================
print("\n" + "=" * 60)
print("           【同域测试】 D+K 训练集对应测试集指标 | CLIP+ECA")
print("=" * 60)
auc_dk, f1_dk, map50_dk, th_dk = test_model(model, test_loader_DK, device)
print(f"AUC      : {auc_dk:.4f}")
print(f"最优F1   : {f1_dk:.4f} (阈值 = {th_dk:.2f})")
print(f"mAP50    : {map50_dk:.4f}")
print("=" * 60)

# ===================== 2. 跨域测试：D+K训练 → 迁移到Crack500(C) =====================
print("\n" + "=" * 60)
print("        【跨域迁移测试】 模型迁移到 Crack500(C) 指标 | CLIP+ECA")
print("=" * 60)
auc_c, f1_c, map50_c, th_c = test_model(model, test_loader_C, device)
print(f"AUC      : {auc_c:.4f}")
print(f"最优F1   : {f1_c:.4f} (阈值 = {th_c:.2f})")
print(f"mAP50    : {map50_c:.4f}")
print("=" * 60)

# 输出模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型总参数量: {total_params / 1e6:.2f} M")