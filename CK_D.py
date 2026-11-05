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
import pandas as pd

# ===================== 1. 一次性导入所有模型（不用再注释） =====================
from models.baseline_unet import unet
from models.new_clip import clipunet
from models.new_clip_ECA import clipunet_eca  # 注意：这里要和你实际的类名一致


# ===================== 2. 数据集类（完全复用你的逻辑） =====================
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

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"KSDD2 GT文件不存在: {gt_path}")
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask


# ===================== 3. 路径配置（完全复用你的路径） =====================
# 训练集
C_TRAIN_IMG = "./datasets/Crack500/train/images"
C_TRAIN_MASK = "./datasets/Crack500/train/masks"
K_TRAIN_DIR = "../datasets/ksdd2/train"

# 同域测试集
C_TEST_IMG = "./datasets/Crack500/test/images"
C_TEST_MASK = "./datasets/Crack500/test/masks"
K_TEST_DIR = "../datasets/ksdd2/test"

# 跨域测试集
D_TEST_IMG = "./datasets/DeepCrack/test_img"
D_TEST_MASK = "./datasets/DeepCrack/test_lab"

# ===================== 4. 一次性加载所有数据集（只加载一次，节省时间） =====================
print("\n" + "=" * 60)
print("正在加载所有数据集...")
print("=" * 60)

# 训练集：C + K 混合
train_C = CrackDataset(C_TRAIN_IMG, C_TRAIN_MASK)
train_K = KSDD2Dataset(K_TRAIN_DIR)
train_CK = ConcatDataset([train_C, train_K])

# 同域测试集
test_C = CrackDataset(C_TEST_IMG, C_TEST_MASK)
test_K = KSDD2Dataset(K_TEST_DIR)
test_CK = ConcatDataset([test_C, test_K])

# 跨域测试集
test_D = CrackDataset(D_TEST_IMG, D_TEST_MASK)

# 数据加载器
batch_size = 8
train_loader = DataLoader(train_CK, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader_CK = DataLoader(test_CK, batch_size=1, shuffle=False, num_workers=0)
test_loader_D = DataLoader(test_D, batch_size=1, shuffle=False, num_workers=0)

print("\n✅ 所有数据集加载完成！")


# ===================== 5. 通用训练函数（封装重复逻辑） =====================
def train_single_model(model, model_name, save_path, device, epochs=20):
    print(f"\n{'=' * 60}")
    print(f"           开始训练：{model_name}")
    print(f"{'=' * 60}")

    # 每个模型独立初始化优化器和调度器
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0], device=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            # 自动适配是否需要is_train参数
            try:
                outputs = model(imgs, is_train=True)
            except TypeError:
                outputs = model(imgs)  # Baseline不需要is_train

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1:2d} | 平均Loss: {avg_loss:.4f}")

    # 保存权重
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ {model_name} 训练完成，权重已保存至：{save_path}")

    return model


# ===================== 6. 通用评估函数（完全复用你的逻辑） =====================
def test_model(model, test_loader, device="cuda"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="测试推理中"):
            imgs = imgs.to(device)

            # 自动适配是否需要is_train参数
            try:
                outputs = model(imgs, is_train=False)
            except TypeError:
                outputs = model(imgs)  # Baseline不需要is_train

            prob = torch.sigmoid(outputs).cpu().numpy().flatten()
            label = masks.cpu().numpy().flatten()
            all_probs.append(prob)
            all_labels.append(label)

    y_true = np.concatenate(all_labels)
    y_pred_prob = np.concatenate(all_probs)

    crack_ratio = np.sum(y_true) / len(y_true)
    auc = roc_auc_score(y_true, y_pred_prob)

    best_f1 = 0.0
    best_threshold = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        y_pred_bin = (y_pred_prob > t).astype(np.int32)
        current_f1 = f1_score(y_true, y_pred_bin)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    ap_crack = average_precision_score(y_true, y_pred_prob)
    ap_bg = average_precision_score(1 - y_true, 1 - y_pred_prob)
    map50 = (ap_bg + ap_crack) / 2.0

    return auc, best_f1, map50, best_threshold, crack_ratio


# ===================== 7. 模型配置中心（控制跑哪些模型，不用改代码） =====================

MODEL_CONFIGS = [
    {
        "name": "Baseline UNet",
        "model": unet(n_channels=3, n_classes=1),
        "save_path": "./weights/baseline_CK_unet.pth"
    },
    {
        "name": "CLIP UNet",
        "model": clipunet(n_channels=3, n_classes=1,
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "save_path": "./weights/clip_CK_unet.pth"
    },
    {
        "name": "CLIP+ECA UNet",
        "model": clipunet_eca(n_channels=3, n_classes=1,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "save_path": "./weights/clip_eca_CK_unet.pth"
    }
]

# ===================== 8. 主循环：依次跑所有模型 =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")

    # 存储所有模型的结果
    all_results = []

    for config in MODEL_CONFIGS:
        model_name = config["name"]
        model = config["model"].to(device)
        save_path = config["save_path"]

        # 训练模型
        model = train_single_model(model, model_name, save_path, device)

        # 同域测试
        print(f"\n{'=' * 60}")
        print(f"           【同域测试】 {model_name} | C+K 测试集")
        print(f"{'=' * 60}")
        auc_ck, f1_ck, map50_ck, th_ck, ratio_ck = test_model(model, test_loader_CK, device)
        print(f"缺陷像素占比: {ratio_ck:.6f}")
        print(f"AUC      : {auc_ck:.4f}")
        print(f"最优F1   : {f1_ck:.4f} (阈值 = {th_ck:.2f})")
        print(f"mAP50    : {map50_ck:.4f}")

        # 跨域测试
        print(f"\n{'=' * 60}")
        print(f"        【跨域迁移测试】 {model_name} | 迁移到 DeepCrack(D)")
        print(f"{'=' * 60}")
        auc_d, f1_d, map50_d, th_d, ratio_d = test_model(model, test_loader_D, device)
        print(f"缺陷像素占比: {ratio_d:.6f}")
        print(f"AUC      : {auc_d:.4f}")
        print(f"最优F1   : {f1_d:.4f} (阈值 = {th_d:.2f})")
        print(f"mAP50    : {map50_d:.4f}")

        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        params_m = total_params / 1e6
        print(f"\n模型总参数量: {params_m:.2f} M")

        # 保存结果
        all_results.append({
            "模型": model_name,
            "同域AUC": auc_ck,
            "同域F1": f1_ck,
            "同域mAP50": map50_ck,
            "同域阈值": th_ck,
            "跨域AUC": auc_d,
            "跨域F1": f1_d,
            "跨域mAP50": map50_d,
            "跨域阈值": th_d,
            "参数量(M)": params_m
        })

        # 释放显存，避免下一个模型OOM
        del model
        torch.cuda.empty_cache()
        print(f"\n✅ {model_name} 全部完成，已释放显存")

    # ===================== 9. 最终结果对比表格 =====================
    print("\n" + "=" * 100)
    print("                     所有模型结果对比")
    print("=" * 100)

    # 转换为DataFrame并格式化
    df = pd.DataFrame(all_results)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df.to_string(index=False))

    # 保存结果到CSV文件，方便后续分析
    df.to_csv("./results/CK_D_all_models_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 所有结果已保存至：./results/CK_D_all_models_results.csv")