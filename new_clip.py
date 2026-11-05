import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from .baseline_unet import DoubleConv, Down, Up
import os




class CLIPTextEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad = False

        # ===================== 训练提示词：C（路面细裂纹）+ K（金属缺陷）=====================
        self.train_prompts = [

            # ========== C：Crack500 特征（细、灰、粗糙背景、低对比）==========
            "medium gray crack on rough textured concrete surface",
            "intermittent linear fracture on gravelly pavement",
            "moderate contrast crack on noisy concrete road",
            "thin dark crack on textured asphalt surface",

            # ========== K：金属缺陷（极细、低对比，和路面完全不同）==========
            "fine scratch on polished metal surface",
            "tiny micro crack on shiny steel sheet",
            "thin linear defect on smooth metal plate",

            # ========== 负样本：正常路面 + 正常金属（必须加）==========
            "clean concrete road with no visible crack",
            "smooth pavement without linear defects",
            "smooth metal surface with no defect"
        ]

        # ===================== 测试提示词：D（DeepCrack 特征）=====================
        self.test_prompts = [
            # ========== D：DeepCrack 特征（粗、黑、高对比、干净背景）==========
            "thick black crack on smooth asphalt surface",
            "wide continuous fracture on plain concrete road",
            "high contrast dark crack on uniform pavement",
            # ========== 负样本：正常路面（和训练负提示词对齐）==========
            "clean asphalt road with no crack",
            "plain concrete road without defect"
        ]
        # 提前tokenize
        self.train_tokens = clip.tokenize(self.train_prompts).to(device)
        self.test_tokens = clip.tokenize(self.test_prompts).to(device)

    def forward(self, img, is_train=True):
        B = img.shape[0]
        with torch.no_grad():
            # 训练/推理 自动切换提示词
            tokens = self.train_tokens if is_train else self.test_tokens
            text_emb = self.model.encode_text(tokens).float()
            text_emb = text_emb.mean(dim=0, keepdim=True)
            text_emb = text_emb.expand(B, -1)
        return text_emb


# ===================== 交叉注意力（不变） =====================
class CrossAttention(nn.Module):
    def __init__(self, dim, text_dim=512):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Linear(text_dim, dim)
        self.v = nn.Linear(text_dim, dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, text):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).flatten(2)
        k = self.k(text).unsqueeze(-1)
        v = self.v(text).unsqueeze(-1)

        attn = torch.matmul(q.transpose(1, 2), k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(1, 2))
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(out)


# ===================== 主模型：CLIP+ECA+UNet（仅forward加is_train） =====================
class clipunet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = CLIPTextEncoder(self.device)
        self.bilinear = bilinear

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.cross_attn = CrossAttention(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)



        self.out = nn.Conv2d(64, n_classes, 1)
        self._set_trainable()

    def _set_trainable(self):
        for name, param in self.named_parameters():
            if "text_encoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x, is_train=True):  # ✅ 加is_train标志
        text_feat = self.text_encoder(x, is_train=is_train)  # 自动切换提示词

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.cross_attn(x5, text_feat)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)


        return self.out(x)