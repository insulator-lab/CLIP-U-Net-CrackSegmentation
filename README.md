Script	Target task	Reported in
D2C.py	DeepCrack → Crack500	Table 2
C_D.py	Crack500 → DeepCrack	Table 2
DK_C.py	DeepCrack + KSDD2 → Crack500	Table 3
CK_D.py	Crack500 + KSDD2 → DeepCrack	Table 3
模型通过 models/ 下不同文件切换（baseline_unet.py = Baseline，new_clip.py = Baseline+CLIP，clip_ECA.py = Baseline+CLIP+ECA）。
