import os, glob, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets.srdataset import SRDataset
from model.span import SPAN30

# Hyper parameter
train_dir = r"C:\github\dataset\DIV2K_train_HR\DIV2K_train_HR"
scale = 4
hr_patch = 256
batch_size = 16
lr = 1e-3    
epochs = 1000       
save_dir = r"C:\github\SRSharp\python\results"



os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

ds = SRDataset(train_dir, hr_size=hr_patch, scale=scale)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

model = SPAN30(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=scale, bias=True).to(device)
criterion = nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

global_step = 0
for epoch in range(1, epochs + 1):
    model.train()
    t0 = time.time()
    loss_sum = 0.0
    n = 0

    last_lr = None
    last_sr = None

    for lr_img, hr_img in dl:
        lr_img = lr_img.to(device, non_blocking=True)   # (B,3,lr,lr), float32 0~255
        hr_img = hr_img.to(device, non_blocking=True)   # (B,3,hr,hr), float32 0~255

        sr = model(lr_img)

        loss = criterion(sr, hr_img)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        loss_sum += float(loss.item())
        n += 1
        global_step += 1

        last_lr = lr_img[0].detach().clamp(0, 255).to("cpu")
        last_sr = sr[0].detach().clamp(0, 255).to("cpu")


    model.eval()
    with torch.no_grad():
        lr_np = last_lr.permute(1, 2, 0).numpy().astype(np.uint8)       # RGB
        sr_np = last_sr.permute(1, 2, 0).numpy().astype(np.uint8)       # RGB

        lr_up = cv2.resize(lr_np, (hr_patch, hr_patch), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("input", cv2.cvtColor(lr_up, cv2.COLOR_RGB2BGR))
        cv2.imshow("output", cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 또는 q로 종료
            break

    avg_loss = loss_sum / max(1, n)
    dt = time.time() - t0
    print(f"[Epoch {epoch:03d}/{epochs}] loss={avg_loss:.4f} time={dt:.1f}s step={global_step}")

    # 체크포인트 저장
    torch.save(model.state_dict(), os.path.join(save_dir, f"span_stage1_e{epoch:03d}.pth"))

cv2.destroyAllWindows()
