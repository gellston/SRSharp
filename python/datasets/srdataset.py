import os, glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, image_dir, hr_size=192, scale=4):
        self.paths = sorted(glob.glob(os.path.join(image_dir, "*")))
        self.hr = int(hr_size)
        self.scale = int(scale)
        assert self.hr % self.scale == 0
        self.lr = self.hr // self.scale

        # ✅ 랜덤 resize augmentation용 interpolation 후보들
        # - downsample: AREA가 기본적으로 좋지만 다양화 위해 섞음
        # - upsample(작은 이미지 키울 때): LINEAR/CUBIC/LANCZOS 섞음
        self.down_inters = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        self.down_probs  = [0.55,          0.20,           0.20,          0.05]

        self.up_inters   = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        self.up_probs    = [0.25,            0.55,           0.20]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read: {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # ✅ 작은 이미지면 키우는 resize도 랜덤 interpolation 적용
        if H < self.hr or W < self.hr:
            s = max(self.hr / H, self.hr / W)
            newW = int(np.ceil(W * s))
            newH = int(np.ceil(H * s))
            up_interp = np.random.choice(self.up_inters, p=self.up_probs)
            img = cv2.resize(img, (newW, newH), interpolation=up_interp)
            H, W = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.abs(dx) + np.abs(dy)

        I = np.pad(edge, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

        hN = H - self.hr + 1
        wN = W - self.hr + 1

        S = I[self.hr:, self.hr:] - I[:-self.hr, self.hr:] - I[self.hr:, :-self.hr] + I[:-self.hr, :-self.hr]

        w = (S.reshape(-1) + 1e-6).astype(np.float64)
        w /= w.sum()
        cdf = np.cumsum(w)
        k = int(np.searchsorted(cdf, np.random.rand(), side="right"))
        y = k // wN
        x = k % wN

        hr = img[y:y + self.hr, x:x + self.hr].copy()

        # flip augment
        if np.random.rand() < 0.5:
            hr = hr[:, ::-1, :]
        if np.random.rand() < 0.5:
            hr = hr[::-1, :, :]
        hr = np.ascontiguousarray(hr)

        # ✅ LR 생성 downsample resize를 랜덤 interpolation으로
        down_interp = np.random.choice(self.down_inters, p=self.down_probs)
        lr = cv2.resize(hr, (self.lr, self.lr), interpolation=down_interp)
        lr = np.ascontiguousarray(lr)

        hr = torch.from_numpy(hr).permute(2, 0, 1).contiguous().to(torch.float32)
        lr = torch.from_numpy(lr).permute(2, 0, 1).contiguous().to(torch.float32)

        return lr, hr