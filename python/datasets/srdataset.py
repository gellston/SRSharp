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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read: {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        if H < self.hr or W < self.hr:
            s = max(self.hr / H, self.hr / W)
            newW = int(np.ceil(W * s))
            newH = int(np.ceil(H * s))
            img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)
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


        if np.random.rand() < 0.5:
            hr = hr[:, ::-1, :]
        if np.random.rand() < 0.5:
            hr = hr[::-1, :, :]
        hr = np.ascontiguousarray(hr)

        lr = cv2.resize(hr, (self.lr, self.lr), interpolation=cv2.INTER_AREA)
        lr = np.ascontiguousarray(lr)

        hr = torch.from_numpy(hr).permute(2, 0, 1).contiguous().to(torch.float32)
        lr = torch.from_numpy(lr).permute(2, 0, 1).contiguous().to(torch.float32)

        return lr, hr
