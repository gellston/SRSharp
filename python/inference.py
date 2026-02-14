import os
import cv2
import numpy as np
import torch

from model.span import SPAN30

# -----------------------
# Config
# -----------------------
img_path    = r"C:\github\dataset\DIV2K_valid_LR_bicubic_X4\DIV2K_valid_LR_bicubic\X4\0838x4.png"
weight_path = r"C:\github\SRSharp\python\results\finetune_final.pth"
save_path   = r"C:\github\SRSharp\python\results\sr_output_tiled.png"

scale  = 4

tile   = 512   # LR 타일(core) 크기 (메모리 부족하면 줄이기: 128/96 등)
margin = 16    # LR 마진(컨텍스트) 크기 (경계 어색하면 늘리기: 24/32 등)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load model
# -----------------------
model = SPAN30(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=scale, bias=True).to(device)
ckpt = torch.load(weight_path, map_location=device)
model.load_state_dict(ckpt, strict=True)
model.eval()

# -----------------------
# Load image (BGR -> RGB)
# -----------------------
bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
if bgr is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없음: {img_path}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

H, W = rgb.shape[:2]
Hs, Ws = H * scale, W * scale

# 출력 캔버스 (RGB로 쌓고 마지막에 BGR로 저장/표시)
out_rgb = np.empty((Hs, Ws, 3), dtype=np.uint8)

# -----------------------
# Tiled inference with margin (no blending)
# -----------------------
with torch.no_grad():
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            # core 영역 (원본 이미지 내)
            y_end = min(y + tile, H)
            x_end = min(x + tile, W)
            core_h = y_end - y
            core_w = x_end - x

            # 확장 패치 좌표 (마진 포함)
            x0 = x - margin
            y0 = y - margin
            x1 = x_end + margin
            y1 = y_end + margin

            # 이미지 경계로 클립
            cx0 = max(0, x0)
            cy0 = max(0, y0)
            cx1 = min(W, x1)
            cy1 = min(H, y1)

            patch = rgb[cy0:cy1, cx0:cx1]

            # 부족한 마진만큼 reflect padding (블렌딩 아님)
            pad_left   = cx0 - x0
            pad_top    = cy0 - y0
            pad_right  = x1 - cx1
            pad_bottom = y1 - cy1

            if pad_left or pad_top or pad_right or pad_bottom:
                patch = cv2.copyMakeBorder(
                    patch,
                    top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )

            # 모델 입력: (1,3,Hp,Wp) float32 0~255
            inp = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).contiguous().to(torch.float32).to(device)

            sr = model(inp)[0].clamp(0, 255)  # (3, Hp*scale, Wp*scale)

            # core에 해당하는 위치만 crop (마진 제거)
            # padding/reflection을 줬으므로 core의 시작은 항상 margin 위치
            core_x0 = margin
            core_y0 = margin
            core_x1 = margin + core_w
            core_y1 = margin + core_h

            sx0 = core_x0 * scale
            sy0 = core_y0 * scale
            sx1 = core_x1 * scale
            sy1 = core_y1 * scale

            sr_core = sr[:, sy0:sy1, sx0:sx1]  # (3, core_h*scale, core_w*scale)
            sr_core_np = sr_core.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # RGB

            # 출력 캔버스에 그대로 붙이기
            oy0 = y * scale
            ox0 = x * scale
            oy1 = y_end * scale
            ox1 = x_end * scale
            out_rgb[oy0:oy1, ox0:ox1] = sr_core_np

        print(f"Row done: y={y}/{H}")

# -----------------------
# Save + Show
# -----------------------
out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
ok = cv2.imwrite(save_path, out_bgr)
if not ok:
    raise RuntimeError(f"저장 실패: {save_path}")
print(f"Saved: {save_path}")

# 원본도 비교용으로 nearest upsample (그냥 비교용)
orig_up = cv2.resize(bgr, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

cv2.namedWindow("original (nearest up)", cv2.WINDOW_NORMAL)
cv2.namedWindow("sr output (tiled)", cv2.WINDOW_NORMAL)
cv2.imshow("original (nearest up)", orig_up)
cv2.imshow("sr output (tiled)", out_bgr)

print("Press ESC or q to quit.")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord('q'):
        break
cv2.destroyAllWindows()
