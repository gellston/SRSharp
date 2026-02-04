import torch
import torch.nn.functional as F


class Conv3XC2(torch.nn.Module):
    """
    목적:
      - 학습(train) 시: (입력 패딩 -> 1x1 -> 3x3(valid) -> 1x1) + shortcut(1x1)
      - 추론(eval) 시: 위 선형 연산을 단일 3x3 Conv(eval_conv)로 가중치/바이어스 퓨즈하여 실행

    주의:
      - 이 퓨즈는 '피처 퓨즈'가 아니라 '가중치(커널) 퓨즈(= re-parameterization)' 입니다.
      - ONNX의 constant folding이 자동으로 Conv-Conv-Conv를 1개 Conv로 바꿔주진 않는 경우가 대부분이라,
        배포 시엔 이렇게 사전에 fuse한 모델이 더 단순/빠를 수 있습니다.
    """

    def __init__(self, c_in, c_out, gain1=1, s=1, groups=1, bias=True, relu=False):
        super().__init__()

        self.stride = s
        self.has_relu = relu
        self.groups = groups
        self.gain = int(gain1)

        # 기본 검증 (그룹 컨브는 채널이 groups로 나누어떨어져야 함)
        assert c_in % groups == 0, f"c_in({c_in}) must be divisible by groups({groups})"
        assert c_out % groups == 0, f"c_out({c_out}) must be divisible by groups({groups})"
        assert (c_in * self.gain) % groups == 0, f"c_in*gain({c_in*self.gain}) must be divisible by groups({groups})"
        assert (c_out * self.gain) % groups == 0, f"c_out*gain({c_out*self.gain}) must be divisible by groups({groups})"

        # shortcut: 1x1 (stride는 메인 3x3와 동일해야 더하기 가능)
        self.sk = torch.nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
            groups=groups,
        )

        # 메인 경로: 1x1 -> 3x3(valid) -> 1x1
        # (원 코드처럼 conv[1]에서 padding=0 이기 때문에, forward에서 입력을 pad해서 크기 맞춤)
        self.conv1 = torch.nn.Conv2d(c_in, c_in * self.gain, kernel_size=1, padding=0, bias=bias, groups=groups)
        self.conv2 = torch.nn.Conv2d(c_in * self.gain, c_out * self.gain, kernel_size=3, stride=s, padding=0, bias=bias, groups=groups)
        self.conv3 = torch.nn.Conv2d(c_out * self.gain, c_out, kernel_size=1, padding=0, bias=bias, groups=groups)

        self.conv = torch.nn.Sequential(self.conv1, self.conv2, self.conv3)

        # 추론용 단일 3x3 (padding=1로 "same" 출력 크기)
        self.eval_conv = torch.nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
            groups=groups,
        )
        self.eval_conv.weight.requires_grad_(False)
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.requires_grad_(False)

        # fuse 캐시 플래그
        self._fused = False

        # 초기 fuse(선택). 학습 중엔 conv 가중치가 계속 바뀌므로, 실제 배포 전엔 다시 fuse해야 함.
        self.fuse()

    def train(self, mode: bool = True):
        """
        train/eval 전환 시 fuse 캐시 처리:
          - train 모드로 들어가면(가중치가 바뀔 수 있으니) fused 캐시를 무효화
        """
        super().train(mode)
        if mode:
            self._fused = False
        return self

    @torch.no_grad()
    def fuse(self):
        """
        (conv1 -> conv2 -> conv3) + sk 를 eval_conv 하나로 합성합니다.
        그룹 컨브를 고려하여 그룹별로 커널/바이어스를 계산합니다.
        """
        G = self.groups

        # 가중치/바이어스 가져오기 (detach 권장)
        w1 = self.conv1.weight.detach()
        w2 = self.conv2.weight.detach()
        w3 = self.conv3.weight.detach()

        b1 = self.conv1.bias.detach() if self.conv1.bias is not None else None
        b2 = self.conv2.bias.detach() if self.conv2.bias is not None else None
        b3 = self.conv3.bias.detach() if self.conv3.bias is not None else None

        # 그룹당 채널 수 계산 (여기가 원 코드에서 가장 크게 잘못되던 부분)
        in_per_g = self.conv1.in_channels // G                    # = c_in / G
        mid1_per_g = self.conv1.out_channels // G                 # = (c_in*gain) / G
        mid2_per_g = self.conv2.out_channels // G                 # = (c_out*gain) / G
        out_per_g = self.conv3.out_channels // G                  # = c_out / G

        # 합성 결과(3x3) 초기화
        weight_fused = torch.zeros_like(self.eval_conv.weight)
        bias_fused = torch.zeros_like(self.eval_conv.bias) if self.eval_conv.bias is not None else None

        # 그룹별로 합성
        for g in range(G):
            # ---- 그룹별 w 슬라이싱 ----
            # w1: (mid1_total, in_per_g, 1,1)  -> 그룹별 (mid1_per_g, in_per_g)
            w1_g = w1[g * mid1_per_g:(g + 1) * mid1_per_g].squeeze(-1).squeeze(-1)  # (mid1_per_g, in_per_g)

            # w2: (mid2_total, mid1_per_g, 3,3) -> 그룹별 (mid2_per_g, mid1_per_g, 3,3)
            w2_g = w2[g * mid2_per_g:(g + 1) * mid2_per_g]  # (mid2_per_g, mid1_per_g, 3,3)

            # w3: (out_total, mid2_per_g, 1,1) -> 그룹별 (out_per_g, mid2_per_g)
            w3_g = w3[g * out_per_g:(g + 1) * out_per_g].squeeze(-1).squeeze(-1)  # (out_per_g, mid2_per_g)

            # ---- 커널 합성 ----
            # 1x1(conv1) -> 3x3(conv2):
            # k12[o2, in, x, y] = sum_m w2[o2,m,x,y] * w1[m,in]
            k12 = torch.einsum("omxy,mi->oixy", w2_g, w1_g)  # (mid2_per_g, in_per_g, 3,3)

            # (k12) -> 1x1(conv3):
            # k123[o, in, x, y] = sum_m2 w3[o,m2] * k12[m2,in,x,y]
            k123 = torch.einsum("om,mixy->oixy", w3_g, k12)  # (out_per_g, in_per_g, 3,3)

            weight_fused[g * out_per_g:(g + 1) * out_per_g] = k123

            # ---- 바이어스 합성 (bias=True인 경우에만) ----
            if bias_fused is not None:
                # None이면 0으로 간주
                dtype = weight_fused.dtype
                device = weight_fused.device

                b1_g = b1[g * mid1_per_g:(g + 1) * mid1_per_g] if b1 is not None else torch.zeros(mid1_per_g, device=device, dtype=dtype)
                b2_g = b2[g * mid2_per_g:(g + 1) * mid2_per_g] if b2 is not None else torch.zeros(mid2_per_g, device=device, dtype=dtype)
                b3_g = b3[g * out_per_g:(g + 1) * out_per_g] if b3 is not None else torch.zeros(out_per_g, device=device, dtype=dtype)

                # conv2에서 b1이 만드는 출력 상수항:
                # b12 = b2 + sum_{m, x,y} w2[o2,m,x,y]*b1[m]
                s2 = w2_g.sum(dim=(2, 3))      # (mid2_per_g, mid1_per_g)
                b12 = b2_g + (s2 @ b1_g)       # (mid2_per_g,)

                # conv3에서 b12가 만드는 출력 바이어스:
                # b_out = b3 + sum_{m2} w3[o,m2] * b12[m2]
                b_out = b3_g + (w3_g @ b12)    # (out_per_g,)

                bias_fused[g * out_per_g:(g + 1) * out_per_g] = b_out

        # ---- shortcut(sk) 합성: 1x1을 3x3 중앙에 임베딩해서 더함 ----
        sk_w = self.sk.weight.detach()  # (c_out, c_in/G, 1,1)
        sk_w_padded = F.pad(sk_w, (1, 1, 1, 1))  # -> (c_out, c_in/G, 3,3)
        weight_fused = weight_fused + sk_w_padded

        if bias_fused is not None and self.sk.bias is not None:
            bias_fused = bias_fused + self.sk.bias.detach()

        # eval_conv에 반영
        self.eval_conv.weight.copy_(weight_fused)
        if bias_fused is not None:
            self.eval_conv.bias.copy_(bias_fused)

        self._fused = True

    def forward(self, x):
        if self.training:
            # 원 코드와 동일: conv2(3x3)가 padding=0이라서 입력을 먼저 pad
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            # eval에서는 가중치가 안 바뀐다고 가정하고 1번만 fuse
            if not self._fused:
                self.fuse()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB1(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB1, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC2(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC2(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC2(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        # return out, out1, sim_att
        return out, out1, sim_att



class SPAN30(torch.nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=48,
                 upscale=4,
                 bias=True,
                 #img_range=255.,
                 #rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super(SPAN30, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        #self.img_range = img_range
        #self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC2(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB1(feature_channels, bias=bias)
        self.block_2 = SPAB1(feature_channels, bias=bias)
        self.block_3 = SPAB1(feature_channels, bias=bias)
        self.block_4 = SPAB1(feature_channels, bias=bias)
        self.block_5 = SPAB1(feature_channels, bias=bias)
        self.block_6 = SPAB1(feature_channels, bias=bias)

        self.conv_cat = torch.nn.Conv2d(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC2(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(feature_channels,
                            out_channels * (upscale ** 2),
                            padding=1,
                            kernel_size=3),
            torch.nn.PixelShuffle(upscale)
        )


        self.eval().cuda()
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        output = self(input_tensor)

    def forward(self, x):

        #self.mean = self.mean.type_as(x)
        #x = (x - self.mean) * self.img_range
        out_feature = self.conv_1(x)

        out_b1, out_b0_2, att1 = self.block_1(out_feature)
        out_b2, out_b1_2, att2 = self.block_2(out_b1)

        out_b3, out_b2_2, att3 = self.block_3(out_b2)
        out_b4, out_b3_2, att4 = self.block_4(out_b3)
        out_b5, out_b4_2, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_final = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_final, out_b1, out_b5_2], 1))
        output = self.upsampler(out)
        output = torch.clamp(output, 0.0, 255.0)
        return output