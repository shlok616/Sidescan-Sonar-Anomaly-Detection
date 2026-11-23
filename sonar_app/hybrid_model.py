"""
Hybrid CNN -> Mamba -> Transformer detector for side-scan sonar (Team Vantyx, SIH26057).

Evolved from the Colab prototype (Debris-sonar.ipynb):
  * CNN stem       : ResNet-18 trunk (ImageNet-initialised, grey input), 16x downsample, with a
                     stride-8 skip for small targets. (The prototype's from-scratch CNNStem is kept
                     below for reference; it reached only ~10 % val mAP on this small dataset.)
  * MambaBlock     : bidirectional selective state-space block (Mamba-2 / SSD style scalar-decay
                     scan, computed in its exact quadratic "masked attention" form so it runs on CPU)
  * TransformerBlock: global multi-head self-attention, as in the prototype
  * Heads          : CenterNet-style anchor-free detection -> per-class centre heatmap,
                     box size and sub-cell offset (stride 8), plus the pooled class head
                     from the prototype for an image-level class distribution.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["aircraft", "fish", "other", "shipwreck"]


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = conv_bn(ch, ch)
        self.c2 = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch))

    def forward(self, x):
        return F.silu(x + self.c2(self.c1(x)))


class CNNStem(nn.Module):
    """1 -> 32 -> 64 -> 96 -> d_model, each stage stride 2 (16x total)."""

    def __init__(self, in_ch=1, out_ch=128):
        super().__init__()
        chs = [in_ch, 32, 64, 96, out_ch]
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [conv_bn(chs[i], chs[i + 1], 3, 2)]
            if i > 0:
                blocks.append(ResBlock(chs[i + 1]))
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, x):
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return feats[2], feats[3]  # stride 8 (96ch), stride 16 (d_model)


class MambaBlock(nn.Module):
    """Selective SSM (Mamba-2 SSD form) over the raster-scanned token sequence, run in both
    directions. h_t = a_t h_{t-1} + B_t x_t ; y_t = C_t h_t, with input-dependent
    a_t = exp(-softplus(dt_t) * A) per head. Uses the exact quadratic (masked) form."""

    def __init__(self, d_model, n_heads=4, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.d_inner = d_model * expand
        self.h = n_heads
        self.p = self.d_inner // n_heads
        self.n = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.bc_proj = nn.Linear(self.d_inner, 2 * n_heads * d_state)
        self.dt_proj = nn.Linear(self.d_inner, n_heads)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, n_heads)))
        self.D = nn.Parameter(torch.ones(n_heads))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(0.05)))

    def _scan(self, x):  # x: (B, L, d_inner) after conv
        B, L, _ = x.shape
        bc = self.bc_proj(x).view(B, L, 2, self.h, self.n)
        Bm, Cm = bc[:, :, 0], bc[:, :, 1]                       # (B,L,H,N)
        dt = F.softplus(self.dt_proj(x))                        # (B,L,H)
        loga = -dt * torch.exp(self.A_log)                      # log decay, <= 0
        cum = torch.cumsum(loga, dim=1).transpose(1, 2)         # (B,H,L)
        seg = cum.unsqueeze(-1) - cum.unsqueeze(-2)             # (B,H,L,L) = sum_{s<k<=t} loga
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        decay = torch.exp(seg.masked_fill(~mask, -float("inf")))
        scores = torch.einsum("blhn,bshn->bhls", Cm, Bm) * decay
        xv = (x * 1.0).view(B, L, self.h, self.p)
        xdt = xv * dt.unsqueeze(-1)
        y = torch.einsum("bhls,bshp->blhp", scores, xdt)
        y = y + xv * self.D.view(1, 1, -1, 1)
        return y.reshape(B, L, self.d_inner)

    def forward(self, x):
        B, L, _ = x.shape
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        y = self._scan(u) + self._scan(u.flip(1)).flip(1)       # bidirectional
        return x + self.out_proj(y * F.silu(z))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(y)) + idt, inplace=True)


class ResNetStem(nn.Module):
    """ResNet-18 trunk (conv1..layer3) used as the CNN stem; single-channel input.
    Can be initialised from ImageNet weights (conv1 RGB filters summed to grey)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.out_channels = (128, 256)

    def load_imagenet(self, path):
        sd = torch.load(path, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith(("layer4", "fc"))}
        sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        return missing, unexpected

    def forward(self, x):
        x = (x - 0.449) / 0.226
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x)), inplace=True), 3, 2, 1)
        x = self.layer1(x)
        f8 = self.layer2(x)
        return f8, self.layer3(f8)


class HybridSonarDetector(nn.Module):
    def __init__(self, d_model=128, num_classes=len(CLASS_NAMES), grid=20):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_stem = ResNetStem()
        c8, c16 = self.backbone_stem.out_channels
        self.token_proj = conv_bn(c16, d_model, 1)
        self.pos = nn.Parameter(torch.zeros(1, d_model, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mamba_layer = MambaBlock(d_model)
        self.transformer_layer = TransformerBlock(d_model)
        # stride-8 fusion neck
        self.lateral = conv_bn(c8, d_model, 1)
        self.fuse = nn.Sequential(conv_bn(d_model, d_model), ResBlock(d_model))
        # detection heads (stride 8)
        def head(out):
            return nn.Sequential(conv_bn(d_model, 64), nn.Conv2d(64, out, 1))
        self.hm_head = head(num_classes)
        self.wh_head = head(2)
        self.off_head = head(2)
        self.hm_head[-1].bias.data.fill_(-2.19)
        # prototype's global class head
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d_model, num_classes))

    def forward(self, x, return_features=False):
        f8, f16 = self.backbone_stem(x)
        f16 = self.token_proj(f16)
        B, C, H, W = f16.shape
        pos = self.pos if self.pos.shape[-2:] == (H, W) else F.interpolate(self.pos, (H, W), mode="bilinear", align_corners=False)
        tokens = (f16 + pos).flatten(2).transpose(1, 2)
        tokens = self.mamba_layer(tokens)
        tokens = self.transformer_layer(tokens)
        g = tokens.transpose(1, 2).reshape(B, C, H, W)
        feat = self.fuse(self.lateral(f8) + F.interpolate(g, size=f8.shape[-2:], mode="nearest"))
        out = {
            "heatmap": self.hm_head(feat),          # logits (B,K,H8,W8)
            "wh": self.wh_head(feat),               # box size in stride-8 cells (log space)
            "offset": self.off_head(feat).sigmoid(),
            "cls_logits": self.cls_head(g),
            "activation": g.norm(dim=1, keepdim=True),  # Mamba+Transformer token energy (for visualisation)
        }
        if return_features:
            out["features"] = feat
        return out


@torch.no_grad()
def decode(out, img_size, k=50, thresh=0.25):
    """Return list (per image) of detections [x1,y1,x2,y2,score,cls] in input-pixel coords."""
    hm = out["heatmap"].sigmoid()
    B, K, H, W = hm.shape
    stride = img_size / W
    peak = (F.max_pool2d(hm, 3, 1, 1) == hm).float() * hm
    res = []
    for b in range(B):
        scores, idx = peak[b].flatten().topk(min(k, K * H * W))
        dets = []
        for s, i in zip(scores.tolist(), idx.tolist()):
            if s < thresh:
                break
            c, rem = divmod(i, H * W)
            y, x = divmod(rem, W)
            ox, oy = out["offset"][b, :, y, x].tolist()
            w, h = out["wh"][b, :, y, x].exp().mul(stride).tolist()
            cx, cy = (x + ox) * stride, (y + oy) * stride
            dets.append([max(0, cx - w / 2), max(0, cy - h / 2), min(img_size, cx + w / 2), min(img_size, cy + h / 2), s, c])
        res.append(nms(dets))
    return res


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: -d[4])
    keep = []
    for d in dets:
        if all(iou(d, k) < thr for k in keep):
            keep.append(d)
    return keep
