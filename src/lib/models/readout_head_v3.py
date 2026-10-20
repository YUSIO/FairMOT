"""Exp134: read-out head v3 = v2 plus a native-resolution pixel branch over the same window and template.

Exp129-133: the read-out on frozen FairMOT features finds members without detector peaks, but places boxes worse than the crop
verifier (IoU >= 0.5: 0.57-0.71 against 0.82) and cannot tell clutter tracks from members (present-vs-ghost AUC 0.80-0.85
against 0.91). The crop verifier's edge is that it looks at pixels. v3 keeps every v2 input and adds, inside the same head:
  - the letterboxed input image read over the window at 96 x 96 (two pixels per window cell), through a small CNN to 48 x 48;
  - the template's pixels (1.5 x the template box, 32 x 32) pooled to a vector and correlated with the window's pixel features.
The image is the network input itself (no second crop network, no rule): the head reads features and pixels at the queried place.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from models import readout_head as R
from models import readout_head_v2 as V2

WIN = V2.WIN
PIX = 2 * WIN
TPIX = 32
TMPL_PIX_SCALE = 1.5
backbone_forward = V2.backbone_forward
target_cell = V2.target_cell
focal_center_loss = V2.focal_center_loss
decode = V2.decode


def read_inputs(feat_t, id_t, fine_t, img_t, idx_t, feat_a, id_a, img_a, idx_a, tmpl_boxes, pred, s0,
                ratio_t, left_t, top_t, ratio_a, left_a, top_a):
    tmpl_feat, tmpl_emb, win_feat, win_fine, win_emb = V2.read_inputs(feat_t, id_t, fine_t, idx_t, feat_a, id_a, idx_a, tmpl_boxes, pred, s0,
                                                                     ratio_t, left_t, top_t, ratio_a, left_a, top_a)
    q_win = R.window_rois(idx_t, pred, s0, ratio_t, left_t, top_t)
    win_rgb = roi_align(img_t, q_win, (PIX, PIX), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    q_t = R.template_rois(idx_a, tmpl_boxes, ratio_a, left_a, top_a)
    cx, cy = (q_t[:, 1] + q_t[:, 3]) / 2, (q_t[:, 2] + q_t[:, 4]) / 2
    hw = torch.maximum(q_t[:, 3] - q_t[:, 1], q_t[:, 4] - q_t[:, 2]) * TMPL_PIX_SCALE / 2
    q_tp = R._rois(idx_a, cx - hw, cy - hw, cx + hw, cy + hw)
    tmpl_rgb = roi_align(img_a, q_tp, (TPIX, TPIX), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return tmpl_feat, tmpl_emb, win_feat, win_fine, win_emb, win_rgb, tmpl_rgb


def _cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.ReLU(inplace=True))


class ReadoutHeadV3(nn.Module):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, feat_ch=64, fine_ch=32, mid=32, width=64, pix=32):
        super().__init__()
        self.tmpl = nn.Conv2d(feat_ch, mid, 1)
        self.win = nn.Conv2d(feat_ch, mid, 1)
        self.fine = nn.Sequential(nn.Conv2d(fine_ch, mid, 3, padding=1), nn.GroupNorm(8, mid), nn.ReLU(inplace=True))
        self.pix_win = nn.Sequential(_cbr(3, 16), _cbr(16, pix, 2), _cbr(pix, pix))          # 96 -> 48
        self.pix_tmpl = nn.Sequential(_cbr(3, 16), _cbr(16, pix, 2), _cbr(pix, pix))         # 32 -> 16
        self.trunk = nn.Sequential(
            nn.Conv2d(3 * mid + 1 + 2 + pix + 1, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True))
        self.center = nn.Conv2d(width, 1, 1)
        self.offset = nn.Conv2d(width, 2, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.exist = nn.Linear(2 * width + pix, 1)
        self.center.bias.data.fill_(-2.19)
        self.register_buffer('mean', torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.STD).view(1, 3, 1, 1))

    def forward(self, tmpl_feat, tmpl_emb, win_feat, win_fine, win_emb, win_rgb, tmpl_rgb, scal, use_template=True):
        q = tmpl_feat.shape[0]
        if not use_template:
            tmpl_feat, tmpl_emb, tmpl_rgb = torch.zeros_like(tmpl_feat), torch.zeros_like(tmpl_emb), torch.zeros_like(tmpl_rgb)
        k = self.tmpl(tmpl_feat)
        w = self.win(win_feat)
        mid = k.shape[1]
        corr = F.conv2d(w.reshape(1, q * mid, WIN, WIN), k.reshape(q * mid, 1, V2.TMPL, V2.TMPL), padding=V2.TMPL // 2, groups=q * mid)
        corr = corr.reshape(q, mid, WIN, WIN) / (V2.TMPL * V2.TMPL)
        cos = (win_emb * tmpl_emb[:, :, None, None]).sum(1, keepdim=True)
        sc = scal[:, :, None, None].expand(-1, -1, WIN, WIN)
        pw = self.pix_win((win_rgb - self.mean) / self.std)                                  # (q, pix, 48, 48)
        pt = self.pix_tmpl((tmpl_rgb - self.mean) / self.std).mean((2, 3))                   # (q, pix)
        pcorr = (F.normalize(pw, dim=1) * F.normalize(pt, dim=1)[:, :, None, None]).sum(1, keepdim=True)
        h = self.trunk(torch.cat([corr, w, self.fine(win_fine), cos, sc, pw, pcorr], 1))
        pooled = torch.cat([h.amax((2, 3)), h.mean((2, 3)), pt], 1)
        return {'center': self.center(h), 'offset': self.offset(h), 'box': self.box(h), 'exist': self.exist(pooled)[:, 0]}
