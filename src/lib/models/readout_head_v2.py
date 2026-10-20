"""Exp131: read-out head v2 on the frozen FairMOT backbone, aimed at localisation.

Exp129's head found the member (IoU >= 0.3 for 93% of present validation states) but placed its box coarsely (IoU >= 0.5
for 57%, against 82% for the crop verifier). v2 keeps the template / window design and adds:
  - the DLA base level at stride 2 (32 channels) read over the same window, for spatial detail;
  - a finer window grid (48 x 48 instead of 32 x 32);
  - a sub-cell centre offset regressed at every cell;
  - ltrb measured from the true centre, decoded from the refined centre.
Backbone, Re-ID and detection heads stay frozen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from models import readout_head as R

WIN = 48
TMPL = R.TMPL
WINDOW_SIDES = R.WINDOW_SIDES
INSIDE = R.INSIDE


@torch.no_grad()
def backbone_forward(model, x):
    """(stride-4 fused features, raw Re-ID map, DLA base level 1 at stride 2)."""
    levels = model.base(x)
    fine = levels[1].clone()
    b = model.dla_up(levels)
    y = [b[i].clone() for i in range(model.last_level - model.first_level)]
    model.ida_up(y, 0, len(y))
    feat = y[-1]
    return feat, model.id(feat), fine


def read_inputs(feat_t, id_t, fine_t, idx_t, feat_a, id_a, idx_a, tmpl_boxes, pred, s0, ratio_t, left_t, top_t, ratio_a, left_a, top_a):
    q_win = R.window_rois(idx_t, pred, s0, ratio_t, left_t, top_t)
    win_feat = roi_align(feat_t, q_win, (WIN, WIN), spatial_scale=1 / 4, sampling_ratio=2, aligned=True)
    win_fine = roi_align(fine_t, q_win, (WIN, WIN), spatial_scale=1 / 2, sampling_ratio=2, aligned=True)
    win_emb = F.normalize(roi_align(id_t, q_win, (WIN, WIN), spatial_scale=1 / 4, sampling_ratio=2, aligned=True), dim=1)
    q_t = R.template_rois(idx_a, tmpl_boxes, ratio_a, left_a, top_a)
    tmpl_feat = roi_align(feat_a, q_t, (TMPL, TMPL), spatial_scale=1 / 4, sampling_ratio=2, aligned=True)
    cx, cy = (q_t[:, 1] + q_t[:, 3]) / 2, (q_t[:, 2] + q_t[:, 4]) / 2
    q_c = R._rois(idx_a, cx - 2, cy - 2, cx + 2, cy + 2)
    tmpl_emb = F.normalize(roi_align(id_a, q_c, (1, 1), spatial_scale=1 / 4, sampling_ratio=2, aligned=True)[:, :, 0, 0], dim=1)
    return tmpl_feat, tmpl_emb, win_feat, win_fine, win_emb


class ReadoutHeadV2(nn.Module):
    def __init__(self, feat_ch=64, fine_ch=32, mid=32, width=64):
        super().__init__()
        self.tmpl = nn.Conv2d(feat_ch, mid, 1)
        self.win = nn.Conv2d(feat_ch, mid, 1)
        self.fine = nn.Sequential(nn.Conv2d(fine_ch, mid, 3, padding=1), nn.GroupNorm(8, mid), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            nn.Conv2d(3 * mid + 1 + 2, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True))
        self.center = nn.Conv2d(width, 1, 1)
        self.offset = nn.Conv2d(width, 2, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.exist = nn.Linear(2 * width, 1)
        self.center.bias.data.fill_(-2.19)

    def forward(self, tmpl_feat, tmpl_emb, win_feat, win_fine, win_emb, scal, use_template=True):
        q = tmpl_feat.shape[0]
        if not use_template:
            tmpl_feat, tmpl_emb = torch.zeros_like(tmpl_feat), torch.zeros_like(tmpl_emb)
        k = self.tmpl(tmpl_feat)
        w = self.win(win_feat)
        mid = k.shape[1]
        corr = F.conv2d(w.reshape(1, q * mid, WIN, WIN), k.reshape(q * mid, 1, TMPL, TMPL), padding=TMPL // 2, groups=q * mid)
        corr = corr.reshape(q, mid, WIN, WIN) / (TMPL * TMPL)
        cos = (win_emb * tmpl_emb[:, :, None, None]).sum(1, keepdim=True)
        sc = scal[:, :, None, None].expand(-1, -1, WIN, WIN)
        h = self.trunk(torch.cat([corr, w, self.fine(win_fine), cos, sc], 1))
        pooled = torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1)
        return {'center': self.center(h), 'offset': self.offset(h), 'box': self.box(h), 'exist': self.exist(pooled)[:, 0]}


def target_cell(gc, pred, s0):
    rel = (gc - pred) / s0[:, None]
    inside = rel.abs().amax(1) < INSIDE
    pos = (rel / WINDOW_SIDES + .5) * WIN
    i = pos[:, 0].floor().clamp(0, WIN - 1).long()
    j = pos[:, 1].floor().clamp(0, WIN - 1).long()
    return j, i, inside, pos


def focal_center_loss(logit, j, i, pos, inside):
    q = logit.shape[0]
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)[:, 0]
    ys, xs = torch.meshgrid(torch.arange(WIN, device=logit.device), torch.arange(WIN, device=logit.device), indexing='ij')
    d2 = (xs[None] + .5 - pos[:, 0, None, None]) ** 2 + (ys[None] + .5 - pos[:, 1, None, None]) ** 2
    gauss = torch.where(inside[:, None, None], torch.exp(-d2 / (2 * 1.5 ** 2)), torch.zeros(q, WIN, WIN, device=logit.device))
    is_pos = torch.zeros_like(gauss, dtype=torch.bool)
    is_pos[inside.nonzero()[:, 0], j[inside], i[inside]] = True
    pos_loss = -(1 - p) ** 2 * torch.log(p) * is_pos
    neg_loss = -(1 - gauss) ** 4 * p ** 2 * torch.log(1 - p) * (~is_pos)
    return (pos_loss.sum() + neg_loss.sum()) / max(int(is_pos.sum()), 1)


def decode(out, pred, s0):
    q = out['center'].shape[0]
    flat = out['center'].reshape(q, -1).argmax(1)
    j, i = flat // WIN, flat % WIN
    ar = torch.arange(q, device=pred.device)
    off = out['offset'][ar, :, j, i]
    cx = pred[:, 0] + ((i.float() + off[:, 0]) / WIN - .5) * WINDOW_SIDES * s0
    cy = pred[:, 1] + ((j.float() + off[:, 1]) / WIN - .5) * WINDOW_SIDES * s0
    ltrb = out['box'][ar, :, j, i].clamp(min=0) * s0[:, None]
    box = torch.stack([cx - ltrb[:, 0], cy - ltrb[:, 1], cx + ltrb[:, 2], cy + ltrb[:, 3]], 1)
    return torch.sigmoid(out['exist']), box
