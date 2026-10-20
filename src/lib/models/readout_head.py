"""Exp129: a read-out head on the frozen FairMOT backbone.

A query is a track slot: a template (the track's last hit box in an earlier frame) and a predicted centre in the current
frame. The head reads a scale-normalised search window of the current frame (6 sides of the template box, resampled to
32 x 32) and answers whether the slot's member is inside it, where its centre is, and its box. Nothing in the backbone,
the Re-ID head or the detection heads is changed or trained.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

WINDOW_SIDES = 6.0
WIN = 32
TMPL = 7
STRIDE = 4
INSIDE = 2.9          # a member is "inside" when its centre is within INSIDE sides of the predicted centre (per axis)
IMG_H, IMG_W = 608, 1088


def letterbox_params(h0, w0, height=IMG_H, width=IMG_W):
    """ratio, left, top of datasets.dataset.jde.letterbox for an h0 x w0 image."""
    ratio = min(float(height) / h0, float(width) / w0)
    new_w, new_h = round(w0 * ratio), round(h0 * ratio)
    dw, dh = (width - new_w) / 2, (height - new_h) / 2
    return ratio, round(dw - 0.1), round(dh - 0.1)


@torch.no_grad()
def backbone_forward(model, x):
    """The DLASeg forward without the detection heads: (64-channel stride-4 feature map, raw 128-d Re-ID map)."""
    b = model.dla_up(model.base(x))
    y = [b[i].clone() for i in range(model.last_level - model.first_level)]
    model.ida_up(y, 0, len(y))
    feat = y[-1]
    return feat, model.id(feat)


def _rois(batch_idx, x1, y1, x2, y2):
    return torch.stack([batch_idx.float(), x1, y1, x2, y2], 1)


def window_rois(idx, pred, s0, ratio, left, top):
    """RoIs (input-pixel coordinates) of the search windows. pred: (Q, 2) predicted centres in original pixels."""
    half = WINDOW_SIDES / 2 * s0
    cx, cy = pred[:, 0], pred[:, 1]
    return _rois(idx, (cx - half) * ratio + left, (cy - half) * ratio + top, (cx + half) * ratio + left, (cy + half) * ratio + top)


def template_rois(idx, boxes, ratio, left, top):
    return _rois(idx, boxes[:, 0] * ratio + left, boxes[:, 1] * ratio + top, boxes[:, 2] * ratio + left, boxes[:, 3] * ratio + top)


def read_inputs(feat_t, id_t, idx_t, feat_a, id_a, idx_a, tmpl_boxes, pred, s0, ratio_t, left_t, top_t, ratio_a, left_a, top_a):
    """Template / window tensors of Q queries. *_t: banks of the current frames, *_a: banks of the template frames."""
    q_win = window_rois(idx_t, pred, s0, ratio_t, left_t, top_t)
    win_feat = roi_align(feat_t, q_win, (WIN, WIN), spatial_scale=1 / STRIDE, sampling_ratio=2, aligned=True)
    win_emb = F.normalize(roi_align(id_t, q_win, (WIN, WIN), spatial_scale=1 / STRIDE, sampling_ratio=2, aligned=True), dim=1)
    q_t = template_rois(idx_a, tmpl_boxes, ratio_a, left_a, top_a)
    tmpl_feat = roi_align(feat_a, q_t, (TMPL, TMPL), spatial_scale=1 / STRIDE, sampling_ratio=2, aligned=True)
    cx, cy = (q_t[:, 1] + q_t[:, 3]) / 2, (q_t[:, 2] + q_t[:, 4]) / 2
    q_c = _rois(idx_a, cx - 2, cy - 2, cx + 2, cy + 2)
    tmpl_emb = F.normalize(roi_align(id_a, q_c, (1, 1), spatial_scale=1 / STRIDE, sampling_ratio=2, aligned=True)[:, :, 0, 0], dim=1)
    return tmpl_feat, tmpl_emb, win_feat, win_emb


def scalars(s0, age):
    return torch.stack([torch.log(s0 / 20.0), age / 30.0], 1)


class ReadoutHead(nn.Module):
    """Template cross-correlation over the window; outputs centre logits, ltrb boxes (in template sides) and existence."""

    def __init__(self, feat_ch=64, emb_dim=128, mid=32, width=64):
        super().__init__()
        self.tmpl = nn.Conv2d(feat_ch, mid, 1)
        self.win = nn.Conv2d(feat_ch, mid, 1)
        self.trunk = nn.Sequential(
            nn.Conv2d(2 * mid + 1 + 2, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True))
        self.center = nn.Conv2d(width, 1, 1)
        self.box = nn.Conv2d(width, 4, 1)
        self.exist = nn.Linear(2 * width, 1)
        self.center.bias.data.fill_(-2.19)

    def forward(self, tmpl_feat, tmpl_emb, win_feat, win_emb, scal, use_template=True):
        q = tmpl_feat.shape[0]
        if not use_template:
            tmpl_feat, tmpl_emb = torch.zeros_like(tmpl_feat), torch.zeros_like(tmpl_emb)
        k = self.tmpl(tmpl_feat)                                   # (Q, mid, 7, 7)
        w = self.win(win_feat)                                     # (Q, mid, 32, 32)
        mid = k.shape[1]
        corr = F.conv2d(w.reshape(1, q * mid, WIN, WIN), k.reshape(q * mid, 1, TMPL, TMPL), padding=TMPL // 2, groups=q * mid)
        corr = corr.reshape(q, mid, WIN, WIN) / (TMPL * TMPL)
        cos = (win_emb * tmpl_emb[:, :, None, None]).sum(1, keepdim=True)
        sc = scal[:, :, None, None].expand(-1, -1, WIN, WIN)
        h = self.trunk(torch.cat([corr, w, cos, sc], 1))
        pooled = torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1)
        return {'center': self.center(h), 'box': self.box(h), 'exist': self.exist(pooled)[:, 0]}


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def decode(out, pred, s0):
    """Highest centre cell -> original-pixel box (x1, y1, x2, y2) and the existence probability."""
    q = out['center'].shape[0]
    flat = out['center'].reshape(q, -1).argmax(1)
    j, i = flat // WIN, flat % WIN
    cx = pred[:, 0] + ((i.float() + .5) / WIN - .5) * WINDOW_SIDES * s0
    cy = pred[:, 1] + ((j.float() + .5) / WIN - .5) * WINDOW_SIDES * s0
    ltrb = out['box'][torch.arange(q), :, j, i].clamp(min=0) * s0[:, None]
    box = torch.stack([cx - ltrb[:, 0], cy - ltrb[:, 1], cx + ltrb[:, 2], cy + ltrb[:, 3]], 1)
    return torch.sigmoid(out['exist']), box


def target_cell(gc, pred, s0):
    """Cell (j, i) of a ground-truth centre inside the window, and whether it is inside the INSIDE-sides region."""
    rel = (gc - pred) / s0[:, None]
    inside = (rel.abs().amax(1) < INSIDE)
    pos = (rel / WINDOW_SIDES + .5) * WIN
    i = pos[:, 0].floor().clamp(0, WIN - 1).long()
    j = pos[:, 1].floor().clamp(0, WIN - 1).long()
    return j, i, inside, pos


def focal_center_loss(logit, j, i, pos, inside):
    """CenterNet focal loss with a one-cell Gaussian around each positive centre; negatives only where there is none."""
    q = logit.shape[0]
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)[:, 0]
    ys, xs = torch.meshgrid(torch.arange(WIN, device=logit.device), torch.arange(WIN, device=logit.device), indexing='ij')
    gauss = torch.zeros(q, WIN, WIN, device=logit.device)
    d2 = (xs[None] + .5 - pos[:, 0, None, None]) ** 2 + (ys[None] + .5 - pos[:, 1, None, None]) ** 2
    gauss = torch.where(inside[:, None, None], torch.exp(-d2 / 2), gauss)
    is_pos = torch.zeros_like(gauss, dtype=torch.bool)
    is_pos[inside.nonzero()[:, 0], j[inside], i[inside]] = True
    pos_loss = -(1 - p) ** 2 * torch.log(p) * is_pos
    neg_loss = -(1 - gauss) ** 4 * p ** 2 * torch.log(1 - p) * (~is_pos)
    return (pos_loss.sum() + neg_loss.sum()) / max(int(is_pos.sum()), 1)
