"""Exp137: the v3 read-out head on a frozen YOLOX detector (V1 YOLOX-S 1088, Exp088 epoch 50).

Same head as Exp134 (template correlation, fine level, embedding cosine, pixel branch over the window and the template; centre,
offset, ltrb and existence). Only the detector side changes:
  - shared features: the PAFPN stride-8 output (128 channels for YOLOX-S) instead of FairMOT's stride-4 map;
  - fine level: the CSPDarknet `dark2` output at stride 4 (64 channels) instead of the DLA stride-2 level;
  - embedding: YOLOX has no Re-ID branch, so the cosine is taken on the L2-normalised stride-8 features themselves;
  - input: YOLOX preprocessing (resize by r = min(1088 / h, 1088 / w), pad bottom-right with 114, RGB, ImageNet mean / std);
    the pixel branch reads the same network input, de-normalised to [0, 1].
The functions keep the signatures of readout_head_v2 / v3 so the training and closed-loop code can use either detector.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from models import readout_head as R
from models import readout_head_v2 as V2
from models import readout_head_v3 as V3

SIZE = (1088, 1088)
FEAT_STRIDE, FINE_STRIDE = 8, 4
FEAT_CH, FINE_CH = 128, 64
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WIN, TMPL, PIX, TPIX = V3.WIN, V2.TMPL, V3.PIX, V3.TPIX
decode = V2.decode
target_cell = V2.target_cell
focal_center_loss = V2.focal_center_loss


def ReadoutHeadV3():
    return V3.ReadoutHeadV3(feat_ch=FEAT_CH, fine_ch=FINE_CH)


def load_model(bytetrack_root, exp_file, ckpt, device):
    """Frozen YOLOX from a ByteTrack checkout or export; a forward hook keeps the backbone's dark2 output of the last call."""
    root = str(Path(bytetrack_root).resolve())
    if root not in sys.path:
        sys.path.append(root)  # appended: FairMOT's own `datasets` / `models` packages keep precedence
    from yolox.exp import get_exp
    exp = get_exp(str(Path(root) / exp_file), None)
    assert tuple(exp.input_size) == SIZE, exp.input_size
    model = exp.get_model()
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    grab = {}
    model.backbone.backbone.dark2.register_forward_hook(lambda m, i, o: grab.__setitem__('dark2', o))
    model._readout_grab = grab
    return model


def load_frame(seq, f):
    """YOLOX network input of frame f (float32 CHW) and (ratio, left, top) = (r, 0, 0)."""
    img0 = cv2.imread(str(seq.dir / 'img1' / f'{f:06d}.jpg'))
    assert img0 is not None, f'missing frame {f} of {seq.dir.name}'
    r = min(SIZE[0] / img0.shape[0], SIZE[1] / img0.shape[1])
    padded = np.full((SIZE[0], SIZE[1], 3), 114.0, np.float32)
    h, w = int(img0.shape[0] * r), int(img0.shape[1] * r)
    padded[:h, :w] = cv2.resize(img0, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (padded[:, :, ::-1] / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32), (r, 0.0, 0.0)


@torch.no_grad()
def backbone_forward(model, x):
    """(stride-8 PAFPN features, the same map as embedding source, dark2 at stride 4)."""
    fpn = model.backbone(x)
    feat = fpn[0]
    return feat, feat, model._readout_grab['dark2']


def _unit(img):
    m = torch.tensor(MEAN, device=img.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=img.device).view(1, 3, 1, 1)
    return img * s + m


def read_inputs(feat_t, id_t, fine_t, img_t, idx_t, feat_a, id_a, img_a, idx_a, tmpl_boxes, pred, s0,
                ratio_t, left_t, top_t, ratio_a, left_a, top_a):
    q_win = R.window_rois(idx_t, pred, s0, ratio_t, left_t, top_t)
    win_feat = roi_align(feat_t, q_win, (WIN, WIN), spatial_scale=1 / FEAT_STRIDE, sampling_ratio=2, aligned=True)
    win_fine = roi_align(fine_t, q_win, (WIN, WIN), spatial_scale=1 / FINE_STRIDE, sampling_ratio=2, aligned=True)
    win_emb = F.normalize(roi_align(id_t, q_win, (WIN, WIN), spatial_scale=1 / FEAT_STRIDE, sampling_ratio=2, aligned=True), dim=1)
    q_t = R.template_rois(idx_a, tmpl_boxes, ratio_a, left_a, top_a)
    tmpl_feat = roi_align(feat_a, q_t, (TMPL, TMPL), spatial_scale=1 / FEAT_STRIDE, sampling_ratio=2, aligned=True)
    cx, cy = (q_t[:, 1] + q_t[:, 3]) / 2, (q_t[:, 2] + q_t[:, 4]) / 2
    q_c = R._rois(idx_a, cx - 4, cy - 4, cx + 4, cy + 4)
    tmpl_emb = F.normalize(roi_align(id_a, q_c, (1, 1), spatial_scale=1 / FEAT_STRIDE, sampling_ratio=2, aligned=True)[:, :, 0, 0], dim=1)
    win_rgb = roi_align(_unit(img_t), q_win, (PIX, PIX), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    hw = torch.maximum(q_t[:, 3] - q_t[:, 1], q_t[:, 4] - q_t[:, 2]) * V3.TMPL_PIX_SCALE / 2
    q_tp = R._rois(idx_a, cx - hw, cy - hw, cx + hw, cy + hw)
    tmpl_rgb = roi_align(_unit(img_a), q_tp, (TPIX, TPIX), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return tmpl_feat, tmpl_emb, win_feat, win_fine, win_emb, win_rgb, tmpl_rgb
