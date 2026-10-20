#!/usr/bin/env python3
"""Exp129 unit tests of the read-out head geometry and losses (no checkpoint, no images needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import exp128_dense_export  # noqa: F401  (puts src/lib on sys.path)
from datasets.dataset.jde import letterbox
from models import readout_head as R


def test_letterbox_params():
    for shape in ((428, 812), (1080, 1920), (608, 1088), (720, 1280)):
        img = np.zeros((*shape, 3), np.uint8)
        out, ratio, dw, dh = letterbox(img, height=R.IMG_H, width=R.IMG_W)
        r, left, top = R.letterbox_params(*shape)
        assert out.shape[:2] == (R.IMG_H, R.IMG_W) and abs(r - ratio) < 1e-12
        assert left == round(dw - 0.1) and top == round(dh - 0.1), (shape, left, top, dw, dh)


def test_shapes_and_budget():
    head = R.ReadoutHead()
    assert R.n_parameters(head) <= 1_000_000, R.n_parameters(head)
    q = 5
    args = (torch.randn(q, 64, 7, 7), torch.nn.functional.normalize(torch.randn(q, 128), dim=1), torch.randn(q, 64, 32, 32),
            torch.nn.functional.normalize(torch.randn(q, 128, 32, 32), dim=1), torch.randn(q, 2))
    for use in (True, False):
        out = head(*args, use_template=use)
        assert out['center'].shape == (q, 1, 32, 32) and out['box'].shape == (q, 4, 32, 32) and out['exist'].shape == (q,)
    zero = head(args[0] * 0, args[1] * 0, *args[2:], use_template=True)
    off = head(*args, use_template=False)
    assert torch.allclose(zero['exist'], off['exist'], atol=1e-5)      # no-template == zeroed template


def test_window_geometry():
    pred, s0 = torch.tensor([[200., 120.]]), torch.tensor([20.])
    roi = R.window_rois(torch.tensor([0]), pred, s0, torch.tensor([.5]), torch.tensor([3.]), torch.tensor([2.]))
    cx, cy = (roi[0, 1] + roi[0, 3]) / 2, (roi[0, 2] + roi[0, 4]) / 2
    assert abs(cx - (200 * .5 + 3)) < 1e-4 and abs(cy - (120 * .5 + 2)) < 1e-4
    assert abs((roi[0, 3] - roi[0, 1]) - R.WINDOW_SIDES * 20 * .5) < 1e-4


def test_target_decode_roundtrip():
    s0 = torch.tensor([20., 20., 20.])
    pred = torch.tensor([[100., 100.], [100., 100.], [100., 100.]])
    gc = pred + torch.tensor([[0., 0.], [30., -20.], [-45., 10.]])
    j, i, inside, pos = R.target_cell(gc, pred, s0)
    assert inside.tolist() == [True, True, True]
    logit = torch.full((3, 1, R.WIN, R.WIN), -9.)
    for k in range(3):
        logit[k, 0, j[k], i[k]] = 9.
    out = {'center': logit, 'box': torch.ones(3, 4, R.WIN, R.WIN) * .5, 'exist': torch.zeros(3)}
    p, box = R.decode(out, pred, s0)
    cell = R.WINDOW_SIDES * 20 / R.WIN
    centre = (box[:, :2] + box[:, 2:]) / 2
    assert (centre - gc).abs().max() <= cell / 2 + 1e-4, (centre, gc)
    assert torch.allclose(p, torch.full((3,), .5))
    far = pred[:1] + torch.tensor([[70., 0.]])
    assert not R.target_cell(far, pred[:1], s0[:1])[2].item()    # 3.5 sides away is outside the INSIDE region


def test_focal():
    j = torch.tensor([3, 5]); i = torch.tensor([4, 6])
    logit = torch.full((2, 1, R.WIN, R.WIN), -3.)
    pos = torch.tensor([[4.4, 3.6], [6.2, 5.1]])
    inside = torch.tensor([True, False])
    loss = R.focal_center_loss(logit, j, i, pos, inside)
    assert torch.isfinite(loss) and loss > 0
    none = R.focal_center_loss(logit, j, i, pos, torch.tensor([False, False]))
    assert torch.isfinite(none)
    good = logit.clone(); good[0, 0, 3, 4] = 6.
    assert R.focal_center_loss(good, j, i, pos, inside) < loss


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('exp129 fork tests OK')
