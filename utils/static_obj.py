#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffeq.simulation_renderer import SimulationRendererCuboidsOnPlaneWithHand
from utils.loss import visibility_diou_loss


def static_obj_optimization(renderer: SimulationRendererCuboidsOnPlaneWithHand,
                            ref_hpos, ref_hrot, ref_opos, ref_orot, ref_osize,
                            ref_hand_masks, ref_obj_masks, ref_sobj_masks,
                            init_sp, init_sz, sr, cam_pos=None, cam_rot=None, iterations=100, min_sz=0.02):
    n = ref_opos.shape[0]
    if cam_pos is None:
        cam_pos = torch.zeros(n, 3)
    if cam_rot is None:
        cam_rot = torch.eye(3).unsqueeze(0).expand(n, 3, 3)
    tparam = dict(dtype=ref_opos.dtype, device=ref_opos.device)
    sp = init_sp.to(**tparam, copy=True).requires_grad_()
    sz_sqrt = (init_sz - min_sz).sqrt().to(**tparam, copy=True).requires_grad_()
    tr = tqdm.trange(iterations)
    optimizer = torch.optim.Adam([sp, sz_sqrt], lr=1e-2, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    for i in tr:
        if optimizer.param_groups[0]["lr"] < 1e-5:
            break
        optimizer.zero_grad()
        sz = sz_sqrt.square() + min_sz
        imgs = renderer.render(ref_opos, ref_orot, ref_osize, ref_hpos, ref_hrot, cam_pos, cam_rot,
                               sp=sp.view(1, 1, 3).expand(n, 1, 3),
                               sr=sr.view(1, 1, 3, 3).expand(n, 1, 3, 3),
                               sz=sz.view(1, 1, 3).expand(n, 1, 3))

        loss_hand_iou = visibility_diou_loss(ref_hand_masks, renderer.hand_prob(imgs)).mean()
        loss_obj_iou = visibility_diou_loss(ref_obj_masks, renderer.manipulated_obj_prob(imgs)).mean()
        loss_sobj_iou = visibility_diou_loss(ref_sobj_masks, renderer.static_obj_prob(imgs)).mean()

        loss_depth_reg = 1. * (ref_opos[:, 2] - sp[2]).square().mean()

        loss = loss_hand_iou + loss_obj_iou + loss_sobj_iou + loss_depth_reg
        loss.backward()
        tr.set_description(
            f'Static obj fitting. '
            f'LR: {optimizer.param_groups[0]["lr"]:.1e}; '
            f'{scheduler.num_bad_epochs}/{scheduler.patience} || '
            f'L: {loss.item():.4f}; '
            f'PercH: {loss_hand_iou:.4f}; '
            f'PercO: {loss_obj_iou:.4f}; '
            f'PercSO: {loss_sobj_iou:.4f}; '
            f'Dist: {loss_depth_reg:.4f}; '
        )
        optimizer.step()
        scheduler.step(loss)
    return sp.detach(), sz_sqrt.square().detach() + min_sz
