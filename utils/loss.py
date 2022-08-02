#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-01-27
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from packaging import version
from pytorch3d.transforms import so3_relative_angle, RotateAxisAngle
from torch import Tensor
from typing import Tuple, Optional
import torch


def _get_xx_yy(mask):
    H, W = mask.shape[-2:]
    if version.parse(torch.__version__) > version.parse('1.10.0'):
        yy, xx = torch.meshgrid(torch.arange(H).to(mask), torch.arange(W).to(mask), indexing='ij')
    else:
        yy, xx = torch.meshgrid(torch.arange(H).to(mask), torch.arange(W).to(mask))
    return xx, yy


def mask_center(mask: Tensor, xx: Optional[Tensor] = None, yy: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if xx is None or yy is None:
        xx, yy = _get_xx_yy(mask)
    center_x = (mask * xx).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))
    center_y = (mask * yy).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))
    return center_x, center_y


def mask_width_height(mask: Tensor) -> Tuple[Tensor, Tensor]:
    w = mask.sum(dim=-1).max(dim=-1).values
    h = mask.sum(dim=-2).max(dim=-1).values
    return w, h


def mask_bb(mask: Tensor, cx: Tensor, cy: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ Return bounding box of the mask in a form TLx, TLy, BRx, BRy"""
    w, h = mask_width_height(mask)
    tlx = cx - w / 2
    brx = cx + w / 2
    tly = cy - h / 2
    bry = cy + h / 2
    return tlx, tly, brx, bry


def iou(a: Tensor, b: Tensor) -> Tensor:
    """ Compute Intersection of Unions """
    intersection = a * b
    union = a + b - intersection
    iou = intersection.sum([-2, -1]) / union.sum([-2, -1])
    return iou


def diou_loss(a: Tensor, b: Tensor, xx: Optional[Tensor] = None, yy: Optional[Tensor] = None):
    """ Compute 1 - iou + normalized centroids distance between two masks in a form [..., HxW]. """
    if xx is None or yy is None:
        xx, yy = _get_xx_yy(a)

    a_xc, a_yc = mask_center(a, xx, yy)
    b_xc, b_yc = mask_center(b, xx, yy)
    a_tlx, a_tly, a_brx, a_bry = mask_bb(a, a_xc, a_yc)
    b_tlx, b_tly, b_brx, b_bry = mask_bb(b, b_xc, b_yc)

    tlx = torch.min(a_tlx, b_tlx)
    tly = torch.min(a_tly, b_tly)
    brx = torch.max(a_brx, b_brx)
    bry = torch.max(a_bry, b_bry)

    d = (a_xc - b_xc).square() + (a_yc - b_yc).square()
    c = (brx - tlx).square() + (bry - tly).square() + 1e-7
    return 1 - iou(a, b) + d / c


def visibility_diou_loss(a: Tensor, b: Tensor, xx: Optional[Tensor] = None, yy: Optional[Tensor] = None):
    a_visible = a.sum([-2, -1]) > 0
    b_visible = b.sum([-2, -1]) > 0
    visible = torch.bitwise_and(a_visible, b_visible)
    loss = torch.zeros(*a.shape[:-2]).to(a)
    a_invisible = torch.bitwise_not(a_visible)
    b_invisible = torch.bitwise_not(b_visible)
    b_avg_area = b[b_visible].sum([-2, -1]).mean().detach()
    a_avg_area = a[a_visible].sum([-2, -1]).mean().detach()
    loss[a_invisible] = loss[a_invisible] + b[a_invisible].sum([-2, -1]) / b_avg_area
    loss[b_invisible] = loss[b_invisible] + a[b_invisible].sum([-2, -1]) / a_avg_area
    loss[visible] = loss[visible] + diou_loss(a[visible], b[visible], xx, yy)
    return loss


def loss_rot_smooth(rot, eps=1e-4):
    return (1 - so3_relative_angle(rot[:-1], rot[1:], cos_angle=True, eps=eps)).square().mean()


def loss_pos_smooth(pos):
    return (pos[:-1] - pos[1:]).square().mean()


def unproject_img_points(cameras, img_points_xydepth):
    ns = torch.arange(0, img_points_xydepth.shape[0], 1)
    r = RotateAxisAngle(180, axis='Z', device=img_points_xydepth.device, dtype=img_points_xydepth.dtype)
    return r.transform_points(cameras.unproject_points(img_points_xydepth)[ns, ns])
