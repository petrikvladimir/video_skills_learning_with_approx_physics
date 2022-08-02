#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-01-31
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import torch
import tqdm
import numpy as np
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, so3_rotation_angle, so3_exp_map, \
    axis_angle_to_matrix, RotateAxisAngle, so3_relative_angle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffeq.distances import sdf_to_box_batch
from diffeq.simulation_renderer import SimulationRendererCuboidsOnPlaneWithHand
from utils.loss import visibility_diou_loss, loss_pos_smooth, loss_rot_smooth
from utils.data_loader import DataLoaderWrapper
from utils.plot import render_frames
from utils.params import get_perspective_camera, get_default_arg_parser
from utils.preprocessing import estimate_obj_position_in_image, estimate_hand_position_orientation_in_image

parser = get_default_arg_parser()
parser.add_argument('-iterations', type=int, default=1000)
parser.add_argument('-wiou', type=float, default=1.)
parser.add_argument('-wtsmooth', type=float, default=1e3)
parser.add_argument('-wrsmooth', type=float, default=1e6)
parser.add_argument('-wsize', type=float, default=1e1)
parser.add_argument('-winteraction', type=float, default=1e3)

options = parser.parse_args()
torch.manual_seed(options.seed)

vid = options.vid
print(f'Processing: {vid}')
""" Loading and subsampling """
data = DataLoaderWrapper(vid, img_subsample=options.subsample_img, time_subsample=options.subsample_time,
                         prepare_segmentations=True, prepare_100doh=True, prepare_video=True)

""" General params"""
device = f'cuda:{options.cuda_id}' if torch.cuda.is_available() else 'cpu'
tparam = dict(device=device, dtype=torch.float32)
n = data.num_frames()
image_size = data.data_segmentations[0].shape[:2]
cameras = get_perspective_camera(device=device, image_size=image_size, batch_size=n)
in_contact = data.in_contact_with_fix_start_end.to(device)
not_in_contact = torch.bitwise_not(in_contact)
hand_masks = data.hand_masks().to(**tparam)
obj_masks = data.manipulated_obj_masks().to(**tparam)

""" Compute initialization """
hpos_img, hrot_img = estimate_hand_position_orientation_in_image(data.data_100doh, hand_masks, device=device)
ns = torch.arange(0, n, 1)
hpos_init = RotateAxisAngle(180, axis='Z', **tparam).transform_points(
    cameras.unproject_points(torch.cat([hpos_img, 0.5 * torch.ones(n, 1, **tparam)], dim=-1))[ns, ns]
)
hrot_init = so3_exp_map(torch.tensor([0., -90., 0], **tparam).deg2rad().unsqueeze(0)).repeat(n, 1, 1)
hrot_init = so3_exp_map(torch.cat((torch.zeros(n, 2, **tparam), -hrot_img.unsqueeze(-1)), dim=-1)).bmm(hrot_init)
obj_center_img_depth = torch.empty(n, 3, **tparam)
obj_center_img_depth[:, :2] = estimate_obj_position_in_image(data.data_100doh, device)
obj_center_img_depth[:, 2] = hpos_init[:, 2].to(**tparam)
opos_init = RotateAxisAngle(180, axis='Z', **tparam).transform_points(
    cameras.unproject_points(obj_center_img_depth)[ns, ns])

""" State space definition """
opos = opos_init.to(**tparam, copy=True).requires_grad_()
orot_z = torch.zeros(n, **tparam).requires_grad_()
orot_xy = torch.zeros(n, 2, **tparam)
osize_sqrt = (np.sqrt(0.05) * torch.ones(3, **tparam)).requires_grad_()

hpos = hpos_init.to(**tparam, copy=True).requires_grad_()
hrot6d = matrix_to_rotation_6d(hrot_init).to(**tparam, copy=True).requires_grad_()

""" Define renderer """
cam_pos = torch.zeros(n, 3, **tparam)
cam_rot = torch.eye(3, **tparam).unsqueeze(0).expand(n, 3, 3)

renderer = SimulationRendererCuboidsOnPlaneWithHand(
    data, cameras, image_size=image_size, render_static_object=False, **tparam
)

optimizer = torch.optim.Adam([opos, orot_z, osize_sqrt, hpos, hrot6d], lr=1e-2, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

tr = tqdm.trange(options.iterations)
wiou, wtsmooth, wrsmooth, wsize = options.wiou, options.wtsmooth, options.wrsmooth, options.wsize
winteraction = options.winteraction
for i in tr:
    if optimizer.param_groups[0]["lr"] < 1e-5:
        break
    optimizer.zero_grad()

    osize = osize_sqrt.square()
    orot = axis_angle_to_matrix(torch.cat((orot_xy, orot_z.view(n, 1)), dim=-1))
    hrot = rotation_6d_to_matrix(hrot6d)

    imgs = renderer.render(opos, orot, osize, hpos, hrot, cam_pos, cam_rot)

    loss_hand_iou = wiou * visibility_diou_loss(hand_masks, renderer.hand_prob(imgs)).mean()
    loss_obj_iou = wiou * visibility_diou_loss(obj_masks, renderer.manipulated_obj_prob(imgs)).mean()

    loss_smooth_t = wtsmooth * (loss_pos_smooth(opos) + loss_pos_smooth(hpos))
    loss_smooth_r = wrsmooth * (loss_rot_smooth(orot) + loss_rot_smooth(hrot))

    loss_hand_rotation_reg = 1e-5 * (1 - so3_rotation_angle(hrot[:1], cos_angle=True)).square().mean()
    loss_obj_size = wsize * torch.min((osize[:2] - osize[2]).square(), dim=-1)[0]

    dist_in_contact = sdf_to_box_batch(hpos[in_contact], opos[in_contact], orot[in_contact], osize.view(1, 3))
    loss_obj_hand_interaction = winteraction * dist_in_contact.clamp_min(0.).square().mean()
    if not_in_contact[:-1].sum() > 0:
        v = (opos[:-1] - opos[1:])
        w = (1 - so3_relative_angle(orot[:-1], orot[1:], cos_angle=True, eps=1e-4))
        loss_obj_hand_interaction = loss_obj_hand_interaction + \
                                    1e3 * wtsmooth * v[not_in_contact[:-1]].square().mean() + \
                                    1e3 * wrsmooth * w[not_in_contact[:-1]].square().mean()

    loss = loss_obj_iou + loss_hand_iou + loss_smooth_t + loss_smooth_r + loss_hand_rotation_reg \
           + loss_obj_size + loss_obj_hand_interaction

    tr.set_description(
        f'LR: {optimizer.param_groups[0]["lr"]}; '
        f'L: {loss.item():.2e}; '
        f'PercO.: {loss_obj_iou:.2e}; '
        f'PercH.: {loss_hand_iou:.2e}; '
        f'Intera.: {loss_obj_hand_interaction:.2e}; '
    )
    loss.backward()
    optimizer.step()
    scheduler.step(loss.detach())

""" Store the data """
osize = osize_sqrt.square()
orot = axis_angle_to_matrix(torch.cat((orot_xy, orot_z.view(n, 1)), dim=-1))
hrot = rotation_6d_to_matrix(hrot6d)
imgs = renderer.render(opos, orot, osize, hpos, hrot, cam_pos, cam_rot).detach().cpu()
timg = renderer.render_top(opos, orot, osize, hpos, hrot, in_camera=True).detach().cpu()


def frame_me(img):
    if isinstance(img, torch.Tensor):
        img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    cameras.T = cam_pos
    cameras.R = cam_rot
    render_frames(cameras.cpu(), img, hpos.detach().cpu(), hrot.detach().cpu())
    render_frames(cameras.cpu(), img, opos.detach().cpu(), orot.detach().cpu())
    return img


data.write_obj_hand_initialization(
    obj_pos=opos.detach(), obj_rot=orot.detach(), obj_size=osize.detach(),
    hand_pos=hpos.detach(), hand_rot=hrot.detach(),
    images_dict={
        'render': frame_me(imgs),
        'top_view': (timg.detach().cpu().numpy() * 255).astype(np.uint8),
        'segmentation': frame_me(data.data_segmentations.astype(np.uint8)),
        'video': frame_me(data.data_video.astype(np.uint8)),
    }
)
