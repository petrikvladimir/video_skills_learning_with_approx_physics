#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-3
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
import torch
import tqdm
from pytorch3d.transforms import so3_relative_angle, so3_exp_map
from torch import ones

from diffeq.box import Box
from diffeq.simulation_renderer import SimulationRendererCuboidsOnPlaneWithHand
from diffeq.diff_renderer import CameraModelTzAndElevation
from diffeq.distances import sdf_to_box_batch
from diffeq.hand import HandSpline
from diffeq.hand_motion_eq import HandEquationOfMotion
from diffeq.simulation import SimulationCuboidsOnPlaneWithHand
from diffeq.utils import fix_rot
from utils.data_loader import DataLoaderWrapper
from utils.loss import visibility_diou_loss, mask_center, unproject_img_points, mask_bb
from utils.params import get_default_arg_parser, get_perspective_camera
from utils.plot import render_frames
from utils.preprocessing import estimate_is_object_initially_moving, estimate_is_object_moving_at_the_end
from utils.scheduler import ReduceLROnPlateauResetToBest
from utils.static_obj import static_obj_optimization

parser = get_default_arg_parser()
parser.add_argument('-iterations', type=int, default=250)
parser.add_argument('-hand_fit_iterations', type=int, default=50)
parser.add_argument('-static_obj_fit_iterations', type=int, default=1000)
parser.add_argument('-whiou', type=float, default=1.)
parser.add_argument('-woiou', type=float, default=10.)
parser.add_argument('-wsoiou', type=float, default=2.)
parser.add_argument('-prefix', type=str, default='')
parser.add_argument('--release_fix', dest='release_fix', action='store_true')

options = parser.parse_args()
torch.manual_seed(options.seed)

vid = options.vid
print(f'Processing: {vid}')
device = 'cpu'
tparam = dict(device=device, dtype=torch.float64)
render_device = f'cuda:{options.cuda_id}' if torch.cuda.is_available() and options.cuda_id >= 0 else 'cpu'
render_tparam = dict(device=render_device, dtype=torch.float32)

""" Loading and subsampling """
data = DataLoaderWrapper(vid, img_subsample=options.subsample_img, time_subsample=options.subsample_time,
                         prepare_segmentations=True, prepare_100doh=True, prepare_video=True,
                         physics_folder_prefix=options.prefix)
n = data.num_frames()
image_size = data.data_segmentations[0].shape[:2]
has_static = data.has_other_object()
ref_hand_masks = data.hand_masks().to(**render_tparam)
ref_obj_masks = data.manipulated_obj_masks().to(**render_tparam)
ref_sobj_masks = data.first_other_obj_masks().to(**render_tparam) if has_static else None
in_contact = data.in_contact_with_fix_start_end.to(tparam['device'])
if options.release_fix:
    in_contact = data.in_contact_with_fix_start_end_release.to(tparam['device'])
not_in_contact = torch.bitwise_not(in_contact)
has_no_contact = not_in_contact.sum() > 0
first_contact_ind = torch.where(in_contact)[0][0]
# last_contact_ind = torch.where(in_contact)[0][-1]
ref_opos, ref_orot, ref_osize, ref_hpos, ref_hrot = data.load_obj_hand_initialization(to_torch=True, **tparam)
ref_obj_masks_cpu = data.manipulated_obj_masks().to(**tparam)
is_initially_moving = estimate_is_object_initially_moving(data)
obj_not_detected = ref_obj_masks.sum((-2, -1)) == 0

""" Camera and renderer """
camera_transform = CameraModelTzAndElevation(elevation_deg=45., **tparam)
cameras = get_perspective_camera(device=render_device, image_size=image_size, batch_size=n)
renderer = SimulationRendererCuboidsOnPlaneWithHand(data, cameras=cameras, image_size=image_size, **render_tparam)

""" Static object initialization and camera transformation """
sp_w, sz = None, None
if has_static:
    static_visible = ref_sobj_masks.sum(dim=[-2, -1]) > 0
    imgx, imgy = mask_center(ref_sobj_masks[static_visible])
    img_points_xydepth = torch.stack([imgx.mean(), imgy.mean(), ref_opos[:, 2].mean().to(**render_tparam)], dim=-1)
    sp_cam = unproject_img_points(cameras=cameras, img_points_xydepth=img_points_xydepth.unsqueeze(0)).to(**tparam)[0]

    sr_cam = camera_transform.world2cam_rot.detach()  # so that it is identity in world
    sp_cam, sz = static_obj_optimization(
        renderer, ref_hpos=ref_hpos, ref_hrot=ref_hrot,
        ref_opos=ref_opos, ref_orot=ref_orot, ref_osize=ref_osize,
        ref_hand_masks=ref_hand_masks, ref_obj_masks=ref_obj_masks, ref_sobj_masks=ref_sobj_masks,
        init_sp=sp_cam, init_sz=(0.02 * torch.ones(3)).sqrt(), sr=sr_cam,
        iterations=options.static_obj_fit_iterations
    )
    sp_w = camera_transform.transform_cam2world(sp_cam.unsqueeze(0), sr_cam.unsqueeze(0), detach=True)[0][:1]
    sz = sz.unsqueeze(0)
    camera_transform.cam2world_tz.data[0] = -sp_w[0, 2] + sz[0, 2] / 2
    sp_w = camera_transform.transform_cam2world(sp_cam.unsqueeze(0), sr_cam.unsqueeze(0), detach=True)[0][:1]
else:
    box = Box()
    p, r = camera_transform.transform_cam2world(ref_opos, ref_orot, detach=True)
    p = box.q_s(r[first_contact_ind], ref_osize) + p[first_contact_ind]
    camera_transform.cam2world_tz.data[0] = -torch.min(p[:, 2]).detach()
    # p0 = torch.min(camera_transform.transform_cam2world(ref_opos, ref_orot, detach=True)[0][:, 2])
    # camera_transform.cam2world_tz.data[0] = -p0 + ref_osize[1] / 2

""" Transfer all the references into the world space. """
ref_hpos_w, ref_hrot_w = camera_transform.transform_cam2world(ref_hpos, ref_hrot, detach=True)
ref_opos_w, ref_orot_w = camera_transform.transform_cam2world(ref_opos, ref_orot, detach=True)
obj_pos_init = ref_hpos_w[0] if is_initially_moving else ref_opos_w[first_contact_ind]
obj_rot_init = so3_exp_map(torch.tensor([[camera_transform.elevation.detach(), 0., 0.]]))[0] @ ref_orot_w[0]

""" Initialize hand spline through equation of motion to follow world coords """
hand = HandSpline(**HandSpline.get_spline_initialization_from_motion(
    ref_hpos_w, ref_hrot_w, in_contact, fps=12 // options.subsample_time, spline_complexity=5,
))
hand_motion_eq = HandEquationOfMotion(hand)
hand_motion_eq.fit(ref_hpos_w, ref_hrot_w, in_contact, iterations=options.hand_fit_iterations)

""" Simulation  """
simulation = SimulationCuboidsOnPlaneWithHand(
    hand=hand,
    obj_pos0=obj_pos_init, obj_rot0=obj_rot_init, obj_size=ref_osize,
    number_of_static_objects=1 if has_static else 0, static_objs_pos0=sp_w, static_objs_size=sz,
)

""" What to optimize """
# Manipulated object
simulation.obj_pos0.requires_grad = True
simulation.obj_rot6d0.requires_grad = True
simulation.obj_size_sqrt.requires_grad = True
simulation.obj_vel0.requires_grad = False
simulation.obj_omega0.requires_grad = False
# Static object
if has_static:
    simulation.static_objs_pos0_xy.requires_grad = True
    simulation.static_objs_rot0z.requires_grad = True
    simulation.static_objs_size_sqrt.requires_grad = True
# Hand
simulation.hand.spline_y.requires_grad = True
simulation.hand.pos0.requires_grad = True
simulation.hand.rot6d0.requires_grad = True
# Camera
camera_transform.elevation.requires_grad = True
camera_transform.cam2world_tz.requires_grad = True

""" Loss Weights """
wgrasp = 1e4
whiou, woiou, wsoiou = options.whiou, options.woiou, options.wsoiou
whand_ang_velocity, whand_lin_velocity = 1e-1, 1e-1
wstability_pos, wstability_rot = 1e2, 1e2

""" Optimization loop """
timesteps = torch.cat([-ones(1), torch.linspace(0, simulation.max_t, n), (simulation.max_t + 1) * ones(1)]).to(**tparam)
tr = tqdm.trange(options.iterations)
optimizer = torch.optim.Adam(list(simulation.parameters()) + list(camera_transform.parameters()), lr=5e-3, amsgrad=True)
scheduler = ReduceLROnPlateauResetToBest(optimizer, factor=0.5, patience=5, models=[simulation, camera_transform],
                                         recover_nth_best=0)

for i in tr:
    if optimizer.param_groups[0]["lr"] < 1e-5:
        break
    optimizer.zero_grad()

    state = simulation.integrate_between_events(simulation.get_all_events(timesteps[0]), timesteps=timesteps)
    oposS, _, orotS, _, _, _, _, _ = [s[0] for s in state]
    oposE, _, orotE, _, _, _, _, _ = [s[-1] for s in state]
    opos, ovel, orot, oomega, ograsped, hpos, hrot, hgrasp = [s[1:-1] for s in state]
    hvel, homega, _ = simulation.hand.velocities(timesteps[1:-1])
    orotS, orotE, orot, hrot = fix_rot(orotS), fix_rot(orotE), fix_rot(orot), fix_rot(hrot)

    """ Loss stability start/end, i.e. object should not move before video and after video. """
    loss_stability_pos = wstability_pos * ((oposS - opos[0]).square().mean() + (oposE - opos[-1]).square().mean())
    loss_stability_rot = wstability_rot * (
            (1 - so3_relative_angle(orotS.unsqueeze(0), orot[0].unsqueeze(0), cos_angle=True)).square().mean() +
            (1 - so3_relative_angle(orotE.unsqueeze(0), orot[-1].unsqueeze(0), cos_angle=True)).square().mean()
    )
    if obj_not_detected.sum() > 0:
        loss_stability_invisible = wstability_pos * ovel[obj_not_detected].square().mean() + \
                                   wstability_pos * oomega[obj_not_detected].square().mean()
    else:
        loss_stability_invisible = 0
    loss_stability = loss_stability_rot + loss_stability_pos + loss_stability_invisible

    """ Grasp loss, i.e. hand close to object if in contact and hand closed if in contact """
    sdf = sdf_to_box_batch(hpos, opos, orot, simulation.obj_size.unsqueeze(0).expand(n, 3))
    gsig = torch.max(sdf, -hgrasp.squeeze(-1))  # negative iff grasped
    # loss_grasp = (wgrasp * gsig[in_contact].clamp_min(min=0.).square()).mean()
    # let's decompose the loss for grasp, so that gradient flow to griper and sdf simultaneously
    sdf_loss = (sdf[in_contact] + simulation.grasped_eps).clamp_min(0.).square().mean()  # penalize positive sdf
    hgrasp_loss = hgrasp[in_contact].clamp_max(0.).square().mean()  # penalize neg hg in contact
    if has_no_contact:
        hgrasp_loss = hgrasp_loss + hgrasp[not_in_contact].clamp_min(0.).square().mean()  # penalize pos not in contact
    loss_grasp = wgrasp * (sdf_loss + hgrasp_loss)

    """ Perceptual loss """
    cam_pos, cam_rot = camera_transform.cam_pos_rot(n)
    imgs = renderer.render_sim(simulation, opos, orot, hpos, hrot, cam_pos, cam_rot)
    loss_perceptual_obj = woiou * visibility_diou_loss(ref_obj_masks, renderer.manipulated_obj_prob(imgs)).mean()
    loss_perceptual_hand = whiou * visibility_diou_loss(ref_hand_masks, renderer.hand_prob(imgs)).mean()
    if has_static:
        loss_perceptual_static = wsoiou * visibility_diou_loss(ref_sobj_masks, renderer.static_obj_prob(imgs)).mean()
    else:
        loss_perceptual_static = 0
    loss_perceptual = (loss_perceptual_obj + loss_perceptual_hand + loss_perceptual_static).to(**tparam)

    """ Other regularization """
    reg_obj_start = 1e-1 * (simulation.obj_pos0 - opos[0]).square().mean()
    loss_hand_vel = whand_ang_velocity * homega.square().mean() + whand_lin_velocity * hvel.square().mean()

    loss = loss_grasp + loss_perceptual + loss_stability + loss_hand_vel + reg_obj_start
    if scheduler.update_best_if_lower(loss):
        # timgs = renderer.render_sim_top(simulation, opos, orot, hpos, hrot)
        data.write_physics_state(
            images_dict={
                f'best_render': (imgs.detach().cpu() * 255).numpy().astype(np.uint8),
                # f'best_top': (timgs.detach().cpu() * 255).numpy().astype(np.uint8),
            }
        )

    loss.backward()

    tr.set_description(
        f'LR: {optimizer.param_groups[0]["lr"]:.1e}; '
        f'{scheduler.num_bad_epochs}/{scheduler.patience} || '
        f'L: {loss.item():.4f} / {scheduler._best_loss:.4f}; '
        f'PercO: {loss_perceptual_obj:.4f}; '
        f'PercSO: {loss_perceptual_static:.4f}; '
        f'PercH: {loss_perceptual_hand:.4f}; '
        f'Grasp: {loss_grasp:.4f}; '
        f'Grasps: {ograsped[0, 0].detach() > 0}; {torch.max(ograsped).detach() > 0}; {ograsped[-1, 0].detach() > 0}; '
        f'StT: {loss_stability_pos:.4f}; '
        f'StR: {loss_stability_rot:.4f}; '
        f'StIn: {loss_stability_invisible:.4f}; '
    )

    optimizer.step()
    scheduler.step(loss)

""" Rendering of the results """
scheduler.recover_best(recover_nth_best=0)
state = simulation.integrate_between_events(simulation.get_all_events(timesteps[0]), timesteps=timesteps)
opos, ovel, orot, oomega, ograsped, hpos, hrot, hgrasp = [s[1:-1] for s in state]

cam_pos, cam_rot = camera_transform.cam_pos_rot(n)
imgs = renderer.render_sim(simulation, opos, orot, hpos, hrot, cam_pos, cam_rot).detach().cpu()
timgs = renderer.render_sim_top(simulation, opos, orot, hpos, hrot).detach().cpu()

cameras.T, cameras.R = cam_pos.to(**render_tparam), cam_rot.to(**render_tparam)


def frame_me(img):
    if isinstance(img, torch.Tensor):
        img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    t = render_tparam['dtype']
    render_frames(cameras.cpu(), img, hpos.detach().cpu().to(dtype=t), hrot.detach().cpu().to(dtype=t))
    render_frames(cameras.cpu(), img, opos.detach().cpu().to(dtype=t), orot.detach().cpu().to(dtype=t))
    return img


data.write_physics_state(
    simulation=simulation, camera_transform=camera_transform,
    obj_pos=opos.detach().cpu(), obj_rot=orot.detach().cpu(), obj_grasped=ograsped.detach().cpu(),
    hand_pos=hpos.detach().cpu(), hand_rot=hrot.detach().cpu(),
    images_dict={
        'top_view': (timgs.detach().cpu() * 255).numpy().astype(np.uint8),
        'render': frame_me(imgs),
        'segmentation': frame_me(data.data_segmentations.astype(np.uint8)),
        'video': frame_me(data.data_video.astype(np.uint8)),
    }
)
