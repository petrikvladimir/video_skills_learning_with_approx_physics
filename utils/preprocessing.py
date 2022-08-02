#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-01-19
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
import pytorch3d
import torch
import tqdm
from packaging import version
from pytorch3d.transforms import rotation_6d_to_matrix, so3_rotation_angle

from utils.loss import mask_center, mask_width_height
from utils.params import get_perspective_camera


def estimate_dominant_hand(data_frankmocap):
    """ Return 'left_hand' or 'right_hand' based on the number of hands detections. """
    smpl_detected_l_hand = [bool(d['pred_output_list'][0]['left_hand']) for d in data_frankmocap]
    smpl_detected_r_hand = [bool(d['pred_output_list'][0]['right_hand']) for d in data_frankmocap]
    dominant_hand = 'left_hand' if sum(smpl_detected_l_hand) > sum(smpl_detected_r_hand) else 'right_hand'
    return dominant_hand


def estimate_dominant_hand_points_in_camera(data_frankmocap, image_size, num_iterations=5000,
                                            w_project=1e-3, w_translation_smooth=1., w_rotation_smooth=1.):
    """
    From frankmocap data, estimate hand pose by optimizing via neural renderer for fixed focal length.
    :param data_frankmocap:
    :param image_size:
    :param num_iterations:
    :param w_project:
    :param w_translation_smooth:
    :param w_rotation_smooth:
    :return: hand keypoints in the camera frame
    """
    dominant_hand = estimate_dominant_hand(data_frankmocap)
    smpl_detected_hand = [bool(d['pred_output_list'][0][dominant_hand]) for d in data_frankmocap]
    smpl_shape_predictions = np.array(
        [d['pred_output_list'][0][dominant_hand].get('pred_joints_smpl') for d in data_frankmocap], dtype=object
    )
    smpl_image_predictions = np.array(
        [d['pred_output_list'][0][dominant_hand].get('pred_joints_img') for d in data_frankmocap], dtype=object
    )

    # Optimize camera pose
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    rot_6d = torch.tensor(data=np.array([[1, 0, 0, 0, 1, 0]]).repeat(len(smpl_detected_hand), axis=0),
                          requires_grad=True, dtype=torch.float32, device=device)
    t = torch.tensor(np.random.normal(loc=[0, 0, -0.25], scale=0.01, size=(len(smpl_detected_hand), 3)),
                     requires_grad=True, dtype=torch.float32, device=device)
    points = torch.from_numpy(np.stack(np.asarray(smpl_shape_predictions[smpl_detected_hand])))
    projected_points_ref = torch.from_numpy(np.stack(smpl_image_predictions[smpl_detected_hand]))

    nframes = points.shape[0]

    camera = get_perspective_camera(device, image_size, batch_size=nframes)
    optimizer = torch.optim.Adam([rot_6d, t], lr=3e-4)
    tbar = tqdm.trange(num_iterations)
    for _ in tbar:
        optimizer.zero_grad()
        rot = rotation_6d_to_matrix(rot_6d)
        projected_points = camera.transform_points_screen(
            points, image_size=[image_size] * nframes, R=rot[smpl_detected_hand], T=t[smpl_detected_hand],
        )
        l_projection = (projected_points[:, :, :2] - projected_points_ref[:, :, :2]).square().mean()
        l_translation_smoothness = (t[1:] - t[:-1]).square().mean()
        l_rotation_smoothness = (1 - so3_rotation_angle(torch.transpose(rot[:-1], -2, -1).bmm(rot[1:]),
                                                        cos_angle=True)).square().mean()

        loss = w_project * l_projection + \
               w_translation_smooth * l_translation_smoothness + \
               w_rotation_smooth * l_rotation_smoothness
        tbar.set_postfix_str(f'loss: {loss.item()}')
        loss.backward()
        optimizer.step()

    hand_poses = pytorch3d.transforms.Rotate(rotation_6d_to_matrix(rot_6d)).compose(
        pytorch3d.transforms.Translate(t)
    ).inverse()

    # interpolate/extrapolate hand shapes
    n = len(smpl_detected_hand)
    interpolated_points = torch.empty(n, points.shape[-2], points.shape[-1])
    xp = np.where(smpl_detected_hand)[0]
    x = np.arange(0, n, 1)
    for i in range(points.shape[-2]):
        for j in range(points.shape[-1]):
            interpolated_points[:, i, j] = torch.from_numpy(np.interp(x, xp=xp, fp=points[:, i, j].numpy()))

    return hand_poses.transform_points(interpolated_points)


def estimate_hand_position_orientation_in_image(data_100doh, hand_masks, device=None, interpolate=True):
    """
    From 100DOH and hand masks, estimate center of the hand (center of BB from DOH) and rotation of the arm.
    :return: [interpolated] hand center, [interpolated] hand orientation
    """
    n = len(data_100doh)
    image_size = hand_masks[0].shape[:2]
    if device is None:
        device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    hand_detected = torch.tensor([d['hand'] is not None for d in data_100doh]).to(device)
    hand_bb_top_left = float('nan') * torch.zeros(n, 2).to(device)
    hand_bb_bottom_right = float('nan') * torch.zeros(n, 2).to(device)
    for i in torch.where(hand_detected)[0]:
        hand_bb_top_left[i] = torch.from_numpy(data_100doh[i]['hand'][0, 0:2]).to(device)
        hand_bb_bottom_right[i] = torch.from_numpy(data_100doh[i]['hand'][0, 2:4]).to(device)

    if version.parse(torch.__version__) > version.parse('1.10.0'):
        yy, xx = torch.meshgrid(torch.arange(image_size[0]).to(device), torch.arange(image_size[1]).to(device),
                                indexing='ij')
    else:
        yy, xx = torch.meshgrid(torch.arange(image_size[0]).to(device), torch.arange(image_size[1]).to(device))
    hand_center_img = torch.mean(torch.stack((hand_bb_top_left, hand_bb_bottom_right)), dim=0)

    hand_rot_img = float('nan') * torch.zeros(n).to(device)
    for i in torch.where(hand_detected)[0]:
        # if hand bb is close to image boudary, then estimation is probably wrong, let's not compute it.
        k = 2
        if hand_bb_top_left[i, 0] < k or hand_bb_top_left[i, 1] < k:
            continue
        if hand_bb_bottom_right[i, 1] > (image_size[0] - k) or hand_bb_bottom_right[i, 0] > (image_size[1] - k):
            continue
        seg = hand_masks[i].clone()
        seg[
        torch.floor(hand_bb_top_left[i, 1]).int():torch.ceil(hand_bb_bottom_right[i, 1]).int(),
        torch.floor(hand_bb_top_left[i, 0]).int():torch.ceil(hand_bb_bottom_right[i, 0]).int()
        ] = 0
        dist = seg * ((xx - hand_center_img[i, 0]).square() + (yy - hand_center_img[i, 1]).square())
        m = torch.max(dist)
        if m > 0.:  # zero only if not detected
            y, x = torch.nonzero(dist == m, as_tuple=True)
            if x.shape[0] > 1:
                assert 'more then one maximum detected'
            hand_rot_img[i] = torch.atan2(-(y - hand_center_img[i, 1]), (x - hand_center_img[i, 0]))  # -y

    if not interpolate:
        return hand_center_img, hand_rot_img

    # interpolate hand center
    xspan = np.arange(0, n, 1)
    xp = torch.where(hand_detected)[0].cpu()
    hand_center_img[:, 0] = torch.from_numpy(np.interp(xspan, xp=xp, fp=hand_center_img[xp, 0].cpu())).to(device)
    hand_center_img[:, 1] = torch.from_numpy(np.interp(xspan, xp=xp, fp=hand_center_img[xp, 1].cpu())).to(device)

    # interpolate hand rot
    if torch.all(torch.isnan(hand_rot_img)):
        from_left = (hand_bb_top_left[xp, 0] < (image_size[1] - hand_bb_bottom_right[xp, 0])).sum() > xp.shape[0] / 2
        hand_rot_img[:] = np.pi if from_left else 0.
        return hand_center_img, hand_rot_img

    xp = torch.where(torch.bitwise_not(torch.isnan(hand_rot_img)))[0].cpu()
    m = torch.atan2(hand_rot_img[xp].sin().sum(), hand_rot_img[xp].cos().sum())  # use mean for extrapolation
    hand_rot_img[:] = m
    # hand_rot_img[:] = torch.from_numpy(np.interp(xspan, xp=xp, fp=hand_rot_img[xp].cpu(), left=m, right=m)).to(device)

    return hand_center_img, hand_rot_img


def estimate_obj_position_in_image(data_100doh, device, interpolate=True):
    """ From 100DOH estimate object positions in image and interpolate it. """
    n = len(data_100doh)
    obj_detected = torch.tensor([d['obj'] is not None for d in data_100doh]).to(device)
    obj_bb_top_left = float('nan') * torch.zeros(n, 2).to(device)
    obj_bb_bottom_right = float('nan') * torch.zeros(n, 2).to(device)
    for i in torch.where(obj_detected)[0]:
        obj_bb_top_left[i] = torch.from_numpy(data_100doh[i]['obj'][0, 0:2]).to(device)
        obj_bb_bottom_right[i] = torch.from_numpy(data_100doh[i]['obj'][0, 2:4]).to(device)
    obj_center_img = torch.mean(torch.stack((obj_bb_top_left, obj_bb_bottom_right)), dim=0)

    if not interpolate:
        return obj_center_img

    xspan = np.arange(0, n, 1)
    xp = torch.where(obj_detected)[0].cpu()
    obj_center_img[:, 0] = torch.from_numpy(np.interp(xspan, xp=xp, fp=obj_center_img[xp, 0].cpu())).to(device)
    obj_center_img[:, 1] = torch.from_numpy(np.interp(xspan, xp=xp, fp=obj_center_img[xp, 1].cpu())).to(device)
    return obj_center_img


def estimate_colors(data_100doh, data_segmentations, fixed=True):
    """
    This method estimates the color of the hand, manipulated objects and other objects in the scene. It uses 100DOH
    estimations to predict where which object can be and returns mode of the colors.
    :param data_100doh:
    :param data_segmentation:
    :return: hand_color [4], manipulated_object_color [4], other_objects_colors [Nx4]
    """
    seg = torch.from_numpy(data_segmentations).clone()
    bg_color = torch.tensor([0, 0, 0, 255], dtype=seg.dtype)
    if fixed:
        k = 8
        colors = torch.unique(seg[:, ::k, ::k].reshape(-1, 4), dim=0)
        if colors.shape[0] == 3:
            assert torch.any(torch.all(colors == torch.tensor([0, 128, 0, 255], dtype=seg.dtype), dim=-1))
            assert torch.any(torch.all(colors == torch.tensor([128, 0, 0, 255], dtype=seg.dtype), dim=-1))
            return torch.tensor([0, 128, 0, 255], dtype=seg.dtype), \
                   torch.tensor([128, 0, 0, 255], dtype=seg.dtype), \
                   torch.zeros(0, 4, dtype=seg.dtype)
        elif colors.shape[0] == 4:
            assert torch.any(torch.all(colors == torch.tensor([0, 128, 0, 255], dtype=seg.dtype), dim=-1))
            assert torch.any(torch.all(colors == torch.tensor([128, 0, 0, 255], dtype=seg.dtype), dim=-1))
            assert torch.any(torch.all(colors == torch.tensor([128, 128, 0, 255], dtype=seg.dtype), dim=-1))
            return torch.tensor([128, 128, 0, 255], dtype=seg.dtype), \
                   torch.tensor([0, 128, 0, 255], dtype=seg.dtype), \
                   torch.tensor([[128, 0, 0, 255]], dtype=seg.dtype)
        else:
            raise NotImplementedError('Not supported segmentation mask.')

    hand_detected = torch.where(torch.tensor([d['hand'] is not None for d in data_100doh]))[0].long()
    hand_centers = torch.tensor(
        np.stack([0.5 * (data_100doh[i]['hand'][0, 0:2] + data_100doh[i]['hand'][0, 2:4]) for i in hand_detected])
    ).long()

    seg_hand_center = seg[hand_detected, hand_centers[:, 1], hand_centers[:, 0]].clone()
    # filter black
    seg_hand_center = seg_hand_center[
        torch.any(seg_hand_center != torch.tensor([0, 0, 0, 255], dtype=seg.dtype), dim=-1)]
    hand_color = torch.mode(seg_hand_center, dim=0).values

    obj_detected = torch.where(torch.tensor([d['obj'] is not None for d in data_100doh]))[0].long()
    obj_centers = torch.tensor(
        np.stack([0.5 * (data_100doh[i]['obj'][0, 0:2] + data_100doh[i]['obj'][0, 2:4]) for i in obj_detected])
    ).long()
    seg_obj_center = seg[obj_detected, obj_centers[:, 1], obj_centers[:, 0]].clone()
    # filter bg and hand color
    seg_obj_center = seg_obj_center[torch.any(seg_obj_center != bg_color, dim=-1)]
    seg_obj_center = seg_obj_center[torch.any(seg_obj_center != hand_color, dim=-1)]
    if seg_obj_center.shape[0] == 0:
        manipulated_obj_color = None
    else:
        manipulated_obj_color = torch.mode(seg_obj_center, dim=0).values

    # get unique colors and  filter bg, hand, and manipulated obj
    k = 8
    other_objects = torch.unique(seg[:, ::k, ::k].reshape(-1, 4), dim=0)
    other_objects = other_objects[torch.any(other_objects != bg_color, dim=-1)]
    other_objects = other_objects[torch.any(other_objects != hand_color, dim=-1)]
    if manipulated_obj_color is None:
        manipulated_obj_color = other_objects[0].clone()
    other_objects = other_objects[torch.any(other_objects != manipulated_obj_color, dim=-1)]

    return hand_color, manipulated_obj_color, other_objects


def estimate_is_object_initially_moving(data, n=5, movement_px=5):
    if data.in_contact[0]:
        return True
    obj_mask = data.manipulated_obj_masks()
    obj_areas = obj_mask.sum((-2, -1))
    obj_mask_detected = torch.nonzero(obj_areas, as_tuple=False)
    masks = obj_mask[obj_mask_detected[:n, 0]]
    cx, cy = mask_center(masks)
    cx0, cy0 = cx.mean(), cy.mean()
    return (cx - cx0).abs().mean() > movement_px or (cy - cy0).abs().mean() > movement_px


def estimate_is_object_moving_at_the_end(data, n=5, eps=0.1, movement_px=5):
    if data.in_contact[-1]:
        return True
    obj_mask = data.manipulated_obj_masks()
    obj_areas = obj_mask.sum((-2, -1))
    obj_mask_detected = torch.nonzero(obj_areas, as_tuple=False)
    masks = obj_mask[obj_mask_detected[-n:, 0]]
    cx, cy = mask_center(masks)
    cx0, cy0 = cx.mean(), cy.mean()
    return (cx - cx0).abs().mean() > movement_px or (cy - cy0).abs().mean() > movement_px
