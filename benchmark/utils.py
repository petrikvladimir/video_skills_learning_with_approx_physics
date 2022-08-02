#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
import torch
from diffeq.box import Box
from diffeq.distances import sdf_to_box_batch


def get_action_for_video(vid):
    if vid in ['8675', '13956', '6848', '20193', '10960', '15606']:
        return 'pull_left_to_right'
    if vid in ['13309', '3107', '11240', '2458', '10116', '11732']:
        return 'pull_right_to_left'
    if vid in ['3694', '601', '9889', '4018', '6955', '6132']:
        return 'push_left_to_right'
    if vid in ['3949', '1987', '4218', '4378', '1040', '8844']:
        return 'push_right_to_left'
    if vid in ['7194', '38559', '12359', '1838', '2875', '24925']:
        return 'pick_up'
    if vid in ['4053', '19', '874', '3340', '1890', '1642']:
        return 'put_next_to'
    if vid in ['1663', '41177', '28603', '5967', '14841', '2114']:
        return 'put_in_front_of'
    if vid in ['1044', '6538', '3074', '12986', '779', '4388']:
        return 'put_behind'
    if vid in ['8801', '7655', '13390', '757', '7504', '16310']:
        return 'put_onto'


def hand_azimuth_deg_from_hrot(hrot):
    """ Return azimuth in degrees from given rotation matrices """
    v = hrot.bmm(torch.tensor([0, 0, 1.]).view(1, 3, 1).expand(hrot.shape[0], 3, 1)).squeeze(-1)
    hand_azimuth = torch.atan2(v[:, 1], v[:, 0]).rad2deg()
    return hand_azimuth


def check_obj_above_ground(opos, orot, osize, eps=0.02, p=0.25):
    """ Check that object is above ground p% of the time """
    n = opos.shape[0]
    b = Box(dtype=opos.dtype)
    min_z = []
    for pos, rot in zip(opos, orot):
        pp, _ = b.points_positions_and_velocities(b.q_s(rot, osize), pos, torch.zeros(3), torch.zeros(3))
        min_z.append(torch.min(pp[:, 2]))
    perc_below_ground = (torch.tensor(min_z) < -eps).sum().float() / n
    if perc_below_ground > p:
        return False
    return True


def eval_vid(vid, opos, orot, osize, hpos, hrot, sopos=None, sorot=None, sosize=None):
    """
    Return true/false if the given trajectories passed the benchmark defined in CoRL paper.
    :param vid_id:
    :param opos: [nx3] manipulated object position
    :param orot: [nx3x3] manipulated object rotation
    :param osize: [3] manipulated obj. size
    :param hpos: [nx3] hand position
    :param hrot: [nx3x3] hand rotation
    :param sopos: [1x3] static object position or None
    :param sorot: [1x3x3] static object rotation or None
    :param sosize: [1x3]  static obj. size or None
    """
    action = get_action_for_video(vid)

    """ Hand azimuth must be correct at least 75% of the time """
    if action == 'pull_left_to_right' or action == 'push_right_to_left':  # hand azimuth in +-90deg
        hand_azimuth = hand_azimuth_deg_from_hrot(hrot)
        azimuth_ok = hand_azimuth.abs() < 90
        hand_ok = azimuth_ok.sum() > 0.75 * azimuth_ok.shape[0]
    elif action == 'pull_left_to_right' or action == 'push_right_to_left':  # azim. > 90 or < 90, assuming normalized
        hand_azimuth = hand_azimuth_deg_from_hrot(hrot)
        azimuth_ok = torch.bitwise_or(hand_azimuth < 90, hand_azimuth > 90)
        hand_ok = azimuth_ok.sum() > 0.75 * azimuth_ok.shape[0]
    else:
        hand_ok = True

    dang = 60

    b = Box(dtype=opos.dtype)
    if sosize is not None:
        pp, _ = b.points_positions_and_velocities(b.q_s(sorot[0], sosize[0]), sopos[0], torch.zeros(3), torch.zeros(3))
        so_min_y, so_max_y = torch.min(pp[:, 1]), torch.max(pp[:, 1])
        so_min_x, so_max_x = torch.min(pp[:, 0]), torch.max(pp[:, 0])
        ang_from_bl = torch.atan2(opos[-1, 1] - so_min_y, opos[-1, 0] - so_min_x).rad2deg()
        ang_from_br = torch.atan2(opos[-1, 1] - so_min_y, opos[-1, 0] - so_max_x).rad2deg()
        ang_from_tl = torch.atan2(opos[-1, 1] - so_max_y, opos[-1, 0] - so_min_x).rad2deg()
        ang_from_tr = torch.atan2(opos[-1, 1] - so_max_y, opos[-1, 0] - so_max_x).rad2deg()

    if action == 'pull_left_to_right' or action == 'push_left_to_right':
        obj_ok = opos[0, 0] < opos[-1, 0] - 0.05  # at least 5cm righ
    elif action == 'pull_right_to_left' or action == 'push_right_to_left':
        obj_ok = opos[0, 0] > opos[-1, 0] + 0.05  # at least 5cm left
    elif action == 'pick_up':
        obj_ok = opos[0, 2] < opos[-1, 2] - 0.01  # at least 1cm up
        # print(f'{vid}: {opos[-1, 2] - opos[0, 2]}')
    elif action == 'put_next_to':
        on_the_left = (ang_from_bl < -180 + dang or ang_from_bl > 90) and (
                ang_from_tl > 180 - dang or ang_from_tl < -90)
        on_the_right = -dang < ang_from_br < 90 and -90 < ang_from_tr < dang
        obj_ok = on_the_left or on_the_right
    elif action == 'put_in_front_of':
        obj_ok = -90 - dang < ang_from_bl < 0 and -180 < ang_from_br < -90 + dang
    elif action == 'put_behind':
        obj_ok = 0 < ang_from_tl < 90 + dang and 90 - dang < ang_from_br < 180
    elif action == 'put_onto':
        R = sorot[0].T
        t = -R @ sopos[0]
        o_in_s = R @ opos[-1] + t
        b = Box(dtype=opos.dtype)
        pp, _ = b.points_positions_and_velocities(b.q_s(orot[-1], osize), opos[-1], torch.zeros(3), torch.zeros(3))
        min_obj_height = torch.min(pp[:, 2])
        min_height = sopos[0, 2] + sosize[0, 2] / 2 - 0.03  # eps for contact resolution
        max_height = sopos[0, 2] + sosize[0, 2] / 2 + 0.03
        obj_ok = torch.all(o_in_s[:2].abs() < sosize[0][:2]) and min_height < min_obj_height < max_height
    else:
        raise NotImplementedError()

    def _check_box_close_to_ground(pos, rot, eps=0.01):
        b = Box(dtype=opos.dtype)
        pp, _ = b.points_positions_and_velocities(b.q_s(rot, osize), pos, torch.zeros(3), torch.zeros(3))
        return torch.min(pp[:, 2]) < eps

    if action in ['pull_left_to_right', 'push_left_to_right', 'pull_right_to_left', 'push_right_to_left']:
        b = Box(dtype=opos.dtype)
        osize_z = torch.zeros(1)
        for i in range(opos.shape[0]):
            pp, _ = b.points_positions_and_velocities(b.q_s(orot[i], osize), opos[i], torch.zeros(3), torch.zeros(3))
            osize_z = torch.max(osize_z, torch.max(pp[:, 2]) - torch.min(pp[:, 2]))
        obj_z_motion = (torch.max(opos[:, 2]) - torch.min(opos[:, 2]))
        obj_on_ground = obj_z_motion < osize_z
    elif action in ['put_next_to', 'put_in_front_of', 'put_behind']:
        # min object position is below max static obj position
        ps, _ = b.points_positions_and_velocities(b.q_s(sorot[0], sosize[0]), sopos[0], torch.zeros(3), torch.zeros(3))
        po, _ = b.points_positions_and_velocities(b.q_s(orot[-1], osize), opos[-1], torch.zeros(3), torch.zeros(3))
        obj_on_ground = torch.min(po[:, 2]) < torch.max(ps[:, 2])
    elif action == 'pick_up':
        obj_on_ground = _check_box_close_to_ground(opos[0], orot[0])
    else:
        obj_on_ground = True  # not valid for this action

    return hand_ok and obj_ok and obj_on_ground and check_obj_above_ground(opos, orot, osize)
