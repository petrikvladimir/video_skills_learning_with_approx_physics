#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import torch

from typing import Optional

from pytorch3d.transforms import rotation_6d_to_matrix
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs as nccoeffs
from torch.nn import Parameter as P

from diffeq.utils import init_rot_param, init_pos_param, velocities_from_positions, velocities_from_rotations, \
    init_vec_param


class HandSpline(torch.nn.Module):

    def __init__(self, spline_complexity=5, max_t=1., init_vel_grasp=None, init_vel_lin=None, init_vel_ang=None,
                 init_pos0=None, init_rot0=None, init_grasp0=None, device='cpu', dtype=torch.float64):
        """
        Continuous hand motion representation by velocities.
        :param spline_complexity: number of kepoints [N]
        :param max_t: maximum time
        :param init_vel_grasp: [N x 1]
        :param init_vel_lin:  [N x 3]
        :param init_vel_ang:  [N x 3]
        :param device:
        :param dtype:
        """
        super().__init__()
        n = spline_complexity
        ttype = dict(device=device, dtype=dtype)
        self.spline_t = torch.linspace(0, max_t, n, **ttype)
        init_spline = torch.cat(
            [
                init_vel_lin if init_vel_lin is not None else 0.01 * torch.randn(n, 3, **ttype),
                init_vel_ang if init_vel_ang is not None else 0.01 * torch.randn(n, 3, **ttype),
                init_vel_grasp if init_vel_grasp is not None else 0.01 * torch.randn(n, 1, **ttype),
            ], dim=-1
        )
        self.spline_y = P(init_spline, requires_grad=True)
        self.grasp0 = init_vec_param(init_grasp0.view(1) if init_grasp0 is not None else None, **ttype,
                                     init_value=torch.zeros(1, **ttype))
        self.pos0 = init_pos_param(init_pos0, **ttype)
        self.rot6d0 = init_rot_param(init_rot0, **ttype)

        # self.spline: Optional[NaturalCubicSpline] = None
        # self.reset()

    @property
    def rot0(self):
        """ Return 3x3 matrix representing the rotation. """
        return rotation_6d_to_matrix(self.rot6d0.unsqueeze(0)).squeeze(0)

    def velocities(self, t):
        """ Return linear and angular velocity and the grasp signal for given time(s) t"""
        spline = NaturalCubicSpline(nccoeffs(self.spline_t, self.spline_y))
        out = spline.evaluate(torch.clamp(t, min=self.spline_t[0], max=self.spline_t[-1]))
        ind_bellow = t < self.spline_t[0]
        ind_above = t > self.spline_t[-1]
        out[ind_bellow, :] = 0
        out[ind_above, :] = 0
        return out[..., :3], out[..., 3:6], out[..., 6:]

    def forward(self, t):
        assert NotImplemented()

    @staticmethod
    def in_contact_signal_from_contacts(in_contact):
        return 0.1 * 2 * (in_contact.float() - 0.5)

    @staticmethod
    def get_spline_initialization_from_motion(ref_hand_pos, ref_hand_rot, in_contact, spline_complexity=5, fps=12):
        in_contact_signal = HandSpline.in_contact_signal_from_contacts(in_contact).to(ref_hand_pos)
        n = ref_hand_pos.shape[0]
        inds = torch.cat(
            [torch.zeros(1), torch.linspace(0, n, spline_complexity * 2 - 1)[1::2], torch.tensor([n - 1])]
        ).long().to(ref_hand_pos.device)
        timestamps = inds.float() / fps
        lin_vel = velocities_from_positions(ref_hand_pos[inds], timestamps)
        ang_vel = velocities_from_rotations(ref_hand_rot[inds], timestamps)
        init_vel_grasp = velocities_from_positions(in_contact_signal[inds].unsqueeze(-1), timestamps)
        return dict(
            max_t=n / fps,
            spline_complexity=spline_complexity,
            init_vel_lin=lin_vel, init_vel_ang=ang_vel,
            init_pos0=ref_hand_pos[0], init_rot0=ref_hand_rot[0], init_grasp0=in_contact_signal[0],
            init_vel_grasp=init_vel_grasp,
            device=ref_hand_pos.device, dtype=ref_hand_pos.dtype,
        )
