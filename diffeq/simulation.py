#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-03-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import torch
from torch import Tensor
from typing import Tuple
from diffeq.base_module import BaseSystem
from diffeq.distances import sdf_to_box, sdf_to_box_with_normals_without_bottom
from diffeq.hand import HandSpline
from diffeq.box import Box
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, axis_angle_to_matrix
from diffeq.utils import init_pos_param, init_rot_param, init_vec_param, vhat, fix_rot
from torch.nn import Parameter as Par


class SimulationCuboidsOnPlaneWithHand(BaseSystem):

    def __init__(self, hand,
                 obj_pos0=None, obj_rot0=None, obj_size=None, obj_vel0=None, obj_omega0=None,
                 number_of_static_objects=0, static_objs_pos0=None, static_objs_rot0z=None, static_objs_size=None,
                 min_static_objs_size=0.02, **kwargs):
        """
        This module simulates hand and one manipulated objects. Other static objects might be added into the scene.
        All input parameters are used for initialization of the nn parameters.
        """
        super().__init__(max_t=hand.spline_t[-1], device=hand.spline_t.device, dtype=hand.spline_t.dtype)
        self.ode_options = {
            'atol': 1e-3,
            'rtol': 1e-3,
            'method': 'dopri5',
        }
        self.grasped_eps = 1e-3
        self.tangential_velocity_reduction = 10.
        self.angular_damping = 0.5
        self.linear_damping = 0.5
        ttype = dict(device=self.device, dtype=self.dtype)
        self.dog_ = torch.zeros(1, **ttype)

        self.hand: HandSpline = hand
        self.box = Box(num_points_edge=5, **ttype)

        self.obj_pos0 = init_pos_param(obj_pos0, **ttype)
        self.obj_rot6d0 = init_rot_param(obj_rot0, **ttype)
        osize_sqrt = obj_size.sqrt() if obj_size is not None else (0.1 * torch.ones(3, **ttype)).sqrt()
        self.obj_size_sqrt = init_vec_param(osize_sqrt, **ttype)
        self.obj_vel0 = init_vec_param(obj_vel0, **ttype)
        self.obj_omega0 = init_vec_param(obj_omega0, **ttype)

        self.number_of_static_objects = number_of_static_objects
        if number_of_static_objects > 0:
            n = self.number_of_static_objects
            op = (static_objs_pos0[:, :2] if static_objs_pos0 is not None else torch.zeros(n, 2)).to(**ttype)
            self.static_objs_pos0_xy = Par(op, True)
            rz = (static_objs_rot0z if static_objs_rot0z is not None else torch.zeros(n, 1)).to(**ttype)
            self.static_objs_rot0z = Par(rz, True)
            init_size = ((0.1 - min_static_objs_size) * torch.ones(n, 3, **ttype)).sqrt()
            sz = (static_objs_size - min_static_objs_size).sqrt() if static_objs_size is not None else init_size
            self.static_objs_size_sqrt = Par(sz, True)
            self.min_static_objs_size = min_static_objs_size

    @property
    def obj_rot0(self):
        return rotation_6d_to_matrix(self.obj_rot6d0.unsqueeze(0)).squeeze(0)

    @property
    def obj_size(self):
        return self.obj_size_sqrt.square()

    @property
    def static_objs_size(self):
        return self.static_objs_size_sqrt.square() + self.min_static_objs_size

    @property
    def static_objs_pos0(self):
        """ Return NumStaticObjs x 3 tensor of positions. Assuming object lies on the ground. """
        return torch.cat([self.static_objs_pos0_xy, 0.5 * self.static_objs_size[:, 2:]], dim=-1)

    @property
    def static_objs_rot0(self):
        """ Return NumStaticObjs x 3 tensor of positions. Assuming object lies on the ground. """
        rz = self.static_objs_rot0z
        return axis_angle_to_matrix(torch.cat([torch.zeros(self.number_of_static_objects, 2).to(rz), rz], dim=-1))

    def get_box_state(self, t, state):
        """ Get state of the box, i.e. it's position, velocity, rotation, omega, is_grasped. """
        return state[:5]

    def get_hand_state(self, t, state):
        """ Get state of the hand, i.e. it's position, velocity, rotation, omega, can_grasp_signal. """
        hv, hw, _ = self.hand.velocities(t)
        return state[5], hv, state[6], hw, state[7]

    @staticmethod
    def grasped_obj_velocity(opos: Tensor, hpos: Tensor, hvel: Tensor, homega: Tensor) -> Tuple[Tensor, Tensor]:
        """ Compute object linear and angular velocity that is grasped by the hand. """
        return hvel + homega.cross(opos - hpos), homega

    def reset(self, t0):
        """
        Compute start state based on fixed initial pose and velocities. Computes missing obj_grasped signal.
        Initial object position is translated in z-axis such that all object's points are above ground.
        """
        super().reset(t0)
        # self.hand.reset()
        # hv0, _, hg0 = self.hand.velocity_and_grasp(t0)
        q_s = self.box.q_s(self.obj_rot0, self.obj_size)
        ppos, _ = self.box.points_positions_and_velocities(q_s, self.obj_pos0, self.obj_vel0, self.obj_omega0)
        obj_pos0 = self.obj_pos0 - torch.tensor([0., 0., torch.min(ppos[:, 2]).clamp_max(0.)]).to(self.obj_pos0)

        # is above the ground, let's project outside the box
        ppos, _ = self.box.points_positions_and_velocities(q_s, obj_pos0, self.obj_vel0, self.obj_omega0)
        for i in range(self.number_of_static_objects):
            sdf, n = sdf_to_box_with_normals_without_bottom(ppos, self.static_objs_pos0[i], self.static_objs_rot0[i],
                                                            self.static_objs_size[i])
            min_sdf, ind = torch.min(sdf, dim=-1)
            if min_sdf < 0.:
                obj_pos0 = obj_pos0 - min_sdf * n[ind]

        hg0 = self.hand.grasp0
        g = self.box.is_grasped(pos=obj_pos0, rot=self.obj_rot0, size=self.obj_size, hpos=self.hand.pos0, hg=hg0)
        return obj_pos0, self.obj_vel0, self.obj_rot0, self.obj_omega0, g, self.hand.pos0, self.hand.rot0, hg0

    def event_fn(self, t, state):
        """ Detect events based on grasp/release of the box. """
        op, ov, orot, ow, og = self.get_box_state(t, state)
        hp, hv, hrot, hw, hg = self.get_hand_state(t, state)
        eps = self.grasped_eps * og
        return self.box.obj_not_grasped_signal(pos=op, rot=orot, size=self.obj_size, hpos=hp, hg=hg) - eps

    def event_state_update(self, t, state):
        """
        Called if state of grasped/released changed. Update state accordingly, i.e. release with hand velocity if
        grasped before, or ze velocity to zero if released before.
        """
        super().event_state_update(t, state)
        op, ov, orot, ow, og = self.get_box_state(t, state)
        hp, hv, hrot, hw, hg = self.get_hand_state(t, state)
        nov = ov
        now = ow
        nog = self.box.is_grasped(pos=op, rot=orot, size=self.obj_size, hpos=hp, hg=hg)
        if og > 0. > nog:  # object was grasped and is being released now. use hand velocity for setting current obj. velocity
            nov, now = self.grasped_obj_velocity(opos=op, hpos=hp, hvel=hv, homega=hw)
        elif og < 0. < nog:
            nov = torch.zeros_like(ov)
            now = torch.zeros_like(ow)
        return op, nov, orot, now, nog, hp, hrot, hg

    def forward(self, t, state):
        """
        Compute derivative of the system state by taking into account plane collision and grasped/released signal.
        """
        op, ov, orot, ow, og = self.get_box_state(t, state)
        hp, hv, hrot, hw, hg = self.get_hand_state(t, state)
        orot, hrot = fix_rot(orot), fix_rot(hrot)
        if og < 0.:  # obj is not grasped, use gravity dynamics
            dop = ov
            dov = torch.tensor([0., 0., -9.81]).to(op) - self.linear_damping * ov
            dorot = vhat(ow) @ orot
            dow = -self.angular_damping * ow  # torch.zeros(3).to(op)
        else:  # obj is grasped, let's use hand motion
            nov, now = self.grasped_obj_velocity(opos=op, hpos=hp, hvel=hv, homega=hw)
            dop = nov + ov  # note, ov is used here to react to impulses caused by collisions
            dov = -100. * ov  # collision impulse quick decay
            dorot = vhat(now + ow) @ orot
            dow = -100. * ow

        imp_dv, imp_dw = self.box.impulse_by_plane_collision(
            pos=op, vel=ov, size=self.obj_size, rot=orot, omega=ow,
            tangential_velocity_reduction=self.tangential_velocity_reduction,
            kp=500, kv=5,
        )
        """ Static objects collisions """
        imp_sdv = torch.zeros_like(imp_dv)
        imp_sdw = torch.zeros_like(imp_dw)
        for i in range(self.number_of_static_objects):
            imp_dv2, imp_dw2 = self.box.impulse_by_static_box_collision(
                pos=op, vel=ov, rot=orot, omega=ow, size=self.obj_size,
                other_box_pos=self.static_objs_pos0[i], other_box_rot=self.static_objs_rot0[i],
                other_box_size=self.static_objs_size[i],
                tangential_velocity_reduction=self.tangential_velocity_reduction,
                kp=500, kv=5,
            )
            imp_sdv = imp_sdv + imp_dv2
            imp_sdw = imp_sdw + imp_dw2
        _, _, hgv = self.hand.velocities(t)
        return dop, dov + imp_dv + imp_sdv, dorot, dow + imp_dw + imp_sdw, self.dog_, hv, vhat(hw) @ hrot, hgv
