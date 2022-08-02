#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-16
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from torch import Tensor

from diffeq.box import Box
from diffeq.diff_renderer import MeshesRenderer, CameraModelTzAndElevation
from diffeq.simulation import SimulationCuboidsOnPlaneWithHand
from utils.data_loader import DataLoaderWrapper
from utils.params import get_cylinder_hand_mesh


class SimulationRendererCuboidsOnPlaneWithHand(MeshesRenderer):
    def __init__(self, data: DataLoaderWrapper, cameras, image_size, device, dtype, render_static_object=None):
        self.tparam = dict(device=device, dtype=dtype)
        self.has_static = data.has_other_object() if render_static_object is None else render_static_object
        self.hand_color = data.hand_color.to(device=device, dtype=dtype) / 255
        self.manipulated_obj_color = data.manipulated_obj_color.to(device=device, dtype=dtype) / 255
        self.static_obj_color = None

        meshes = [Box.unit_trimesh_box(), get_cylinder_hand_mesh()]
        meshes_colors = [self.manipulated_obj_color.cpu().numpy()[:3], self.hand_color.cpu().numpy()[:3]]

        if self.has_static:
            self.static_obj_color = data.other_objs_color[0].to(device=device, dtype=dtype) / 255
            meshes += [Box.unit_trimesh_box()]
            meshes_colors += [self.static_obj_color.cpu().numpy()[:3]]

        super().__init__(meshes, meshes_colors, cameras, device, image_size, dtype, False)

    @staticmethod
    def fix_pos_rot_shape(pos, rot):
        return pos.unsqueeze(1) if len(pos.shape) == 2 else pos, rot.unsqueeze(1) if len(rot.shape) == 3 else rot

    def render(self, opos, orot, osize, hpos, hrot, cam_pos, cam_rot, sp=None, sr=None, sz=None):
        opos, orot = self.fix_pos_rot_shape(opos, orot)
        hpos, hrot = self.fix_pos_rot_shape(hpos, hrot)
        n = opos.shape[0]
        osize = osize.view(1, 1, 3).expand(n, 1, 3)
        hsize = torch.ones(n, 1, 3).to(hpos)
        if not self.has_static:
            return super().forward(
                torch.cat([opos, hpos], dim=1).to(**self.tparam),
                torch.cat([orot, hrot], dim=1).to(**self.tparam),
                torch.cat([osize, hsize], dim=1).to(**self.tparam),
                cam_pos.to(**self.tparam), cam_rot.to(**self.tparam),
            )
        return super().forward(
            torch.cat([opos, hpos, sp], dim=1).to(**self.tparam),
            torch.cat([orot, hrot, sr], dim=1).to(**self.tparam),
            torch.cat([osize, hsize, sz], dim=1).to(**self.tparam),
            cam_pos.to(**self.tparam), cam_rot.to(**self.tparam),
        )

    def render_sim(self, simulation: SimulationCuboidsOnPlaneWithHand, opos, orot, hpos, hrot, cam_pos, cam_rot):
        n = opos.shape[0]
        return self.render(
            opos=opos, orot=orot, osize=simulation.obj_size,
            hpos=hpos, hrot=hrot, cam_pos=cam_pos, cam_rot=cam_rot,
            sp=simulation.static_objs_pos0[:1].unsqueeze(1).expand(n, 1, 3) if self.has_static else None,
            sr=simulation.static_objs_rot0[:1].unsqueeze(1).expand(n, 1, 3, 3) if self.has_static else None,
            sz=simulation.static_objs_size[:1].unsqueeze(1).expand(n, 1, 3) if self.has_static else None,
        )

    def top_cam_pos_rot(self, opos, in_camera=False):
        if in_camera:
            cam2world_t = opos.view(-1, 3).mean(dim=0)
            cam2world_t[1] = 1.
            cam2world_rot = axis_angle_to_matrix(torch.tensor([np.pi / 2, 0., 0.]).to(opos))
        else:
            cam2world_t = opos.view(-1, 3).mean(dim=0)
            cam2world_t[2] = 1.
            cam2world_rot = axis_angle_to_matrix(torch.tensor([0., 0., np.pi]).to(opos)) @ \
                            axis_angle_to_matrix(torch.tensor([np.pi, 0., 0.]).to(opos))
        world2cam_rot = cam2world_rot.T
        world2cam_t = -(cam2world_rot.T @ cam2world_t.unsqueeze(-1)).squeeze(-1)  # -R.T @ t
        n = opos.shape[0]
        cam_rot = world2cam_rot.T.unsqueeze(0).expand(n, 3, 3)
        cam_pos = world2cam_t.unsqueeze(0).expand(n, 3)
        return cam_pos, cam_rot

    def render_sim_top(self, simulation: SimulationCuboidsOnPlaneWithHand, opos, orot, hpos, hrot):
        cam_pos, cam_rot = self.top_cam_pos_rot(opos, in_camera=False)
        return self.render_sim(simulation=simulation, opos=opos, orot=orot, hpos=hpos, hrot=hrot, cam_pos=cam_pos,
                               cam_rot=cam_rot)

    def render_top(self, opos, orot, osize, hpos, hrot, sp=None, sr=None, sz=None, in_camera=False):
        cam_pos, cam_rot = self.top_cam_pos_rot(opos, in_camera)
        return self.render(opos=opos, orot=orot, osize=osize, hpos=hpos, hrot=hrot, cam_pos=cam_pos, cam_rot=cam_rot,
                           sp=sp, sr=sr, sz=sz)

    def static_obj_prob(self, imgs):
        return (-10. * (imgs - self.static_obj_color).norm(dim=-1)).exp()

    def manipulated_obj_prob(self, imgs):
        return (-10. * (imgs - self.manipulated_obj_color).norm(dim=-1)).exp()

    def hand_prob(self, imgs):
        return (-10. * (imgs - self.hand_color).norm(dim=-1)).exp()
