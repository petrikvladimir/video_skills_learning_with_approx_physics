#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-03-3
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

from pyphysx import *
from pyphysx_render.meshcat_render import MeshcatViewer
import quaternion as npq
from pyphysx_render.utils import gl_color_from_matplotlib

from utils.params import get_cylinder_hand_mesh


class CuboidsWithHandViewer:
    def __init__(self, obj_size, static_objs_size=None, static_objs_pos0=None, static_objs_rot0=None, fps=12) -> None:
        self.scene = Scene()

        "Create box grasped"
        self.box_grasped = RigidDynamic()
        box_shape_grasped = Shape.create_box(obj_size.detach(), Material())
        grasped_color = gl_color_from_matplotlib('tab:green', alpha=0.75, return_rgba=True) / 255.
        box_shape_grasped.set_user_data({'color': grasped_color})
        self.box_grasped.attach_shape(box_shape_grasped)
        self.scene.add_actor(self.box_grasped)

        "Create box released"
        self.box_released = RigidDynamic()
        box_shape_released = Shape.create_box(obj_size.detach(), Material())
        released_color = gl_color_from_matplotlib('tab:red', alpha=0.75, return_rgba=True) / 255.
        box_shape_released.set_user_data({'color': released_color})
        self.box_released.attach_shape(box_shape_released)
        self.scene.add_actor(self.box_released)

        "Create hand"
        self.hand = RigidDynamic()
        hand_mesh = get_cylinder_hand_mesh()
        hand_shape: Shape = Shape.create_convex_mesh_from_points(hand_mesh.vertices, Material())
        hand_shape.set_user_data({'color': 'tab:orange', 'visual_mesh': hand_mesh})
        self.hand.attach_shape(hand_shape)
        self.scene.add_actor(self.hand)

        num_static_objects = static_objs_size.shape[0] if static_objs_size is not None else 0
        assert static_objs_pos0 is None or static_objs_pos0.shape[0] == num_static_objects
        assert static_objs_rot0 is None or static_objs_rot0.shape[0] == num_static_objects
        for i in range(num_static_objects):
            static_box = RigidDynamic()
            box_shape = Shape.create_box(static_objs_size[i].detach(), Material())
            static_color = gl_color_from_matplotlib('tab:blue', alpha=0.75, return_rgba=True) / 255.
            box_shape.set_user_data({'color': static_color})
            static_box.attach_shape(box_shape)
            static_box.set_global_pose((static_objs_pos0[i], npq.from_rotation_matrix(static_objs_rot0[i])))
            self.scene.add_actor(static_box)

        "Create viewer"
        self.viewer = MeshcatViewer(open_meshcat=True, wait_for_open=True, render_to_animation=True, animation_fps=fps)
        self.viewer.vis["/Background"].set_property("visible", False)
        self.viewer.add_physx_scene(scene=self.scene)

    def animate(self, opos, orot, ograsped, hpos, hrot):
        for p, r, hp, hr, g in zip(opos, orot, hpos, hrot, ograsped):
            self.box_released.set_global_pose((p, npq.from_rotation_matrix(r)))
            self.box_grasped.set_global_pose((p, npq.from_rotation_matrix(r)))
            if g > 0:
                self.box_released.set_global_pose([-100.] * 3)
            else:
                self.box_grasped.set_global_pose([-100.] * 3)
            self.hand.set_global_pose((hp, npq.from_rotation_matrix(hr)))
            self.viewer.update()
        self.viewer.publish_animation()
