#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-8
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import torch
import tqdm
from pytorch3d.transforms import so3_relative_angle
from diffeq.base_module import BaseSystem
from diffeq.hand import HandSpline
from diffeq.utils import vhat


class HandEquationOfMotion(BaseSystem):

    def __init__(self, hand: HandSpline):
        super().__init__(hand.spline_t[-1], hand.spline_t.device, hand.spline_t.dtype)
        self.hand = hand
        self.ode_options = {
            'atol': 1e-5,
            'rtol': 1e-5,
            'method': 'dopri5',
            'options': dict(step_t=torch.linspace(0, self.max_t, 20).to(hand.spline_t)),
        }

    def start_state(self):
        return self.hand.pos0, self.hand.rot0, self.hand.grasp0

    def reset(self, t0):
        # self.hand.reset()
        return self.start_state()

    def forward(self, t, state):
        _, hrot, _ = state
        hv, hw, hgv = self.hand.velocities(t)
        return hv, vhat(hw) @ hrot, hgv

    def fit(self, ref_hand_pos, ref_hand_rot, in_contact, iterations=100):
        in_contact_signal = HandSpline.in_contact_signal_from_contacts(in_contact).to(ref_hand_pos)
        timesteps = torch.linspace(0, self.max_t, ref_hand_pos.shape[0]).to(ref_hand_pos)
        optimizer = torch.optim.Adam(self.hand.parameters(), lr=1e-2, amsgrad=True)

        tr = tqdm.trange(iterations)
        for _ in tr:
            optimizer.zero_grad()
            hpos, hrot, hgrasp = self.integrate_between_events(self.get_all_events(), timesteps=timesteps)

            loss_pos = (hpos - ref_hand_pos).square().mean()
            loss_rot = (1 - so3_relative_angle(hrot, ref_hand_rot, cos_angle=True)).square().mean()
            loss_grasp = (hgrasp.squeeze() - in_contact_signal).square().mean()

            loss = loss_pos + loss_rot + loss_grasp
            tr.set_description(f'Hand spline fitting loss: {loss.item()}')
            loss.backward()
            optimizer.step()
