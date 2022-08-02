#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2021-02-2
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

from abc import abstractmethod
from typing import List, Tuple
import torch
from torch import Tensor
from torchdiffeq import odeint, odeint_event


class BaseSystem(torch.nn.Module):

    def __init__(self, max_t=1., device='cpu', dtype=torch.float64):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.max_t = max_t

        self.ode_options = {'atol': 1e-9, 'rtol': 1e-7}
        self.odeint_interface = odeint

    @abstractmethod
    def reset(self, t0):
        """
            Reset is called at the beginning of the integration (only for t0, not for all events).
            It returns the start state (tuple of tensors).
        """
        pass

    @abstractmethod
    def event_fn(self, t, state):
        return torch.zeros(0).to(t)

    def event_state_update(self, t, state):
        """ Update state on event that was detected at time t. """
        return state

    def _event_fn(self, t, state):
        """ Check for event_fn and for maximum time. """
        return torch.cat([self.event_fn(t.to(self.dtype), state), t.view(1) - self.max_t])

    def get_all_events(self, t0=None, max_num_of_events=100, verbose=False) -> List[Tuple[Tensor, Tensor]]:
        """ Detect all events (i.e. intersections of zero) given by self.event_fn(t, s). Return list of tuples t+s.  """
        t0 = torch.zeros(1, device=self.device, dtype=self.dtype) if t0 is None else t0
        events = [(t0, self.reset(t0))]
        while events[-1][0] < self.max_t - 1e-3 and len(events) < max_num_of_events:
            t, s = events[-1]
            time_event, sol = odeint_event(
                self, s, t, odeint_interface=self.odeint_interface, event_fn=self._event_fn, **self.ode_options
            )
            events.append((time_event, self.event_state_update(time_event, tuple(s[-1] for s in sol))))
            if verbose:
                print(f'Detected event at time: {time_event}')
        return events

    def integrate_between_events(self, events, timesteps=None, detach=False):
        """
        :param timesteps:
        :param events: list of events as returned from get all events
        :return: list of tensors
        """
        assert len(events) > 0
        self.reset(events[0][0])
        if timesteps is None:
            timesteps = torch.linspace(0., self.max_t, 100, device=self.device, dtype=self.dtype)
        out = [torch.empty(timesteps.shape[0], *y.shape, device=self.device, dtype=self.dtype) for y in events[0][1]]
        for s, o in zip(events[0][1], out):
            o[0] = s
        for i in range(len(events) - 1):  # todo can be batched or at least parallelized
            t0, s0 = events[i]
            t1, s1 = events[i + 1]
            # we assume last event is given by stop time
            mask = timesteps.gt(t0) if i == len(events) - 2 else torch.bitwise_and(timesteps.gt(t0), timesteps.le(t1))
            sol = self.odeint_interface(self, s0, torch.cat([t0.reshape(1), timesteps[mask]]), **self.ode_options)
            for s, o in zip(sol, out):
                o[mask] = s[1:]
        if detach:
            return [o.detach() for o in out]
        return out
