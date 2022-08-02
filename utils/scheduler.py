#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from copy import deepcopy

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ReduceLROnPlateauResetToBest(ReduceLROnPlateau):

    def __init__(self, optimizer: Optimizer, factor=0.5, patience=5,
                 best_model_recovery_fn=None, models=None, recover_nth_best=0) -> None:
        super().__init__(optimizer, factor=factor, patience=patience)
        assert recover_nth_best < 2
        self.recover_nth_best = recover_nth_best
        self.best_model_recovery_fn = best_model_recovery_fn
        self.models = [] if models is None else models
        self._best_params = [deepcopy(m.state_dict()) for m in self.models]
        self._best_params_t_minus_1 = deepcopy(self._best_params)  # i.e. parameters one iteration before the best
        self._last_params = deepcopy(self._best_params)
        self._best_loss = None

    def update_best_if_lower(self, loss):
        updated = False
        if self._best_loss is None or loss < self._best_loss:
            self._best_params_t_minus_1 = deepcopy(self._last_params)
            self._best_params = [deepcopy(m.state_dict()) for m in self.models]
            self._best_loss = loss.detach()
            updated = True
        self._last_params = [deepcopy(m.state_dict()) for m in self.models]
        return updated

    def recover_best(self, recover_nth_best):
        for m, s in zip(self.models, self._best_params if recover_nth_best == 0 else self._best_params_t_minus_1):
            m.load_state_dict(deepcopy(s))
        state = deepcopy(self.optimizer.state_dict())
        state['state'] = {}  # let's clear momentum
        self.optimizer.load_state_dict(state)
        if self.best_model_recovery_fn is not None:
            self.best_model_recovery_fn()

    def _reduce_lr(self, epoch):
        super(ReduceLROnPlateauResetToBest, self)._reduce_lr(epoch)
        self.recover_best(self.recover_nth_best)
