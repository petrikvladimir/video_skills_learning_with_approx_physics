#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-20
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import argparse
import csv

import tqdm
from pytorch3d.transforms import so3_exp_map
from torch import tensor

from benchmark.utils import eval_vid, get_action_for_video
from utils.data_loader import DataLoaderWrapper, _default_cache_data
from collections import defaultdict

parser = argparse.ArgumentParser(description='Benchmark evaluation.')
parser.add_argument('-prefix', type=str, default='all')
options = parser.parse_args()

success_action_all = []
consistent_action_all = []
prefixes = [options.prefix] if options.prefix != 'all' else ['corlO', 'corlN',  # 'feb18', 'feb20', 'feb21', 'feb23',
                                                             'feb25', 'feb27', 'feb28_init']
for prefix in prefixes:
    success_action = defaultdict(lambda: 0)
    consistent_action = defaultdict(lambda: 0)
    success_vid = defaultdict(lambda: 0)
    for vid in tqdm.tqdm(DataLoaderWrapper.all_video_ids('all')):
        action = get_action_for_video(vid)
        data = DataLoaderWrapper(vid, physics_folder_prefix=prefix)
        if data.physics_state_computed():
            d = data.load_physics_state(to_torch=True)
            hrot = d['trajectory']['hand_rot']
            if prefix.startswith('corl'):
                hrot = so3_exp_map(tensor([[0., 0., 180.]]).to(hrot).deg2rad()).expand(hrot.shape[0], 3, 3).bmm(hrot)
            osize = d['obj_size'] if len(d['obj_size'].shape) == 1 else d['obj_size'].mean(dim=0)
            passed = eval_vid(
                vid=vid, opos=d['trajectory']['obj_pos'], orot=d['trajectory']['obj_rot'],
                hpos=d['trajectory']['hand_pos'],
                hrot=hrot, sopos=d['static_objs_pos0'], sorot=d['static_objs_rot0'],
                osize=osize, sosize=d['static_objs_size'],
            )
            # print(f'{vid}:{passed}')
            passed_consistent = True  # todo consider physical consistency evaluation
        else:
            print(f'not computed: {vid}')
            passed = False
            passed_consistent = False
        success_vid[action + ';' + vid] = 1 if passed else 0
        success_action[action] += 1 if passed else 0
        consistent_action[action] += 1 if passed_consistent else 0

    success_action['total'] = sum(success_action.values())
    success_action['perc'] = int(100 * success_action['total'] / 54)
    consistent_action['total'] = sum(consistent_action.values())
    success_action_all.append(success_action)
    consistent_action_all.append(consistent_action)

if len(prefixes) > 1:
    with open(_default_cache_data().joinpath(f'all.csv'), 'w') as f:
        w = csv.writer(f, delimiter="&")
        w.writerow(['exp_id'] + list(success_action_all[0].keys()))
        # corl_old = [5, 4, 5, 6, 2, 6, 4, 1, 0]
        # w.writerow(['corlP'] + corl_old + [sum(corl_old)])
        for success_action, prefix in zip(success_action_all, prefixes):
            w.writerow([prefix] + [' ' + str(v) + '/6 ' for v in list(success_action.values())[:-2]] + \
                       [' ' + str(v) + '/54 ' for v in list(success_action.values())[-2:-1]] + \
                       [' \SI{' + str(v) + '}{\percent} ' for v in list(success_action.values())[-1:]]
                       )

    with open(_default_cache_data().joinpath(f'physics_consistency.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['exp_id'] + list(consistent_action_all[0].keys()))
        for consistent_action, prefix in zip(consistent_action_all, prefixes):
            w.writerow([prefix] + list(consistent_action.values()))
else:
    for success_action, prefix in zip(success_action_all, prefixes):
        print(['exp_id'] + list(success_action_all[0].keys()))
        print(
            [prefix] + [' ' + str(v) + '/6 ' for v in list(success_action.values())[:-2]] + \
            [' ' + str(v) + '/54 ' for v in list(success_action.values())[-2:-1]] + \
            [' \SI{' + str(v) + '}{\percent} ' for v in list(success_action.values())[-1:]]
        )
