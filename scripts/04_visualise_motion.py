#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-02-18
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import torch

from diffeq.box import Box
from utils.data_loader import DataLoaderWrapper
from utils.params import get_default_arg_parser
from viewer.viewer_utils import CuboidsWithHandViewer
from pytorch3d.transforms import so3_exp_map

parser = get_default_arg_parser()
parser.add_argument('-prefix', type=str, default='friday')
options = parser.parse_args()

vid = options.vid
print(f'Processing: {vid}')

data = DataLoaderWrapper(vid, img_subsample=options.subsample_img, time_subsample=options.subsample_time,
                         prepare_segmentations=True, prepare_100doh=True, prepare_video=True,
                         physics_folder_prefix=options.prefix)

data = data.load_physics_state(to_torch=True)

opos = data['trajectory']['obj_pos']
orot = data['trajectory']['obj_rot']
ograsped = data['trajectory'].get('obj_grasped')
hpos = data['trajectory']['hand_pos']
hrot = data['trajectory']['hand_rot']

if options.prefix.startswith('corl'):
    hrot = so3_exp_map(torch.tensor([[0., 0., 180.]]).to(hrot).deg2rad()).expand(hrot.shape[0], 3, 3).bmm(hrot)

viewer = CuboidsWithHandViewer(
    obj_size=data['obj_size'] if len(data['obj_size'].shape) == 1 else data['obj_size'].mean(dim=0),
    static_objs_size=data['static_objs_size'],
    static_objs_pos0=data['static_objs_pos0'],
    static_objs_rot0=data['static_objs_rot0'],
)
if ograsped is None:
    ograsped = torch.zeros(opos.shape[0], 1).to(opos)
viewer.animate(opos, orot, ograsped, hpos, hrot)
