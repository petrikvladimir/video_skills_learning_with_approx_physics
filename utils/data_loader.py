#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-01-19
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from pathlib import Path
import pickle as pkl

import cv2
import numpy as np
import imageio
import torch

from utils.loss import mask_center
from utils.preprocessing import estimate_colors, estimate_is_object_initially_moving, \
    estimate_is_object_moving_at_the_end


def _default_input_data():
    return Path(__file__).absolute().parent.parent.joinpath('input_data')


def _default_cache_data():
    return Path(__file__).absolute().parent.parent.joinpath('cache_data')


def load_100doh(vid, folder=None):
    """ Load detections of the hand and objects detected using https://github.com/ddshan/hand_object_detector """
    if folder is None:
        folder = _default_input_data().joinpath('100doh')
    data = pkl.load(open(Path(folder).joinpath(f'{vid}/detections.pkl'), 'rb'))
    data = {r['image_path']: r for r in data}.values()  # remove duplicates
    return sorted(data, key=lambda d: d['image_path'])


def load_frankmocap(vid, folder=None):
    if folder is None:
        folder = _default_input_data().joinpath('frankmocap')
    pathlist = sorted(Path(folder).joinpath(f'{vid}/mocap/').glob('*.pkl'))
    data = []
    for path in pathlist:
        data.append(pkl.load(open(path, 'rb')))
    return data


def load_video(vid, folder=None):
    if folder is None:
        folder = _default_input_data().joinpath('videos_mp4')
    return np.stack(imageio.mimread(Path(folder).joinpath(f'{vid}.mp4')))


def load_segmentations(vid, folder=None):
    if folder is None:
        folder = _default_input_data().joinpath('segmentations')
    imgs_list = imageio.mimread(Path(folder).joinpath(f'{vid}.gif'))
    for i in range(len(imgs_list)):
        if imgs_list[i].shape[2] == 3:
            imgs_list[i] = np.concatenate((imgs_list[i], 255 * np.ones(imgs_list[i].shape[:2] + (1,))), axis=-1)
    return np.stack(imgs_list[0:1] + imgs_list)  # todo: why segmentation is -1 frame?


def get_hand_key_points_names():
    """ Return names of the keypoint in format 'palm' or {fingername}_{id_from_palm}"""
    return ['palm'] + \
           [f'thumb_{i}' for i in range(4)] + \
           [f'index_{i}' for i in range(4)] + \
           [f'middle_{i}' for i in range(4)] + \
           [f'ring_{i}' for i in range(4)] + \
           [f'pinky_{i}' for i in range(4)]


def hand_finger_tips_ids():
    return [4, 8, 12, 16, 20]


class DataLoaderWrapper:
    def __init__(self, vid, time_subsample=1, img_subsample=1,
                 prepare_video=False, prepare_frankmocap=False, prepare_100doh=False, prepare_segmentations=False,
                 physics_folder_prefix=None,
                 ) -> None:
        self.physics_folder_prefix = physics_folder_prefix
        self.vid = vid
        self.time_subsample = time_subsample
        self.img_subsample = img_subsample
        self.data_video = load_video(vid) if prepare_video else None
        self.data_frankmocap = load_frankmocap(vid) if prepare_frankmocap else None
        self.data_100doh = load_100doh(vid) if prepare_100doh else None
        self.data_segmentations = load_segmentations(vid) if prepare_segmentations else None
        self._subsample(time_subsample=time_subsample, img_subsample=img_subsample)
        self._estimated_colors = None

    @property
    def in_contact(self):
        raw_contacts = np.array(
            [d['hand'] is not None and d['obj'] is not None and np.any(d['hand'][:, 5] > 0) for d in self.data_100doh]
        )
        filtered = np.convolve(raw_contacts, np.array([1, 1, 1, 1, 1]), mode='same') >= 3
        if np.sum(filtered) == 0:  # we assume there is contact, if not after filtering, return unfiltered
            return torch.from_numpy(raw_contacts)
        return torch.from_numpy(filtered)

    @property
    def in_contact_with_fix_start_end(self):
        in_contact = self.in_contact.clone()
        first_contact_ind, last_contact_ind = torch.where(in_contact)[0][[0, -1]]
        if estimate_is_object_initially_moving(self):
            in_contact[:first_contact_ind] = True
        if estimate_is_object_moving_at_the_end(self):
            in_contact[last_contact_ind:] = True
        return in_contact.detach()

    @property
    def in_contact_with_fix_start_end_release(self):
        in_contact = self.in_contact_with_fix_start_end.clone()
        if not in_contact[-1]:
            obj_mask = self.manipulated_obj_masks()
            cx, cy = mask_center(obj_mask)
            last_contact = torch.where(in_contact)[0][-1]
            min_ind = max(last_contact - 12, 0)
            min_ind = min_ind + torch.where(in_contact[min_ind:last_contact + 1])[0][0]
            if torch.any(torch.isnan(cx[min_ind:last_contact + 1])):
                min_ind = torch.where(torch.isnan(cx[:last_contact + 1]))[0][-1] + 1
            a = cx[min_ind:last_contact + 1]
            b = cy[min_ind:last_contact + 1]
            v = (np.square(np.diff(a)) + np.square(np.diff(b))) < 20
            if np.any(v):
                release_ind = int(np.median(np.nonzero(v)[0]))
                in_contact[min_ind + release_ind + 1:] = False
        return in_contact

    def _ensure_colors_estimated(self):
        assert self.data_segmentations is not None
        assert self.data_100doh is not None
        if self._estimated_colors is None:
            self._estimated_colors = estimate_colors(self.data_100doh, self.data_segmentations)

    @property
    def hand_color(self):
        self._ensure_colors_estimated()
        return self._estimated_colors[0]

    @property
    def manipulated_obj_color(self):
        self._ensure_colors_estimated()
        return self._estimated_colors[1]

    @property
    def other_objs_color(self):
        self._ensure_colors_estimated()
        return self._estimated_colors[2]

    def _subsample(self, time_subsample, img_subsample):
        if self.data_video is not None:
            self.data_video = self.data_video[::time_subsample, ::img_subsample, ::img_subsample]

        if self.data_segmentations is not None:
            self.data_segmentations = self.data_segmentations[::time_subsample, ::img_subsample, ::img_subsample]

        if self.data_100doh is not None:
            self.data_100doh = self.data_100doh[::time_subsample]
            for d in self.data_100doh:
                if d['hand'] is not None:
                    d['hand'][:, :4] /= img_subsample
                if d['obj'] is not None:
                    d['obj'][:, :4] /= img_subsample

        if self.data_frankmocap is not None:
            assert img_subsample != 1, 'Image subsample not supported for frankmocap'
            self.data_frankmocap = self.data_frankmocap[::time_subsample]

    def _mask_from_color(self, color, filter=True):
        """ Compute the binary masks from segmentations based on the color. Return float pytorch tensor. """
        assert self.data_segmentations is not None
        if not filter:
            return torch.all(torch.from_numpy(self.data_segmentations) == color, dim=-1).float()
        masks = np.all(self.data_segmentations == color.numpy(), axis=-1)
        obj_area = np.median(masks.sum(axis=(-1, -2)))
        for i in range(masks.shape[0]):  # filter small components
            num_labels, components = cv2.connectedComponents(masks[i].astype(np.uint8), connectivity=8)
            if num_labels > 1:
                components_area = [(components == i).sum() for i in range(1, num_labels)]
                max_component_i = np.argmax(components_area) + 1
                for j, area in zip(range(1, num_labels), components_area):
                    if not (j == max_component_i and area > 0.1 * obj_area):
                        masks[i, components == j] = False
        return torch.from_numpy(masks).float()

    def hand_masks(self, filter=True):
        """ Compute the hand binary masks from segmentations based on the color. Return float pytorch tensor. """
        return self._mask_from_color(self.hand_color, filter)

    def manipulated_obj_masks(self, filter=True):
        """ Compute the manipulated object binary masks from segmentations based on the color.
            Returns float pytorch tensor. """
        return self._mask_from_color(self.manipulated_obj_color, filter)

    def first_other_obj_masks(self, filter=True):
        """ Compute the first other object binary masks from segmentations based on the color.
            Returns float pytorch tensor. """
        return self._mask_from_color(self.other_objs_color[0], filter)

    def has_other_object(self):
        """ Return true if other object has been detected in segmentation mask."""
        return self.other_objs_color.shape[0] > 0

    def num_frames(self):
        """ Return number of frames of the data by assuming they all give the same number of frames. """
        if self.data_segmentations is not None:
            return self.data_segmentations.shape[0]
        if self.data_video is not None:
            return self.data_video.shape[0]
        if self.data_100doh is not None:
            return len(self.data_100doh)
        if self.data_frankmocap is not None:
            return len(self.data_frankmocap)

    def write_hand_pose_initialization(self, hand_pos, hand_rot, images_dict=None):
        """ Write hand pose initialization into the pkl together with the rendered animation. """
        folder = _default_cache_data().joinpath('hand_pose_initialization')
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath(f'pos_rot_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = {
            'hand_pos': hand_pos.detach().cpu().numpy(),
            'hand_rot': hand_rot.detach().cpu().numpy()
        }
        pkl.dump(data, open(file, "wb"))
        if images_dict is not None:
            for name, img in images_dict.items():
                imageio.mimwrite(
                    folder.joinpath(f'render_{self.vid}_{self.img_subsample}_{self.time_subsample}_{name}.gif'),
                    img
                )

    def load_hand_pose_initialization(self, to_torch=False, device='cpu', dtype=torch.float32):
        """ Return numpy arrays with hand positions and orientations. """
        folder = _default_cache_data().joinpath('hand_pose_initialization')
        file = folder.joinpath(f'pos_rot_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = pkl.load(open(file, "rb"))
        if to_torch:
            return (torch.from_numpy(data['hand_pos']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['hand_rot']).to(device=device, dtype=dtype))
        return data['hand_pos'], data['hand_rot']

    def write_manipulated_obj_initialization(self, obj_pos, obj_rot, obj_size, images_dict=None):
        """ Write obj pose initialization into the pkl and store the rendered animation in gifs. """
        folder = _default_cache_data().joinpath('manipulated_obj_initialization')
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath(f'pos_rot_size_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = {
            'obj_pos': obj_pos.detach().cpu().numpy(),
            'obj_rot': obj_rot.detach().cpu().numpy(),
            'obj_size': obj_size.detach().cpu().numpy()
        }
        pkl.dump(data, open(file, "wb"))
        if images_dict is not None:
            for name, img in images_dict.items():
                imageio.mimwrite(
                    folder.joinpath(f'render_{self.vid}_{self.img_subsample}_{self.time_subsample}_{name}.gif'),
                    img
                )

    def load_manipulated_obj_initialization(self, to_torch=False, device='cpu', dtype=torch.float32):
        """ Return numpy arrays or torch tensors with obj positions, orientations, and size. """
        folder = _default_cache_data().joinpath('manipulated_obj_initialization')
        file = folder.joinpath(f'pos_rot_size_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = pkl.load(open(file, "rb"))
        if to_torch:
            return (torch.from_numpy(data['obj_pos']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['obj_rot']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['obj_size']).to(device=device, dtype=dtype),)
        return data['obj_pos'], data['obj_rot'], data['obj_size']

    def write_obj_hand_initialization(self, obj_pos, obj_rot, obj_size, hand_pos, hand_rot, images_dict=None):
        """ Write obj pose initialization into the pkl and store the rendered animation in gifs. """
        folder = _default_cache_data().joinpath('obj_hand_initialization')
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath(f'pos_rot_size_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = {
            'obj_pos': obj_pos.detach().cpu().numpy(),
            'obj_rot': obj_rot.detach().cpu().numpy(),
            'obj_size': obj_size.detach().cpu().numpy(),
            'hand_pos': hand_pos.detach().cpu().numpy(),
            'hand_rot': hand_rot.detach().cpu().numpy()
        }
        pkl.dump(data, open(file, "wb"))
        if images_dict is not None:
            for name, img in images_dict.items():
                imageio.mimwrite(
                    folder.joinpath(f'render_{self.vid}_{self.img_subsample}_{self.time_subsample}_{name}.gif'),
                    img
                )

    def load_obj_hand_initialization(self, to_torch=False, device='cpu', dtype=torch.float32):
        """ Return numpy arrays or torch tensors with obj positions, orientations, size and hand pos and rot. """
        folder = _default_cache_data().joinpath('obj_hand_initialization')
        file = folder.joinpath(f'pos_rot_size_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        time_subsample_needed = False
        if not file.exists():
            print('Warning: exact subsample file does not exists. We use original size/time and subsample.')
            file = folder.joinpath(f'pos_rot_size_{self.vid}_{1}_{1}.pkl')
            time_subsample_needed = True
        data = pkl.load(open(file, "rb"))
        if time_subsample_needed:
            for k in ['obj_pos', 'obj_rot', 'hand_pos', 'hand_rot']:
                data[k] = data[k][::self.time_subsample]
        if to_torch:
            return (torch.from_numpy(data['obj_pos']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['obj_rot']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['obj_size']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['hand_pos']).to(device=device, dtype=dtype),
                    torch.from_numpy(data['hand_rot']).to(device=device, dtype=dtype),)
        return data['obj_pos'], data['obj_rot'], data['obj_size'], data['hand_pos'], data['hand_rot']

    def write_physics_state(self, simulation=None, camera_transform=None,
                            obj_pos=None, obj_rot=None, obj_grasped=None, hand_pos=None, hand_rot=None,
                            images_dict=None):
        """ Write state required to reproduce physics optimization. """
        prefix = self.physics_folder_prefix
        folder = _default_cache_data().joinpath('physics_state' if prefix is None else f'{prefix}/physics_state')
        folder.mkdir(parents=True, exist_ok=True)
        if simulation is not None and camera_transform is not None:
            data = {
                'obj_pos0': simulation.obj_pos0.detach().cpu().numpy(),
                'obj_rot6d0': simulation.obj_rot6d0.detach().cpu().numpy(),
                'obj_size': simulation.obj_size.detach().cpu().numpy(),
                'obj_vel0': simulation.obj_vel0.detach().cpu().numpy(),
                'obj_omega0': simulation.obj_omega0.detach().cpu().numpy(),
                'hand_spline_y': simulation.hand.spline_y.detach().cpu().numpy(),
                'hand_pos0': simulation.hand.pos0.detach().cpu().numpy(),
                'hand_rot6d0': simulation.hand.rot6d0.detach().cpu().numpy(),
                'cam_elevation': camera_transform.elevation.detach().cpu().numpy(),
                'cam_tz': camera_transform.cam2world_tz.detach().cpu().numpy(),
                'max_t': simulation.max_t,
            }
            if simulation.number_of_static_objects > 0:
                data['static_objs_pos0'] = simulation.static_objs_pos0.detach().cpu().numpy()
                data['static_objs_rot0'] = simulation.static_objs_rot0.detach().cpu().numpy()
                data['static_objs_size'] = simulation.static_objs_size.detach().cpu().numpy()
            # save trajectories
            data['trajectory'] = {
                'obj_pos': obj_pos.detach().cpu().numpy(),
                'obj_rot': obj_rot.detach().cpu().numpy(),
                'obj_grasped': obj_grasped.detach().cpu().numpy(),
                'hand_pos': hand_pos.detach().cpu().numpy(),
                'hand_rot': hand_rot.detach().cpu().numpy()
            }

            file = folder.joinpath(f'physics_state_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
            pkl.dump(data, open(file, "wb"))

        if images_dict is not None:
            for name, img in images_dict.items():
                imageio.mimwrite(
                    folder.joinpath(f'render_{self.vid}_{self.img_subsample}_{self.time_subsample}_{name}.gif'),
                    img
                )

    def physics_state_computed(self):
        """ Return true if physics state was computed for the current id. """
        prefix = self.physics_folder_prefix
        folder = _default_cache_data().joinpath('physics_state' if prefix is None else f'{prefix}/physics_state')
        file = folder.joinpath(f'physics_state_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        return file.exists()

    def load_physics_state(self, to_torch=False, device='cpu', dtype=torch.float32):
        """ Return numpy arrays or torch tensors with obj positions, orientations, size and hand pos and rot. """
        prefix = self.physics_folder_prefix
        folder = _default_cache_data().joinpath('physics_state' if prefix is None else f'{prefix}/physics_state')
        file = folder.joinpath(f'physics_state_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = pkl.load(open(file, "rb"))
        if to_torch:
            param_keys = ['obj_pos0', 'obj_rot6d0', 'obj_size', 'obj_vel0', 'obj_omega0', 'hand_spline_y', 'hand_pos0',
                          'hand_rot6d0', 'cam_elevation', 'cam_tz', 'static_objs_pos0', 'static_objs_rot0',
                          'static_objs_size', 'max_t']
            tkeys = ['obj_pos', 'obj_rot', 'obj_grasped', 'hand_pos', 'hand_rot']
            for k in param_keys:
                if k in data.keys():
                    if not isinstance(data[k], torch.Tensor):
                        data[k] = torch.from_numpy(data[k]).to(device=device, dtype=dtype)
                    else:
                        data[k] = data[k].to(device=device, dtype=dtype)
                else:
                    data[k] = None
            for k in tkeys:
                if k in data['trajectory'].keys():
                    if not isinstance(data['trajectory'][k], torch.Tensor):
                        data['trajectory'][k] = torch.from_numpy(data['trajectory'][k]).to(device=device, dtype=dtype)
                    else:
                        data['trajectory'][k] = data['trajectory'][k].to(device=device, dtype=dtype)
        return data

    def write_robot_motion(self, trajectories, actions, obj_poses):
        prefix = self.physics_folder_prefix
        folder = _default_cache_data().joinpath('robot_motion' if prefix is None else f'{prefix}/robot_motion')
        folder.mkdir(parents=True, exist_ok=True)
        file = folder.joinpath(f'robot_motion_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = {
            'trajectories': trajectories,
            'actions': actions,
            'obj_poses': obj_poses,
        }
        pkl.dump(data, open(file, "wb"))

    def load_robot_motion(self):
        prefix = self.physics_folder_prefix
        folder = _default_cache_data().joinpath('robot_motion' if prefix is None else f'{prefix}/robot_motion')
        file = folder.joinpath(f'robot_motion_{self.vid}_{self.img_subsample}_{self.time_subsample}.pkl')
        data = pkl.load(open(file, "rb"))
        return data

    @staticmethod
    def all_video_ids(vids='all'):
        """ Vids is one of the following:
        'all' -- all videos
        'one' -- single object videos
        'two' -- two object videos
        'pull_left_to_right'
        'pull_right_to_left'
        'push_left_to_right'
        'push_right_to_left'
        'pick_up'
        'put_next_to'
        'put_in_front_of'
        'put_behind'
        'put_onto'
        """
        if vids == 'pull_left_to_right':
            return ['8675', '13956', '6848', '20193', '10960', '15606']
        if vids == 'pull_right_to_left':
            return ['13309', '3107', '11240', '2458', '10116', '11732']
        if vids == 'push_left_to_right':
            return ['3694', '601', '9889', '4018', '6955', '6132']
        if vids == 'push_right_to_left':
            return ['3949', '1987', '4218', '4378', '1040', '8844']
        if vids == 'pick_up':
            return ['7194', '38559', '12359', '1838', '2875', '24925']
        if vids == 'put_next_to':
            return ['4053', '19', '874', '3340', '1890', '1642']
        if vids == 'put_in_front_of':
            return ['1663', '41177', '28603', '5967', '14841', '2114']
        if vids == 'put_behind':
            return ['1044', '6538', '3074', '12986', '779', '4388']
        if vids == 'put_onto':
            return ['8801', '7655', '13390', '757', '7504', '16310']

        one = ['6848', '8675', '10960', '13956', '15606', '20193', '2458', '3107', '10116', '11240', '11732', '13309',
               '601', '3694', '4018', '6132', '6955', '9889', '1040', '1987', '3949', '4218', '4378', '8844', '7194',
               '38559', '12359', '1838', '2875', '24925']
        if vids == 'one':
            return one
        two = ['1044', '6538', '3074', '12986', '779', '4388', '1663', '41177', '28603', '5967', '14841', '2114',
               '4053', '19', '874', '3340', '1890', '1642', '8801', '7655', '13390', '757', '7504', '16310']
        if vids == 'two':
            return two
        return one + two
