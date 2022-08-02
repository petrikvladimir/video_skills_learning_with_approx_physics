#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2022-01-31
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import cv2
import numpy as np
import torch


def render_frames(cameras, images, pos, rot, l=0.1):
    """ Render frames into given images. """
    n = images.shape[0]
    c, r = pos, rot
    o = cameras.transform_points_screen(c)
    x = cameras.transform_points_screen(c + r.bmm(torch.tensor([[l, 0, 0]]).unsqueeze(-1).repeat(n, 1, 1)).squeeze(-1))
    y = cameras.transform_points_screen(c + r.bmm(torch.tensor([[0, l, 0]]).unsqueeze(-1).repeat(n, 1, 1)).squeeze(-1))
    z = cameras.transform_points_screen(c + r.bmm(torch.tensor([[0, 0, l]]).unsqueeze(-1).repeat(n, 1, 1)).squeeze(-1))

    ns = torch.arange(0, n, 1)
    o = o[ns, ns, :2].int().numpy()
    x = x[ns, ns, :2].int().numpy()
    y = y[ns, ns, :2].int().numpy()
    z = z[ns, ns, :2].int().numpy()

    for i in range(0, n, 1):
        images[i] = cv2.line(images[i], o[i], x[i], (255, 0, 0), thickness=2)
        images[i] = cv2.line(images[i], o[i], y[i], (0, 255, 0), thickness=2)
        images[i] = cv2.line(images[i], o[i], z[i], (0, 0, 255), thickness=2)
    return images


def resize_batch_of_images(images, scale=1.):
    """ Rescale the batch of images. """
    shape = images.shape[1:3]
    nshape = (np.array(shape) * scale).astype(np.int)
    resized_tensor = np.empty((images.shape[0], nshape[0], nshape[1], 4), dtype=images.dtype)
    for (k, image) in enumerate(images):
        resized_tensor[k] = cv2.resize(image, dsize=(nshape[1], nshape[0]))
    return resized_tensor
