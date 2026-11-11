"""Native supervision loader: observations and simulator labels stay separate."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

INPUT_KEYS = frozenset(('native_clean', 'camera_valid', 'T_world_eef', 'eef_valid',
    'gripper', 'gripper_valid', 'query', 'query_valid', 'camera_id',
    'imaging_config_id', 'imaging_config_valid', 'image_orientation_xy'))


def query_from_capture(d):
    count = len(d['native_clean'])
    parts = [d['query_center_xyz'], d['query_size_wh'],
             np.broadcast_to(d['query_plane_z'], (count, 1)),
             np.zeros((count, 9), dtype=np.float32),
             np.broadcast_to(d['query_texture_hw'], (count, 2)), d['query_orientation_wxyz']]
    widths = [3, 2, 1, 9, 2, 4]
    valid = np.concatenate([np.repeat(d['query_known_valid'][:, i:i+1], n, axis=1)
                            for i, n in enumerate(widths)], axis=1)
    # The captured canonical gauge was derived from simulator camera H/height.
    # It is provenance/label data, not an observation available at deployment.
    valid[:, 6:15] = False
    return np.concatenate(parts, axis=1).astype(np.float32), valid


def image_tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2))).float() / 255


class SpatialTargets(Dataset):
    def __init__(self, root, split, *, include_check=False, verify_files=False):
        self.root = Path(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError('Explicit train/val/test split required')
        if split == 'train' and include_check:
            raise ValueError('Held-out check texture cannot enter the training dataset')
        self.split, self.include_check = split, bool(include_check)
        self.index = json.loads((self.root / 'target_index.json').read_text())['targets']
        groups = defaultdict(set)
        for row in self.index:
            groups[(row['engine'], row['scene_group'])].add(row['split'])
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError('Scene-level split leakage')
        self.rows = [row for row in self.index if row['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty split: {split}')
        if verify_files:
            for row in self.rows:
                path = self.root / row['path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError(f'Checksum mismatch: {path}')

    def __len__(self):
        return len(self.rows)

    def batch_key(self, i):
        row = self.rows[i]
        return (row['engine'], tuple(row['native_hw']), tuple(row['active_camera_indices']))

    def __getitem__(self, i):
        row = self.rows[i]
        with np.load(self.root / row['path'], allow_pickle=False) as archive:
            # Deliberately do not access the held-out image in training.
            keys = [k for k in archive.files if k != 'check_delta' or self.include_check]
            d = {k: archive[k] for k in keys}
        clean = d['native_clean']
        count = len(clean)
        query, query_valid = query_from_capture(d)
        camera_id = d['active_camera_indices'].astype(np.int64)
        engine_config_offset = {'libero': 0, 'robotwin': 3}[row['engine']]
        eef_valid = d['eef_valid'].astype(bool)
        gripper_valid = eef_valid & np.isfinite(d['gripper'])
        inputs = dict(native_clean=image_tensor(clean), camera_valid=torch.ones(count, dtype=torch.bool),
            T_world_eef=torch.from_numpy(d['T_world_eef'].astype(np.float32)),
            eef_valid=torch.from_numpy(eef_valid), gripper=torch.from_numpy(d['gripper'].astype(np.float32)),
            gripper_valid=torch.from_numpy(gripper_valid), query=torch.from_numpy(query),
            query_valid=torch.from_numpy(query_valid), camera_id=torch.from_numpy(camera_id),
            imaging_config_id=torch.from_numpy(camera_id + engine_config_offset),
            imaging_config_valid=torch.ones(count, dtype=torch.bool),
            image_orientation_xy=torch.ones(count, 2))
        if frozenset(inputs) != INPUT_KEYS:
            raise AssertionError('Unexpected model input field')
        reconstructed = {}
        for name in ('black', 'white', 'random') + (('check',) if self.include_check else ()):
            pixels = clean.astype(np.int16) + d[name + '_delta']
            if pixels.min() < 0 or pixels.max() > 255:
                raise ValueError(f'Out-of-range lossless residual: {name}')
            reconstructed[name] = image_tensor(pixels)
        h = d['image_to_texture_H']
        # Old lowdim_valid describes a different 20cm canonical reference
        # square. Determine validity from the actual probe homography instead.
        h_valid = np.isfinite(h).all(axis=(1, 2))
        corners = np.zeros((count, 4, 2), dtype=np.float64)
        texture_points = np.array([[0., 0., 1.], [127., 0., 1.], [127., 127., 1.], [0., 127., 1.]])
        for j in np.flatnonzero(h_valid):
            try:
                norm = np.linalg.norm(h[j])
                if norm == 0:
                    h_valid[j] = False
                    continue
                projected = texture_points @ np.linalg.inv(h[j] / norm).T
                denominator = projected[:, 2]
                near_horizon = np.abs(denominator) <= 1e-12*np.linalg.norm(projected, axis=1)
                same_side = np.all(denominator > 0) or np.all(denominator < 0)
                if np.any(near_horizon) or not same_side:
                    h_valid[j] = False
                else:
                    corners[j] = projected[:, :2] / denominator[:, None]
                    edges = np.roll(corners[j], -1, axis=0) - corners[j]
                    next_edges = np.roll(edges, -1, axis=0)
                    crosses = edges[:, 0]*next_edges[:, 1] - edges[:, 1]*next_edges[:, 0]
                    convex = np.all(crosses > 1e-8) or np.all(crosses < -1e-8)
                    h_valid[j] &= np.isfinite(corners[j]).all() and convex
            except np.linalg.LinAlgError:
                h_valid[j] = False
        targets = dict(B=reconstructed['black'], G=(reconstructed['white']-reconstructed['black']).clamp_min(0),
            H=torch.from_numpy(h.copy()), H_valid=torch.from_numpy(h_valid),
            corners_native=torch.from_numpy(corners).float(), random_native=reconstructed['random'],
            patch_mask=torch.from_numpy(d['patch_mask'].copy()), plane_mask=torch.from_numpy(d['plane_mask'].copy()),
            S_reference=torch.from_numpy(d['sampling_reference'].copy()),
            S_reference_valid=torch.from_numpy(d['sampling_reference_valid'].copy()).bool())
        if self.include_check:
            targets['check_native'] = reconstructed['check']
        metadata = {k: row[k] for k in ('engine', 'task', 'scene_group', 'packed_timestamp', 'split')}
        return inputs, targets, metadata


class ResolutionBatchSampler(Sampler):
    """Group native resolutions/view layouts without changing per-epoch coverage."""
    def __init__(self, dataset, batch_size, *, shuffle=True, seed=19043):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.groups = defaultdict(list)
        for i in range(len(dataset)):
            self.groups[dataset.batch_key(i)].append(i)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for values in self.groups.values():
            order = list(values)
            if self.shuffle:
                rng.shuffle(order)
            batches.extend(order[i:i+self.batch_size] for i in range(0, len(order), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum((len(v)+self.batch_size-1)//self.batch_size for v in self.groups.values())


def collate_targets(batch):
    observations, labels, metadata = zip(*batch)
    return ({k: torch.stack([x[k] for x in observations]) for k in observations[0]},
            {k: torch.stack([x[k] for x in labels]) for k in labels[0]}, list(metadata))
