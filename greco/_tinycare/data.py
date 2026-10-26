# Third-party references and adapted components
# Greco implementation of the cryoCARE spatial train/validation split, even/odd patch sampling and normalization protocol. GPU sampling and data handling are Greco contributions.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/cryocare/LICENSE.txt
#
# Copyright (c) 2020, juglab
# Full license conditions and disclaimers are retained at the paths above.
#

"""Spatially disjoint even/odd patches; sampling and augmentation stay on GPU."""
import numpy as np
import torch


def coordinates(shape, patch, low_y, high_y, count, rng):
    # Match the established upper-exclusive origin range (no boundary padding).
    extent = np.array((shape[0] - patch, high_y - low_y - patch, shape[2] - patch))
    if (extent <= 0).any():
        raise ValueError('TinyCARE has no valid patch origins in the requested split.')
    total = int(np.prod(extent))
    ids = rng.choice(total, count, replace=total < count)
    coords = np.array(np.unravel_index(ids, extent)).T
    coords[:, 1] += low_y
    return coords.astype(np.int64)


class PatchData:
    def __init__(self, even, odd, patch, *, device, seed=17, samples=1200, normalization_samples=500):
        self.rng = np.random.default_rng(seed)
        self.patch = int(patch)
        cut = int(even.shape[1] * .9) - 1
        if even.shape[1] - cut < patch or cut < patch:
            cut = even.shape[1] - patch - 1
        self.train = coordinates(even.shape, patch, 0, cut, int(samples * .9), self.rng)
        self.val = coordinates(even.shape, patch, cut, even.shape[1], int(samples * .1), self.rng)
        self.device = torch.device(device)
        # Retain only the two volumes, not thousands of materialized 72³ patches.
        cache_device = self.device
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            if even.nbytes + odd.nbytes > free * .2:
                cache_device = torch.device('cpu')
        self.even = torch.as_tensor(even, device=cache_device)
        self.odd = torch.as_tensor(odd, device=cache_device)
        self.offset = torch.arange(patch, device=cache_device)
        self.volume_cache = cache_device.type
        self.order = self.rng.permutation(len(self.train))
        self.cursor = 0
        self.mean, self.std = 0., 1.
        # Streaming moments avoid the old ~750 MB temporary sample list.
        total = total2 = 0.
        count = 0
        for start in range(0, normalization_samples, 8):
            ids = np.arange(start, min(start + 8, normalization_samples)) % len(self.train)
            x, _ = self.batch(self.train[ids], augment=True, normalize=False)
            value = x.double()
            total += value.sum().item()
            total2 += value.square().sum().item()
            count += value.numel()
        self.mean = total / count
        self.std = float(np.sqrt(max(0., total2 / count - self.mean ** 2)))
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError('TinyCARE input has zero or invalid variance.')

    def batch(self, coords, *, augment, normalize=True):
        points = torch.as_tensor(coords, device=self.even.device)
        offset = self.offset
        z = points[:, 0, None, None, None] + offset[None, :, None, None]
        y = points[:, 1, None, None, None] + offset[None, None, :, None]
        x = points[:, 2, None, None, None] + offset[None, None, None, :]
        even = self.even[z, y, x].unsqueeze(1)
        odd = self.odd[z, y, x].unsqueeze(1)
        if even.device.type == 'cpu' and self.device.type == 'cuda':
            # Very large tomograms need not fit in VRAM: stage bounded batches.
            even = even.pin_memory().to(self.device, non_blocking=True)
            odd = odd.pin_memory().to(self.device, non_blocking=True)
        if augment:
            rotations = self.rng.integers(0, 4, len(coords))
            for k in (1, 2, 3):
                ids = np.flatnonzero(rotations == k)
                if len(ids):
                    ids = torch.as_tensor(ids, device=self.device)
                    even[ids] = torch.rot90(even[ids], k, dims=(2, 4))
                    odd[ids] = torch.rot90(odd[ids], k, dims=(2, 4))
        # Both train and held-out streams previously swapped even/odd direction.
        swap = torch.as_tensor(self.rng.random(len(coords)) > .5, device=self.device)[:, None, None, None, None]
        first, second = torch.where(swap, odd, even), torch.where(swap, even, odd)
        if normalize:
            first = (first - self.mean) / self.std
            second = (second - self.mean) / self.std
        return first.contiguous(memory_format=torch.channels_last_3d), second

    def next_train(self, count):
        indices = []
        while count:
            take = min(count, len(self.order) - self.cursor)
            indices.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            count -= take
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.train))
                self.cursor = 0
        return self.batch(self.train[indices], augment=True)
