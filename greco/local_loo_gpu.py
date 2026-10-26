"""Device-resident full-support leave-one-out local WBP models."""

from __future__ import annotations

import numpy as np

from .tomography import gpu_reconstruct_wbp_batch_device, gpu_batch_size


class BatchedLocalLOOWBPGPU:
    """GPU equivalent of :class:`BatchedLocalLOOWBP` without volume transfers."""

    def __init__(self, patches_btyx, angles_deg, valid_bt, batch_size=None):
        import cupy as cp

        self.patches = cp.asarray(patches_btyx, dtype=cp.float32)
        self.angles = np.asarray(angles_deg, dtype=np.float64)
        self.valid_host = np.asarray(valid_bt, dtype=bool)
        self.valid = cp.asarray(self.valid_host)
        if self.patches.ndim != 4 or self.valid.shape != self.patches.shape[:2]:
            raise ValueError("patch/valid shape mismatch")
        if self.angles.shape != (self.patches.shape[1],) or np.any(self.valid_host.sum(axis=1) < 2):
            raise ValueError("LOO requires matching angles and at least two views")
        counts = cp.asarray(self.valid_host.sum(axis=1), dtype=cp.float32)
        if batch_size is None:
            full = gpu_reconstruct_wbp_batch_device(
                self.patches, self.angles, valid_bt=self.valid, y_chunk=8)
            full *= (2 * counts[:, None, None, None] / cp.pi)
            self._sum = full
        else:
            n=self.patches.shape[-1]
            self._sum=cp.empty((len(self.patches),n,n,n),cp.float32)
            for start in range(0,len(self.patches),batch_size):
                end=min(start+batch_size,len(self.patches))
                full=gpu_reconstruct_wbp_batch_device(self.patches[start:end],self.angles,
                    valid_bt=self.valid[start:end],y_chunk=8)
                self._sum[start:end]=full*(2*counts[start:end,None,None,None]/cp.pi)

    def target_models(self, target: int, selected_ids=None):
        """Return host landmark ids and device models excluding ``target``."""
        import cupy as cp

        if not 0 <= target < self.patches.shape[1]:
            raise IndexError(target)
        landmark_ids = np.flatnonzero(
            self.valid_host[:, target] & (self.valid_host.sum(axis=1) >= 2))
        if selected_ids is not None:
            landmark_ids=np.intersect1d(landmark_ids,np.asarray(selected_ids,int))
        if not len(landmark_ids):
            n = self.patches.shape[-1]
            return landmark_ids, cp.empty((0, n, n, n), dtype=cp.float32)
        one = self.patches[landmark_ids, target][:, None]
        contribution = gpu_reconstruct_wbp_batch_device(
            one, self.angles[target:target + 1], y_chunk=8)
        contribution_sum = contribution * (2 / cp.pi)
        loo_count = cp.asarray(
            self.valid_host[landmark_ids].sum(axis=1) - 1, dtype=cp.float32)
        volumes = (self._sum[landmark_ids] - contribution_sum) * (
            cp.pi / (2 * loo_count[:, None, None, None]))
        return landmark_ids, volumes.astype(cp.float32, copy=False)

    def iter_target_models(self, target):
        """Bound LOO, transfer FFT and projection work for each target."""
        ids = np.flatnonzero(self.valid_host[:, target] & (self.valid_host.sum(axis=1) >= 2))
        n = self.patches.shape[-1]
        limit = gpu_batch_size(max(1, len(ids)), n**3 * 128)
        for start in range(0, len(ids), limit):
            yield self.target_models(target, ids[start:start + limit])
