# Greco optimizer — implementation of the GRECO algorithm
# GRECO: Geometric Registration of Electron Cryotomography Observations
# Copyright (C) 2026 Carlos Fernández Rodríguez
#
# This file is part of Greco.
#
# Greco is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3 only.
#
# Greco is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Greco. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Fast-mode continuation after safe, unconverged refinement stages."""
import numpy as np


class StageContinuation:
    def __init__(self,limit=3):
        if type(limit) is not int or limit<1:raise ValueError('Stage failure limit must be a positive integer')
        self.limit=limit;self.failures=0;self.stopped=False

    @property
    def config(self):
        return dict(protocol='safe-provisional-stages-v1',consecutive_failure_limit=self.limit)

    def reject_incoming(self,manifest,previous):
        """A rejected expansion does not invalidate a previously accepted core."""
        if manifest['stop_reason'] not in ('insufficient_supported_landmarks','insufficient_measured_observations'):
            return None
        if previous is None:return None
        valid=np.asarray(previous['valid_views'],bool)
        safe=bool(valid.ndim==2 and valid.shape[0]>0 and valid.shape[1]>=3
            and np.all(valid.sum(0)>=8)
            and all(np.isfinite(np.asarray(previous[key],float)).all()
                    for key in ('positions_zyx','cumulative_shifts_yx','angles_deg'))
            and np.isfinite(np.asarray(previous.get('axis_rotation_deg',0.),float)).all())
        if not safe:return None
        return dict(accepted=False,converged=False,provisional=False,
                    incoming_rejected=True,consecutive_stage_failures=self.failures,
                    failure_limit_reached=False,continuation_stopped=False)

    def reject_extra(self, manifest, previous, *, allow_tilt_order=False):
        """Discard an unsupported optional round while retaining accepted geometry."""
        tilted=allow_tilt_order and manifest['stop_reason']=='tilt_order_step_stalled'
        checked=dict(manifest,stop_reason='insufficient_supported_landmarks') if tilted else manifest
        decision = self.reject_incoming(checked, previous)
        if decision is None:
            return None
        decision.pop('incoming_rejected')
        decision['extra_round_rejected'] = True
        if tilted:decision['angular_refinement_stopped']=True
        return decision

    def assess(self,manifest,state):
        reason=manifest['stop_reason'];converged=reason=='converged'
        if converged:self.failures=0
        else:self.failures+=1
        reached=self.failures>=self.limit
        provisional=False
        if reason=='max_passes' and state is not None:
            history=manifest.get('history',[])
            valid=np.asarray(state['valid_views'],bool)
            safe=bool(history and history[-1].get('applied') and valid.ndim==2
                and valid.shape[0]>0 and valid.shape[1]>=3 and np.all(valid.sum(0)>=8)
                and all(np.isfinite(np.asarray(state[key],float)).all()
                    for key in ('positions_zyx','cumulative_shifts_yx','angles_deg'))
                and np.isfinite(np.asarray(state.get('axis_rotation_deg',0.),float)).all())
            provisional=safe and not reached
        hard_failure=not converged and not provisional and reason!='incoming_cc_rejected'
        self.stopped=bool(reached or hard_failure)
        return dict(accepted=bool(converged or provisional),converged=converged,
                    provisional=provisional,consecutive_stage_failures=self.failures,
                    failure_limit_reached=reached,continuation_stopped=self.stopped)


def refine_supported_extra(runner, original, angles, landmarks, shifts, *, exclude_unsupported=True, **kwargs):
    """Retry optional refinement from its initial geometry without unsupported views.

    Exclusions are tentative until the caller accepts the returned candidate.
    Never reuse the geometry of a failed fit.
    """
    keep=np.arange(len(angles)); attempts=[]; elapsed=0.
    view_matrices={'valid_views','per_view_scores','source_direction_mask','detector_valid_views'}
    view_vectors={'support_views','angles_deg'}
    while True:
        subset={key:(np.asarray(value)[:,keep] if key in view_matrices else
                     np.asarray(value)[keep] if key in view_vectors else value)
                for key,value in landmarks.items()}
        options=dict(kwargs)
        for key in ('frozen_shifts_yx','diag_plane_map'):
            if options.get(key) is not None:options[key]=np.asarray(options[key])[keep]
        axis=np.asarray(options.get('axis_deg'))
        if axis.ndim:options['axis_deg']=axis[keep]
        state,manifest=runner(original[keep],np.asarray(angles)[keep],subset,np.asarray(shifts)[keep],**options)
        elapsed+=manifest.get('runtime_s',0.)
        reason=manifest['stop_reason']
        counts=np.asarray(manifest.get('support_counts_per_view',state['valid_views'].sum(0)))
        bad=counts<8
        if not exclude_unsupported or reason not in ('insufficient_supported_landmarks','insufficient_measured_observations') or not bad.any():break
        remaining=keep[~bad]
        support=np.asarray(landmarks['valid_views'],bool)[:,remaining].copy()
        support[support.sum(1)<4]=False
        if len(remaining)<4 or np.any(support.sum(0)<8):break
        attempts.append(dict(excluded_local_indices=keep[bad].tolist(),reason=reason,
                             support_counts=counts[bad].tolist()))
        keep=remaining
    manifest.update(view_exclusion_attempts=attempts,retained_local_indices=keep.tolist(),runtime_s=elapsed)
    return state,manifest,keep,subset
