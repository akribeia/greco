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
