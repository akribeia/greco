"""Submission-time contract for independent Advanced reconstruction options."""
PROTOCOL='greco-advanced-v9'
STAGE_MAX_PASSES=120
NUMERICAL_SOLVER_MAX_ITERATIONS=180

FIELDS=('advanced_mode','refine_common_axis','refine_per_view_axis','refine_tilt_angles')
EXTRA_FIELDS=('reestimate_ctf','fine_scale_refinement','tomogram_leveling')

def validate(raw=None):
    if raw is None:raw={}
    if not isinstance(raw,dict) or set(raw)-{'enabled',*FIELDS,*EXTRA_FIELDS,'defocus_handedness','flip_tomogram_z','fine_scale_binning','fine_scale_rounds'}:
        raise ValueError('Unknown Advanced reconstruction options.')
    out={key:raw.get(key,False) for key in ('enabled',*FIELDS,*EXTRA_FIELDS,'flip_tomogram_z')}
    if any(type(value) is not bool for value in out.values()):
        raise ValueError('Advanced reconstruction options must be checkboxes.')
    hand=raw.get('defocus_handedness',1)
    if type(hand) is not int or hand not in (-1,1):
        raise ValueError('Defocus handedness must be -1 or +1.')
    binning=raw.get('fine_scale_binning',4);rounds=raw.get('fine_scale_rounds',5)
    if type(binning) is not int or binning not in (1,2,4):raise ValueError('Fine-scale binning must be 4, 2 or 1.')
    if type(rounds) is not int or not 1<=rounds<=20:raise ValueError('Fine-scale refinement requires 1–20 additional passes.')
    out.update(fine_scale_binning=binning,fine_scale_rounds=rounds)
    out['defocus_handedness']=hand
    return out

def resolve(raw=None, *, legacy=False):
    selected=validate(raw)
    active=selected['enabled'] and (selected['advanced_mode'] if legacy else True)
    if not legacy:
        selected['advanced_mode']=bool(selected['enabled'])
        selected['reestimate_ctf']=bool(selected['enabled'])
    effective={key:bool(active and selected[key]) for key in (*FIELDS,*EXTRA_FIELDS,'flip_tomogram_z')}
    effective['defocus_handedness']=selected['defocus_handedness']
    return dict(protocol=PROTOCOL,selected=selected,**effective,fine_scale_binning=selected['fine_scale_binning'],fine_scale_rounds=selected['fine_scale_rounds'],
        shift_median_px=1.0,shift_p90_px=2.5,angular_median_px=1.0,angular_p90_px=2.5,
        stable_iterations=3,consecutive_stage_failure_limit=3 if active else 0,
        stage_continuation_policy='safe_provisional_max_passes',common_rounds=5 if active else 0,
        roll_rounds=5 if effective['refine_per_view_axis'] else 0,
        inclination_rounds=10 if effective['refine_tilt_angles'] else 0,
        progressive_max_passes=STAGE_MAX_PASSES,angular_max_passes=STAGE_MAX_PASSES,
        numerical_solver_max_iterations=NUMERICAL_SOLVER_MAX_ITERATIONS,
        normal_radius_px=4,rescue_radius_px=32,rescue_radii_px=[4,16,32],measurement_rms_ratio=1.02,
        alignment_binning=8,step_fraction=1.,rescue_exit_policy='adaptive_4_16_32_confirm_4',ctf_model='flat_global_from_raw_spectra',alignment_reliability=True,tilt_order_policy='joint_step_before_first_crossing',
        rms_reference_policy='current_search_radius_segment')

def checked_resolved(value):
    """Validate new jobs and preserve the frozen settings of v7 submissions."""
    if not isinstance(value,dict) or 'selected' not in value:raise ValueError('Missing Advanced settings snapshot.')
    legacy=value.get('protocol')=='greco-advanced-v7'
    expected=resolve(value['selected'],legacy=legacy)
    if value.get('protocol') in ('greco-advanced-v7','greco-advanced-v8'):
        if value['selected'].get('fine_scale_refinement'):raise ValueError('Fine-scale refinement requires the current protocol.')
        expected['protocol']=value['protocol']
        compared={key:val for key,val in expected.items() if key in value}
        compared['selected']={key:val for key,val in expected['selected'].items() if key in value['selected']}
        if compared!=value:raise ValueError('Advanced settings differ from their resolved protocol.')
        return expected
    if 'tomogram_leveling' not in value and 'tomogram_leveling' not in value['selected']:
        from copy import deepcopy
        value=deepcopy(value);value['tomogram_leveling']=False;value['selected']['tomogram_leveling']=False
    if value!=expected:raise ValueError('Advanced settings differ from their resolved protocol.')
    return expected
