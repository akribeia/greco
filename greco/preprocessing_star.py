"""Preprocessing STAR tables in the current Greco format.

TiltIndex/MdocIndex are one-based on disk and zero-based internally. Frozen shifts are native
raster pixels before viewer rotation/reflection, never geometry corrections.
The legacy field df_hint_A actually stores abs(MDOC defocus_um)*1000 (nm).
"""
import json,math,os,tempfile
from pathlib import Path
from .particle_star import rows as star_rows,write_loop

ALIGNMENT={
 'plane':'grecoTiltIndex','mdoc_index':'grecoMdocIndex','angle_deg':'grecoTiltAngleDeg',
 'dx_raw':'grecoFrozenShiftXNativePixel','dy_raw':'grecoFrozenShiftYNativePixel'}
CTF={
 'z_value':'grecoMdocIndex','tilt_deg':'grecoTiltAngleDeg',
 'dx_fullres':'grecoFrozenShiftXNativePixel','dy_fullres':'grecoFrozenShiftYNativePixel',
 'defocus_A':'grecoCentralDefocusMeanAngst','df_hint_A':'grecoMdocDefocusHintNM',
 'astig_azimuth_raw_deg':'grecoDefocusAngleDeg','astig_magnitude_A':'grecoAstigmatismDifferenceAngst',
 'astig_azimuth_effective_deg':'grecoAlignedDefocusAngleDeg',
 'ctf_transfer_min':'grecoCtfTransferMinimum','ctf_inverse_abs_max':'grecoAlignedCtfInverseAbsMaximum',
 'raw_ctf_inverse_abs_max':'grecoRawCtfInverseAbsMaximum','assembly_runtime_s':'grecoAssemblyRuntimeSeconds',
 'reliability_half_mean':'grecoHalfReliabilityMean','reliability_full_mean':'grecoFullReliabilityMean',
 'phase_agreement_mean':'grecoPhaseAgreementMean','scale_even':'grecoEvenScaleFactor','scale_odd':'grecoOddScaleFactor',
 'reliability_full_mean_bin1':'grecoBin1FullReliabilityMean','ctf_inverse_abs_max_bin1':'grecoBin1CtfInverseAbsMaximum',
 'reliability_full_mean_raw_bin1':'grecoNativeFullReliabilityMean'}
SCHEMAS={'alignment':ALIGNMENT,'ctf':CTF}


def read_table(path):
    """Expose the existing internal field names, preserving their exact units."""
    path=Path(path)
    if path.suffix.lower()!='.star':
        raise ValueError('Preprocessing metadata must use Greco STAR tables.')
    general=None;result=[]
    for block,row in star_rows(path):
        if block=='general':
            if general is not None:raise ValueError('Duplicate preprocessing STAR header.')
            general=row
            if row.get('grecoFormatName')!='greco-preprocessing-v1' or row.get('grecoFormatVersion')!=1:raise ValueError('Unknown preprocessing STAR format.')
            kind=row.get('grecoTableKind')
            if kind not in SCHEMAS:raise ValueError('Unknown preprocessing STAR table.')
        elif block in SCHEMAS:
            if general is None or block!=general['grecoTableKind']:raise ValueError('Preprocessing STAR block/header mismatch.')
            result.append({key:('' if row[label] is None else str(row[label])) for key,label in SCHEMAS[block].items() if label in row})
    if general is None:raise ValueError('Missing preprocessing STAR header.')
    return result


def _value(value):
    if value is None or value=='':return None
    if isinstance(value,(int,float)):
        if not math.isfinite(value):raise ValueError('Non-finite preprocessing metadata.')
        return value
    value=str(value)
    try:return int(value)
    except ValueError:
        try:
            number=float(value)
            if not math.isfinite(number):raise ValueError('Non-finite preprocessing metadata.')
            return number
        except ValueError:
            if value.lower() in ('nan','inf','-inf'):raise ValueError('Non-finite preprocessing metadata.')
            return value


def write_table(path,records,kind):
    path=Path(path);records=list(records)
    if not records:raise ValueError('Cannot write an empty preprocessing table.')
    keys=list(records[0])
    if any(set(row)!=set(keys) for row in records):raise ValueError('Inconsistent preprocessing columns.')
    if path.suffix.lower()!='.star':
        raise ValueError('Preprocessing metadata must use Greco STAR tables.')
    schema=SCHEMAS[kind]
    if set(keys)-schema.keys():raise ValueError('Undefined Greco preprocessing labels: '+str(set(keys)-schema.keys()))
    columns=[schema[key] for key in keys];values=[]
    extras=['grecoCentralDefocusUAngst','grecoCentralDefocusVAngst','grecoCtfAxisDeg',
            'grecoDefocusHand','grecoCtfAlphaOffsetDeg','grecoCtfBetaOffsetDeg'] if kind=='ctf' else []
    for row in records:
        fields=[_value(row[key]) for key in keys]
        if extras:
            folder=path.parent/'tilts'/f"tilt_{int(row['z_value']):04d}"
            fit=json.loads((folder/'ctf.json').read_text()) if (folder/'ctf.json').is_file() else {}
            geometry=json.loads((folder/'ctf_geometry.json').read_text()) if (folder/'ctf_geometry.json').is_file() else fit.get('local_geometry',{})
            fields += [fit.get('defocus_min_A'),fit.get('defocus_max_A'),geometry.get('tilt_axis_deg'),
                       geometry.get('handedness'),geometry.get('alpha_offset_deg'),geometry.get('beta_offset_deg')]
        values.append(fields)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.star-',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            write_loop(stream,'general',['grecoFormatName','grecoFormatVersion','grecoTableKind','grecoFrozenShiftConvention'],
                       [('greco-preprocessing-v1',1,kind,'native_detector_pixels_before_viewer_rotation_and_flip')])
            write_loop(stream,kind,columns+extras,values)
        except Exception:temporary.unlink(missing_ok=True);raise
    os.replace(temporary,path)
