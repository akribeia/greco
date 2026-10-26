"""CTF parameter construction in the Greco raster and unit conventions."""

from __future__ import annotations



from .ctf_model import CTFParameters


def ctf_parameters_for(voltage_kV, cs_mm, amp_contrast, defocus_A,
                       astig_azimuth_deg, astig_magnitude_A, phase_rad=0.0):
    """Map the greco estimate onto the validated CTFParameters convention.

    The azimuth is the raw-raster azimuth; the caller is responsible for the
    post-rotation mapping (see ctf_model.astigmatism_angle_after_raster_rotation).
    """
    return CTFParameters(
        voltage_kV=voltage_kV, cs_mm=cs_mm, amplitude_contrast=amp_contrast,
        defocus_u_A=float(defocus_A - astig_magnitude_A / 2.0),
        defocus_v_A=float(defocus_A + astig_magnitude_A / 2.0),
        astigmatism_angle_deg=float(astig_azimuth_deg),
        phase_shift_rad=float(phase_rad),
    )
