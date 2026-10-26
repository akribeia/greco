# Third-party references and adapted components
# TinyCARE is implemented in PyTorch within Greco. cryoCARE supplies the even/odd denoising protocol; CSBDeep/CARE supplies the reference topology and tiling conventions. These references do not assign upstream authorship to the entire Greco implementation.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/cryocare/LICENSE.txt
# licenses/csbdeep/LICENSE.txt
#
# Copyright (c) 2020, juglab
# Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
# Full license conditions and disclaimers are retained at the paths above.
#

"""TinyCARE: residual 3D Noise2Noise denoising in Greco's shared GPU runtime."""
