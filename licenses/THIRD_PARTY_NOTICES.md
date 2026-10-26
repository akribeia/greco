# Third-party notices

Greco includes the following reference implementations, adapted components and
third-party sources. Copyright and license terms apply to those components;
this index does not assign a single license to all Greco modules.

## Greco licensing scope

Copyright (C) 2026 Carlos Fernández Rodríguez applies to original Greco
software contributions for which he is the copyright holder. Original Greco code is
licensed under AGPL-3.0-only except where otherwise noted. See the root
[LICENSE](../LICENSE) and [COPYRIGHT](../COPYRIGHT).

Existing third-party copyright notices, license conditions and disclaimers
remain applicable. Identified RELION-derived ports retain their GPL version 2
or later notices. For distribution in combination with Greco's AGPLv3 modules,
the permitted GPLv3 terms and the GPLv3/AGPLv3 combination provisions apply.
The GPL terms continue to govern the GPL-covered portions; the AGPLv3 network
source requirements extend to the combined work as specified in section 13.
The original [RELION notices](relion/NOTICE.txt) and [GPLv2 text](relion/COPYING.txt)
are retained, and the [GPLv3 text](gpl3/COPYING.txt) is also supplied.
This is not a replacement of the original RELION license or authorship.

BSD and other third-party components retain their own terms. Model weights
are separate assets; their licensing is not established by Greco's AGPL notice.

| Component | Integration in Greco | Retained notices |
| --- | --- | --- |
| [RELION](https://github.com/3dem/relion) | Python ports of identified CTF, projection/backprojection, tomography and postprocessing routines in `greco/_numerics` | [Source notices](relion/NOTICE.txt), [GPL text](relion/COPYING.txt) |
| Minimal numerical core | Greco C++ implementation using FFTW/glibc, retaining reference FFT normalization and FSC/Guinier accumulation conventions | [Reference notices](relion/NOTICE.txt), [GPL text](relion/COPYING.txt) |
| MotionCor / MotionCor3 | Full-frame motion correction and detector-defect correction in `motion.py` and `detector_defects.py`; EER decoder adapted from the MotionCor component integrated into AreTomo3 | [MotionCor3 BSD notice](motioncor3/LICENSE.txt), [AreTomo3 BSD notice for the integrated EER reference](aretomo3/LICENSE.txt) |
| AreTomo3 FindCtf | CTF spectrum tiling (`CTsTiles`), 2D fitting schedule (`CFindCtf2D::Do2D`) and weak-fit refinement across tilts (`CFindCtfMain::mRefineTilts`) in `ctf_estimator.py` | [BSD notice](aretomo3/LICENSE.txt) |
| SciPy / scikit-image | GPU registration adapters in `align2d_gpu.py`, including FFT convolution and phase correlation with upsampled DFT; selected ndimage resampling routines in `scipy_resample_gpu.py` | [SciPy BSD notice](scipy/LICENSE.txt), [ndimage source notice](scipy/NDIMAGE_NOTICE.txt), [scikit-image notices](scikit-image/LICENSE.txt) |
| PyTom / pytom-match-pick | Historical source overlap in picker Fourier weights, angular sampling and masked phase randomization; numerical helpers replaced on 2026-10-23 while preserving established outputs | See the provenance scope below; no change of upstream licensing is asserted |
| voltools | Selected rotation/interpolation helpers in `picker/geometry.py`; compatible interpolation conventions in `picker/weights.py` | [MIT notice](voltools/LICENSE.txt) |
| CUDA Cubic B-Spline Interpolation | Selected prefilter and cubic interpolation helpers in `picker/geometry.py`; Danny Ruijters implementation based on Philippe Thévenaz’s work | [BSD license](cubic-interpolation/LICENSE.txt), [prefilter source notice](cubic-interpolation/NOTICE.txt) |
| libLBFGS | Included optimizer source and Greco's batched native integration | [License](liblbfgs/LICENSE.txt); original notices also remain in `greco/sta/vendor/liblbfgs` |
| [cryoCARE](https://github.com/juglab/cryoCARE_pip) | TinyCARE training protocol and sampling; Greco implementation uses PyTorch | [BSD notice](cryocare/LICENSE.txt) |
| [CSBDeep](https://github.com/CSBDeep/CSBDeep) | TinyCARE network topology and block-aligned tiling | [BSD notice](csbdeep/LICENSE.txt) |
| [MemBrain-seg](https://github.com/teamtomo/membrain-seg) | Standard pretrained membrane weights, quantized for Greco; inference uses MONAI DynUNet | [Project BSD notice](membrain-seg/LICENSE.txt); checkpoint terms are separate |

## Scope of source attribution

Per-file comments identify the adapted MotionCor3 full-frame alignment and
detector correction routines, the MotionCor EER decoder integrated in
AreTomo3, and the AreTomo3 FindCtf routines. The Greco contributions remain
separately identified; the retained BSD conditions and disclaimers apply to
the corresponding upstream portions.

TinyCARE is implemented in PyTorch within Greco. Its source comments identify
cryoCARE's even/odd sampling and training protocol and CSBDeep/CARE's reference
network topology and block-aligned tiling. These references do not identify
the whole TinyCARE implementation as a verbatim upstream port. The retained
BSD notices apply where upstream material is adapted.

Greco Picker's numerical helpers in `picker/spectral.py` and the HEALPix
orientation builder in `picker/geometry.py` were reimplemented on 2026-10-23.
The previous code had executable source overlap with pytom-match-pick's
Fourier grids, CTF and wedge helpers, angular sampling, and masked phase
randomization. The replacement preserves the same mathematical conventions,
random generator, projection sequence and tilt summation order. A frozen
byte-level contract was used to check numerical equivalence during development.

This was not a clean-room process: earlier implementations were inspected.
The rewrite does not establish independent provenance or extinguish prior
attribution and applicable terms. The reference versions inspected were
pytom-match-pick 0.14.0 (`86ee8edd17258718d26f4240c60b7ce6cf80c523`)
and original PyTom (`5d07de0d5de0b3e49e245c38155c97498ad2f408`).
The former declares GPLv2 in its metadata; the latter has a GPLv3-or-later
README declaration and a conflicting GPLv2 `LICENSE.txt`. Clarification
with the authors remains separate from numerical compatibility. No upstream
license has been changed or inferred from the replacement.

Selected rotation and interpolation helpers retain the notices listed above.
The voltools MIT notice is retained from the installed 0.6.0 distribution.
The cubic interpolation license and prefilter notice come from
[DannyRuijters/CubicInterpolationCUDA](https://github.com/DannyRuijters/CubicInterpolationCUDA)
(`license.txt` and `code/internal/cubicPrefilter_kernel.cu`). The SciPy ndimage
notice comes from `scipy/ndimage/src/ni_interpolation.c` in SciPy v1.17.1.

## External dependencies

The application wheel does not bundle PyTorch, CUDA, FFTW, libtiff or the
scientific Python environment. These are installed separately and retain their
own licenses and notices. Required dependency versions are declared in
`pyproject.toml` and recorded in `runtime-locks/`.

Installed dependencies are kept in the installation's `environments/` tree;
source deployments may refer to them through `runtime/`. Dependency notices
remain with the installed packages, including the NVIDIA CUDA packages.
Neither runtime tree is included in the application source archive or wheel.
A distribution that includes these dependencies must also retain their
component notices.

The MemBrain-seg attribution identifies the source of the standard pretrained
weights. These weights are stored in a quantized format; checkpoint decoding
and inference are integrated using MONAI DynUNet. The MemBrain-seg
application is not a runtime dependency. The retained project BSD notice does
not by itself establish the license of the model weights.

Model checkpoints are separate assets. Their distribution terms need to be
checked with the checkpoint source. The exact upstream release or download
identifier is not recorded; local quantized source and packaged checkpoint
identities are recorded in `runtime-locks/membranes.json`.
