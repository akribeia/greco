# Greco 1.1

Greco processes electron cryotomography (cryo-ET) movies into tomograms and
subtomogram averages. Its web interface manages GPU processing jobs through
Slurm: preprocessing, GRECO tilt-series alignment, reconstruction, TinyCARE
denoising, particle picking, subtomogram averaging and membrane segmentation.

GRECO names the mathematical algorithm, Geometric Registration of Electron
Cryotomography Observations. Greco is the software; its Greco optimizer
performs registration and geometry optimization. The lowercase name `greco`
identifies the executable and repository.

## Requirements

- Linux x86-64, glibc 2.28 or newer, and Python 3.11 with venv/pip support.
- NVIDIA GPU, Volta SM 7.0 or newer with compatible kernels. V100 and H100
  have been tested. R570 is the tested driver baseline; use a compatible
  driver for the installed GPU.
- Slurm client commands and access to a GPU partition.
- C/C++ compilers, FFTW development files (single and double precision),
  and libtiff development files. Compute nodes need the corresponding libraries.
- Internet access for dependency downloads. Allow about 20 GiB free during
  installation, plus separate storage for project data.

The installer downloads the pinned Python dependencies, including
PyTorch 2.10.0+cu128 and CuPy 13.6.0. A separate system CUDA Toolkit installation
is not required. The system NVIDIA driver is required.

## Installation

Choose an individual installation or one shared by several users.
Both download the latest published release. No Git clone is needed.

### Option 1: individual installation

Run these commands in your chosen software directory:

```bash
mkdir greco
cd greco
curl -fsSL https://github.com/akribeia/greco/releases/latest/download/install.py | python3.11 -
export PATH="$PWD/bin:$PATH"
greco start
```

Add the export line printed by the installer to `~/.bashrc` to keep Greco
in your own PATH for future terminals.

### Option 2: shared installation

The installation owner runs these commands in a common software directory.
Replace `/shared/software` with your cluster's shared location:

```bash
cd /shared/software
mkdir greco
cd greco
curl -fsSL https://github.com/akribeia/greco/releases/latest/download/install.py | python3.11 -
```

Add the following line to the cluster's shared shell profile so it applies
to all intended users:

```bash
export PATH=/shared/software/greco/bin:"$PATH"
```

Each user can then run `greco start` from the same installation. The common
folder must be readable and accessible to those users. Each user keeps their
own settings, projects list and socket in `~/.greco`. The installation owner
manages updates; other users only need read access.

Both options create `bin/`, `releases/` and `environments/` in the directory
where the installer is run. Use `--prefix /path/to/greco` to choose another
location. PATH setup is shown at completion; the installer does not edit shell
profiles automatically.

The installer runs directly from the release without saving `install.py`.
It verifies the application archive's SHA-256 before installation and downloads
the Python/GPU dependencies automatically. To select a particular release,
append `--version 1.1` after `python3.11 -`.

## Connect and use

```bash
greco start
greco status
greco restart
greco stop
```

`start` and `status` print an SSH command to run on your PC. Open the displayed
browser address, normally `http://localhost:1212`. The server uses a private
Unix socket on the cluster. `--local-port` changes the PC port; `--ssh-host`
selects the SSH destination.

One managed server runs per Unix user. Shared installations keep each user's
settings, recent projects, sockets and logs in their own `~/.greco`.
Projects and scientific data stay in the folders selected in the web interface.
Stopping the server leaves submitted Slurm jobs running.

> **Privacy:** Each user has a private Greco session, accessed through an
> owner-only Unix socket and an SSH tunnel. Greco runs with that user's file
> permissions. Project-file privacy depends on the permissions of the project
> folders; sharing the installation does not change those permissions.

## Slurm template

Open **Slurm template** in the web interface, paste this example and save it:

```bash
#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=greco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
```

Replace `gpu` with your cluster's GPU partition. Adjust memory and time for
your jobs. If your cluster requires an account, add
`#SBATCH --account=your_account`. Add a GPU type or constraint only if your
cluster requires it.

Greco fills in the job name, log paths, nodes, tasks and CPU count, and adds
the GPU request from the job settings when no GPU directive is present.
It also supplies the environment and processing command. You can paste an
existing cluster batch script instead; only its `#SBATCH` options are kept.
The saved template is shared across your projects, within your Unix account.

## Update

Check whether a new release is available:

```bash
greco update check
```

Download and install the latest release:

```bash
greco update
```

Activate the installed release using the command printed by Greco. For example,
if it installed version 1.1:

```bash
greco restart --version 1.1
```

Use the version shown by `greco update` and restart on the node where your
server is running. A plain `greco restart` keeps the current version.

For a shared installation, the installation owner updates it once. Each user
can then restart their own server with the new version. Updates are also
available from the version button in the web interface.

## Uninstall

Stop your server and let submitted jobs finish. From the parent directory of
your installation, remove the `greco` folder:

```bash
greco stop
rm -rf greco
```

Remove the Greco PATH line from your `~/.bashrc` or the shared shell profile.
For a shared installation, its owner removes the common folder after all
users have stopped their servers and their submitted jobs have finished.

Optionally remove your personal settings and recent-projects list:

```bash
rm -rf ~/.greco
```

Project data are stored in the project folders you selected. These commands
do not remove those projects.

## Models

The quantized pretrained MemBrain-seg checkpoint is included as
`assets/membranes/membrain-seg-quantized.pt`, with its hash recorded in
`runtime-locks/membranes.json`. The installer copies it into the installed
release. TinyCARE trains a model from each tomogram's even/odd half-volumes.
Model attribution and component notices are retained under `licenses/`.

## MATLAB prototype

[MATLAB prototype](https://github.com/akribeia/greco/tree/main/MATLAB) contains the early alignment prototype. It is included in
the repository for reference and excluded from installation releases.

## GRECO and bioRxiv screening

On Thursday, 29 October 2026, bioRxiv declined to post the manuscript **“GRECO: Geometric Registration of Electron Cryotomography Observations”** (submission **BIORXIV/2026/761553**), stating:

> “We cannot verify aspects of the manuscript and feel it would be better disseminated after peer review.”

The decision did not identify which aspects could not be verified.

The manuscript is accompanied by the Greco source code and the original MATLAB prototypes. All datasets used in the reported tests are publicly available. Researchers, including those responsible for screening at bioRxiv, can therefore examine the implementation and attempt to reproduce the results.

Preprint servers were created to accelerate scientific communication and allow researchers to share their work before journal peer review. Rejecting a manuscript because it should first undergo peer review appears to work against that purpose. It places a barrier before the open scientific discussion that preprints were meant to enable.

The concern goes beyond the explanation given for one rejection. Who assesses submissions? What expertise do they bring? How are the criteria applied? What interests, if any, influence which manuscripts are accepted or rejected? Who oversees these decisions, and how can authors challenge them? Without clear answers, researchers cannot meaningfully examine the process that controls access to the platform.

Independence requires more than operating outside traditional journals. The scientific community should question how much control over the circulation of research it delegates to platforms when the process behind their decisions remains unclear.

If you found **GRECO**, **Greco** or **greco** useful, please consider sharing your next preprint elsewhere.

## How to cite

Thanks to Zenodo for providing an open platform to share this work.

[Read the manuscript on Zenodo](https://doi.org/10.5281/zenodo.23871074).

If you use Greco, please cite:

```text
Carlos Fernández Rodríguez. (2026).
GRECO: Geometric Registration of Electron Cryotomography Observations.
Zenodo. https://doi.org/10.5281/zenodo.23871074
```

## License

Copyright (C) 2026 Carlos Fernández Rodríguez.

Original Greco contributions are licensed under AGPL-3.0-only, except where
otherwise noted. See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT) and
[third-party notices](licenses/THIRD_PARTY_NOTICES.md). Third-party code and
model assets retain their applicable attribution and terms.
