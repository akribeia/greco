"""Project persistence and raw-data discovery. No GPU dependencies."""
from __future__ import annotations
from greco.project_layout import join as storage_path

import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import threading
import uuid
from pathlib import Path

FIELDS = {
    "pixel_size_A": (0., None), "voltage_kv": (0., None),
    "dose_per_tilt": (0., None), "cs_mm": (0., None),
    "amplitude_contrast": (0., 1.), "tilt_axis_deg": (None, None),
}
REQUIRED = ("pixel_size_A", "voltage_kv", "dose_per_tilt", "cs_mm", "amplitude_contrast", "tilt_axis_deg")
MOVIE_SUFFIXES = {".tif", ".tiff", ".eer", ".mrc", ".mrcs"}
GAIN_SUFFIXES = {".tif", ".tiff", ".gain", ".mrc", ".mrcs"}


def input_format(path):
    suffix = Path(path).suffix.lower()
    return "TIFF" if suffix in {".tif", ".tiff"} else "MRC" if suffix in {".mrc", ".mrcs"} else "EER"


class InputError(ValueError):
    pass


def number(value):
    try:
        result = float(str(value).split()[0])
        return result if math.isfinite(result) else None
    except (ValueError, IndexError, TypeError):
        return None


def inspect_mdoc(path):
    """Tolerant metadata reader: expose missing values for the import form.

    The production parser requires voltage/pixel spacing before parsing. This
    reader deliberately allows those headers to be absent, but never invents
    a missing movie or tilt angle. It does not modify the original mdoc.
    """
    from ..mdoc import tilt_axis_from_line
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise InputError("MDOC is too large (16 MB limit).")
    headers, records, current = {}, [], None
    axis = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        found_axis = tilt_axis_from_line(line)
        if found_axis is not None:
            axis = found_axis
        if line.startswith("["):
            match = re.search(r"ZValue\s*=\s*(\d+)", line)
            if match:
                current = {"z": int(match.group(1))}
                records.append(current)
            continue
        if "=" not in line:
            continue
        key, value = (v.strip() for v in line.split("=", 1))
        value = value.split(";", 1)[0].strip()
        (headers if current is None else current)[key] = value
    if not records:
        raise InputError("The mdoc contains no ZValue sections.")
    errors, tilts, doses = [], [], []
    image_file = headers.get("ImageFile", "").replace("\\", "/").rsplit("/", 1)[-1]
    stack_mode = (not any(row.get("SubFramePath") for row in records)
                  and Path(image_file).suffix.lower() in {".mrc", ".mrcs"})
    for row in records:
        angle = number(row.get("TiltAngle"))
        movie = row.get("SubFramePath", "").replace("\\", "/").rsplit("/", 1)[-1]
        if stack_mode:
            movie = image_file
        if angle is None:
            errors.append(f"ZValue {row['z']}: missing a valid TiltAngle.")
        if not movie:
            errors.append(f"ZValue {row['z']}: missing SubFramePath.")
        dose = number(row.get("ExposureDose"))
        if dose is None:
            parts = row.get("FrameDosesAndNumber", "").split()
            if len(parts) % 2 == 0 and parts:
                values = [number(v) for v in parts]
                if all(v is not None and v >= 0 for v in values):
                    dose = sum(values[i]*values[i+1] for i in range(0, len(values), 2))
        if dose is not None and dose > 0:
            doses.append(dose)
        frames = number(row.get("NumSubFrames"))
        tilts.append(dict(z=row["z"], angle_deg=angle, movie_name=movie,
                          original_subframe_path=row.get("SubFramePath"),
                          stack_index=row["z"] if stack_mode else None,
                          n_frames=int(frames) if frames is not None else None,
                          dose_from_mdoc=dose))
    if len({t["z"] for t in tilts}) != len(tilts):
        errors.append("Duplicate ZValue records.")
    if not stack_mode and len({t["movie_name"] for t in tilts}) != len(tilts):
        errors.append("The mdoc references the same movie in multiple tilts.")
    def field(key):
        return number(headers.get(key, records[0].get(key)))
    dose = statistics.median(doses) if len(doses) == len(tilts) else None
    mdoc_axis = axis if axis is not None else field("TiltAxisAngle")
    metadata = dict(pixel_size_A=field("PixelSpacing"), voltage_kv=field("Voltage"),
                    cs_mm=field("Cs"), dose_per_tilt=dose,
                    amplitude_contrast=.10, tilt_axis_deg=mdoc_axis)
    warnings = []
    if doses and max(doses)-min(doses) > 1e-3:
        warnings.append("Dose varies between tilts in the mdoc. Enter the shared project dose.")
    fingerprint = [path.stat().st_size, path.stat().st_mtime_ns]
    return dict(name=re.sub(r"(?:\.mrc)?\.mdoc$", "", path.name, flags=re.I), mdoc_path=str(path),
                mdoc_fingerprint=fingerprint,
                mdoc_tilt_axis_deg=mdoc_axis,
                axis_convention='SerialEM TiltAxisAngle, unmodified',
                input_kind="tilt_stack" if stack_mode else "movies",
                metadata=metadata, tilts=tilts, errors=errors, warnings=warnings,
                metadata_sources={k: ("editable default" if k == "amplitude_contrast" else
                                       f"mdoc {mdoc_axis:g}° (unmodified)" if k == 'tilt_axis_deg' and mdoc_axis is not None else
                                       "mdoc" if v is not None else "pending") for k, v in metadata.items()})


def validate_metadata(raw):
    result = {}
    for key, (lo, hi) in FIELDS.items():
        value = raw.get(key)
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise InputError(f"A numeric value is required for {key}.") from None
        if not math.isfinite(value) or (lo is not None and value <= lo) or (hi is not None and value >= hi):
            raise InputError(f"Value out of range: {key}.")
        result[key] = value
    return result


def common_metadata(rows):
    """Suggest shared acquisition values; do not pick an arbitrary conflict."""
    result = {}
    for key in FIELDS:
        values = set()
        for row in rows:
            value = row.get('metadata', {}).get(key)
            values.add(value)
        values.discard(None)
        result[key] = next(iter(values)) if len(values) == 1 else None
    return result


def active_series_ids(rows, identifiers, bulk=False):
    """Keep disabled series inspectable, but exclude them from new work."""
    if any(i not in rows for i in identifiers):raise InputError('Unknown tilt series.')
    disabled=[i for i in identifiers if rows[i].get('enabled',True) is False]
    if disabled and not bulk:
        raise InputError(rows[disabled[0]]['name']+': tilt series is disabled. Enable it in Tilt series first.')
    return [i for i in identifiers if i not in disabled]


class ProjectService:

    def set_series_enabled_many(self, project_id, series_ids, enabled):
        if not isinstance(enabled,bool):raise InputError('Enabled must be a checkbox value.')
        if not isinstance(series_ids,list) or not series_ids or any(not isinstance(i,str) for i in series_ids) or len(series_ids)!=len(set(series_ids)):
            raise InputError('Select distinct tilt series.')
        with self.lock:
            root=self._root(project_id);data=self._read(root)
            rows={s['id']:s for s in data['series']}
            if any(i not in rows for i in series_ids):raise InputError('Unknown tilt series.')
            for identifier in series_ids:rows[identifier]['enabled']=enabled
            self._write(root,data)
            return self._view(project_id)
    def save_orientation(self, project_id, raw):
        import math
        required = {'rotation_deg', 'ctf_axis_deg', 'df_hand'}
        if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - {'flip_y','df_hand_tested','df_hand_convention'}:
            raise InputError('Invalid project orientation.')
        values = {}
        for key, value in raw.items():
            if key=='df_hand_convention':
                from ..ctf_local_refinement import DEFOCUS_CONVENTION
                if value!=DEFOCUS_CONVENTION:raise InputError('Repeat the defocus handedness test with the corrected convention.')
                values[key]=value
                continue
            if key in ('flip_y','df_hand_tested'):
                if not isinstance(value, bool):
                    raise InputError('Flip Y must be a checkbox value.')
                values[key] = value
                continue
            if isinstance(value, bool):
                raise InputError('Invalid orientation value.')
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise InputError('Invalid orientation value.') from None
            if not math.isfinite(value) or abs(value) > 360:
                raise InputError('Orientation angle out of range.')
            values[key] = value
        if values['df_hand'] not in (-1, 1):
            raise InputError('Handedness must be -1 or +1.')
        values['df_hand'] = int(values['df_hand'])
        if values.get('df_hand_tested',False):
            from ..ctf_local_refinement import current_hand_test
            if not current_hand_test(values):raise InputError('Repeat the defocus handedness test before marking its sign as tested.')
        with self.lock:
            root = self._root(project_id)
            data = self._read(root)
            data['orientation'] = values
            data['orientation_convention'] = 'original raster; flip_y first, then rotation_deg counter-clockwise about image centre; not baked into preprocessing stacks'
            self._write(root, data)
            return self._view(project_id)

    def save_slurm(self, project_id, template, submission_host=""):
        from .slurm import normalize_template
        if not isinstance(submission_host, str) or not re.fullmatch(r"[A-Za-z0-9_.@-]*", submission_host):
            raise InputError("Invalid submission host; use a hostname or user@hostname.")
        normalized = normalize_template(template)
        with self.lock:
            root = self._root(project_id)
            data = self._read(root)
            data['slurm'] = dict(normalized, submission_host=submission_host, source=template)
            if self.slurm_store is not None:
                self.slurm_store.save(data['slurm'])
            self._write(root, data)
            return self._view(project_id)

    def __init__(self, initial_directory=None, *, recent_store=None):
        self.initial_directory = Path(initial_directory or Path.home()).resolve(strict=True)
        self.recent_store = recent_store
        from ..user_state import SlurmSettings
        self.slurm_store = SlurmSettings(recent_store.root) if recent_store is not None else None
        self.projects, self.scans, self.previews = {}, {}, {}
        self.lock = threading.RLock()
        from .metadata_cache import MetadataCache
        self.metadata_cache = MetadataCache()

    def path(self, value, *, exists=True):
        if not isinstance(value, str) or not value.strip():
            raise InputError("Select a server path.")
        path = Path(value).expanduser().resolve()
        if exists and not path.exists():
            raise InputError("Path does not exist.")
        return path

    def browse_directory(self):
        return self.recent_store.directory() if self.recent_store is not None else self.initial_directory

    def browse(self, value=None, kind="folder"):
        if kind not in ("folder", "gain", "template"):
            raise InputError("Unknown browser type.")
        path = self.path(value or str(self.browse_directory()))
        if not path.is_dir():
            raise InputError("Select a folder.")
        folders, files = [], []
        for p in sorted(path.iterdir(), key=lambda p: p.name.casefold()):
            if p.name.startswith("."):
                continue
            try:
                if p.is_dir() and os.access(p, os.R_OK | os.X_OK):
                    folders.append(dict(name=p.name, path=str(p)))
                elif kind in ("gain", "template") and p.is_file() and p.suffix.lower() in (GAIN_SUFFIXES if kind == "gain" else {'.mrc','.map','.mrcs'}) and os.access(p, os.R_OK):
                    files.append(dict(name=p.name, path=str(p)))
            except OSError:
                continue
        parent = str(path.parent) if path.parent != path else None
        if self.recent_store is not None:
            self.recent_store.remember_directory(path)
        return dict(path=str(path), parent=parent,
                    folders=folders[:1000], files=files[:1000])

    def gain(self, value):
        if value is None or value == "":
            return None
        path = self.path(value)
        if not path.is_file() or path.suffix.lower() not in GAIN_SUFFIXES:
            raise InputError("Gain must be an existing TIFF, .gain or MRC file.")
        return str(path)

    def _root(self, project_id):
        try:
            return self.projects[project_id]
        except KeyError:
            raise InputError("Open a project first.") from None

    def _read(self, root):
        file = storage_path(root, "project.json")
        if file.exists():
            if not file.resolve().is_relative_to(root):
                raise InputError("The manifest must be inside the project folder.")
            from .metadata_cache import read_json
            data = read_json(self, file)
            if data.get("version") != 1 or not isinstance(data.get("series"), list):
                raise InputError("Unsupported project version.")
            if any(data.get(key) != 3 for key in ('layout_version', 'preprocessing_layout_version', 'reconstruction_layout_version')):
                raise InputError('Unsupported project layout.')
        else:
            data = dict(version=1, layout_version=3, preprocessing_layout_version=3, preprocessing_assemblies=[], reconstruction_layout_version=3, segmentation_layout_version=2, picking_layout_version=1, sta_layout_version=2, name=root.name, path=str(root), series=[])
        if self.slurm_store is not None:
            shared = self.slurm_store.load()
            if shared is None and data.get('slurm', {}).get('template'):
                shared = self.slurm_store.save(data['slurm'], only_if_missing=True)
            if shared is not None:
                data['slurm'] = shared
        return data

    def _write(self, root, data, *, publish=True):
        from .preprocessing_assemblies import capture as capture_preprocessing
        capture_preprocessing(data)
        from .pick_assemblies import capture
        capture(data)
        from .reconstruction_assemblies import capture as capture_recon
        capture_recon(data)
        from ..project_layout import initialize
        root.mkdir(parents=True,exist_ok=True)
        previous = self.metadata_cache.previous(root/'project.json')
        if publish:initialize(root,data,previous=previous)
        directory = root
        directory.mkdir(exist_ok=True)
        if not directory.resolve().is_relative_to(root):
            raise InputError("Project storage must be inside the project folder.")
        fd, temp = tempfile.mkstemp(prefix="project-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, storage_path(directory, "project.json"))
            self.metadata_cache.remember_json(directory/"project.json", data)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def _view(self, project_id):
        root = self._root(project_id)
        data = self._read(root)
        from .reconstruction_assemblies import ensure as ensure_reconstructions
        if ensure_reconstructions(data):self._write(root,data)
        from .segmentation import ensure_reconstruction_links
        if ensure_reconstruction_links(root,data):self._write(root,data)
        data["id"] = project_id
        data["path"] = str(root)
        from .slurm_arrays import ensure_indices
        if ensure_indices(data):self._write(root,data)
        for row in data['series']:row.setdefault('enabled',True)
        from .reconstruction_steps import ensure
        if any([ensure(row) for row in data['series']]):self._write(root,data)
        data.setdefault("metadata", common_metadata(data["series"]))
        data.setdefault('particle_assemblies',[])
        from .pending_extraction import refresh_members
        members_changed=False
        for record in data['particle_assemblies']:
            resolved=refresh_members(self,root,record)
            if resolved and not record.get('default_mask_path'):
                from .assemblies import default_mask_path
                mask=default_mask_path(record)
                if mask:record['default_mask_path']=mask
            members_changed=resolved or members_changed
        if members_changed:self._write(root,data)
        if "gain_path" not in data:
            gains = {row.get("gain_path") for row in data["series"]}
            data["gain_path"] = next(iter(gains)) if len(gains) == 1 else None
        from .preprocessing import preprocessing_state
        for row in data['series']:
            if row.get('preprocessing'):
                row['preprocessing']=preprocessing_state(row['preprocessing'],row.get('ctf_redo',{}))
        # Imported is not processed: CTF and downstream stages stay pending.
        complete = bool(data["series"]) and all(s["status"] == "imported" for s in data["series"])
        active=[s for s in data['series'] if s.get('enabled',True)]
        reconstructed=bool(active) and all(s.get('reconstruction',{}).get('state')=='completed' for s in active)
        from .reconstruction_assemblies import has_active_jobs
        reconstructed=reconstructed and not has_active_jobs(data)
        preprocessed=bool(active) and all(s.get('preprocessing',{}).get('state')=='completed' for s in active)
        from .picking_source import source_job
        has_volume=any(source_job(s).get('state') in ('completed','submitted','running') for s in data['series'])
        picked=bool(active) and all(s.get('picking',{}).get('state')=='completed' for s in active)
        has_picks=any(s.get('picking',{}).get('state')=='completed' and s.get('pick_count',0)>0 for s in data['series'])
        has_picks=has_picks or any(s.get('picking',{}).get('state')=='completed' and s.get('picking',{}).get('pick_count',0)>0 for a in data.get('pick_assemblies',[]) for s in a.get('series',{}).values())
        from .assemblies import PARTICLE_KINDS
        def sta_done(record):
            jobs=[record[k] for k in PARTICLE_KINDS if record.get(k)]
            return bool(jobs) and all(j.get('state')=='completed' for j in jobs)
        sta_complete=bool(data['particle_assemblies']) and all(sta_done(a) for a in data['particle_assemblies'])
        segmentations=data.get('segmentations',[])
        segmentation_complete=bool(segmentations) and all(
            bool(a.get('tomograms')) and all(bool(t.get('steps')) and all(s.get('state')=='completed' for s in t['steps']) for t in a['tomograms'])
            for a in segmentations)
        has_volume=has_volume or any(saved.get('reconstruction',{}).get('state') in ('completed','submitted','running') for record in data.get('reconstruction_assemblies',[]) for saved in record.get('series',{}).values())
        data["stages"] = [dict(id="tilts", label="Tilt series", enabled=True, complete=complete),
                          dict(id="preprocessing", label="Preprocessing", enabled=complete, complete=preprocessed),
                          dict(id="reconstruction", label="Reconstruction", enabled=complete, complete=reconstructed),
                          dict(id="picking", label="Picking", enabled=has_volume, complete=picked),
                          dict(id="sta", label="STA", enabled=bool(data['particle_assemblies']) or has_picks or any(s.get('picking',{}).get('state') in ('submitted','running') for s in data['series']) or any(s.get('picking',{}).get('state') in ('submitted','running') for a in data.get('pick_assemblies',[]) for s in a.get('series',{}).values()), complete=sta_complete),
                          dict(id="segmentation", label="Segmentation", enabled=has_volume or bool(data.get('segmentations')), complete=segmentation_complete)]
        return data

    def open(self, value, create=False):
        with self.lock:
            root = self.path(value, exists=not create)
            if create:
                root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise InputError("The project must be a folder.")
            pid = hashlib.sha256(str(root).encode()).hexdigest()[:16]
            self.projects[pid] = root
            # Recover disk-only jobs once; all subsequent views use the manifest.
            data = self._read(root)
            from .assemblies import restore_job_history, default_mask_path
            changed = False
            for record in data.get('particle_assemblies', []):
                before = len(record.get('job_history', []))
                restore_job_history(root, record)
                changed = changed or len(record.get('job_history', [])) != before
                if not record.get('default_mask_path'):
                    mask = default_mask_path(record)
                    if mask:
                        record['default_mask_path'] = mask
                        changed = True
            if changed:
                self._write(root, data)
            view = self._view(pid)
            if self.recent_store is not None:
                self.recent_store.remember(root, view['name'])
            return view

    def get(self, project_id):
        with self.lock:
            return self._view(project_id)

    def scan(self, project_id, mdoc_dir, movie_dir):
        with self.lock:
            self._root(project_id)
            roots = [self.path(mdoc_dir), self.path(movie_dir)]
            results = []
            for root, suffixes in zip(roots, ({".mdoc"}, MOVIE_SUFFIXES)):
                if not root.is_dir():
                    raise InputError("Source paths must be folders.")
                files, visited = [], 0
                for parent, directories, names in os.walk(root, followlinks=False):
                    directories[:] = sorted(d for d in directories if not d.startswith("."))
                    for name in names:
                        visited += 1
                        if visited > 100000 or len(files) >= 20000:
                            raise InputError("Too many files: select a more specific subfolder.")
                        p = storage_path(Path(parent), name)
                        if p.suffix.lower() in suffixes:
                            files.append(str(p.resolve()))
                results.append(sorted(set(files)))
            sid = uuid.uuid4().hex
            self.scans[sid] = dict(project_id=project_id, mdocs=results[0], movies=results[1])
            return dict(scan_id=sid, mdocs=results[0], movies=results[1])

    def preview(self, project_id, scan_id, mdocs, movies, gain_path=None, gain_flip_y=False):
        with self.lock:
            scan = self.scans.get(scan_id)
            if not scan or scan["project_id"] != project_id:
                raise InputError("Scan the folders again.")
            if not mdocs or not movies or len(mdocs) != len(set(mdocs)):
                raise InputError("Select mdocs and TIFF/EER/MRC files without duplicates.")
            if not set(mdocs).issubset(scan["mdocs"]) or not set(movies).issubset(scan["movies"]):
                raise InputError("The selection does not belong to this scan.")
            index = {}
            for movie in movies:
                index.setdefault(Path(movie).name.casefold(), []).append(movie)
            rows = []
            gain_path = self.gain(gain_path)
            if type(gain_flip_y) is not bool:raise InputError("Gain Flip Y must be true or false.")
            for mdoc in mdocs:
                row = inspect_mdoc(self.path(mdoc))
                row["id"] = hashlib.sha256(mdoc.encode()).hexdigest()[:16]
                found, missing, ambiguous = 0, [], []
                for tilt in row["tilts"]:
                    matches = index.get(tilt["movie_name"].casefold(), [])
                    tilt["movie_path"] = matches[0] if len(matches) == 1 else None
                    tilt["input_format"] = input_format(matches[0]) if len(matches) == 1 else None
                    if len(matches) == 1:
                        found += 1
                    elif matches:
                        ambiguous.append(tilt["movie_name"])
                    else:
                        missing.append(tilt["movie_name"])
                row.update(found=found, expected=len(row["tilts"]), missing=missing, ambiguous=ambiguous)
                row["input_format"] = " / ".join(sorted({t["input_format"] for t in row["tilts"] if t["input_format"]}))
                row["gain_path"] = gain_path
                row["gain_flip_y"] = gain_flip_y
                rows.append(row)
            token = uuid.uuid4().hex
            self.previews[token] = dict(project_id=project_id, rows=rows)
            return dict(preview_id=token, series=rows)

    def import_series(self, project_id, preview_id, overrides, gains=None, gain_flip_y=False):
        with self.lock:
            preview = self.previews.get(preview_id)
            if not preview or preview["project_id"] != project_id:
                raise InputError("Prepare the import again.")
            if set(overrides) != {r["id"] for r in preview["rows"]}:
                raise InputError("Complete the metadata for all selected tilt series.")
            if gains is not None and set(gains) != set(overrides):
                raise InputError("Specify a gain reference or no gain for each series.")
            shared_metadata = validate_metadata(next(iter(overrides.values())))
            if any(validate_metadata(values) != shared_metadata for values in overrides.values()):
                raise InputError("Metadata must be shared by the entire project.")
            shared_gains = {self.gain(gains[row['id']] if gains is not None else row.get('gain_path')) for row in preview['rows']}
            if len(shared_gains) != 1:
                raise InputError("Select a shared project gain reference or no gain.")
            shared_gain = next(iter(shared_gains))
            fresh = []
            for source in preview["rows"]:
                row = json.loads(json.dumps(source))
                if row["errors"] or row["missing"] or row["ambiguous"]:
                    raise InputError(f"{row['name']}: missing or ambiguous data files, or mdoc errors.")
                mdoc = self.path(row["mdoc_path"])
                if [mdoc.stat().st_size, mdoc.stat().st_mtime_ns] != row['mdoc_fingerprint']:
                    raise InputError(f"{row['name']}: the mdoc changed; review the import again.")
                for tilt in row["tilts"]:
                    path = self.path(tilt["movie_path"])
                    if not path.is_file():
                        raise InputError(f"Data file unavailable: {path.name}")
                row["original_metadata"] = dict(row["metadata"])
                row["gain_path"] = self.gain(gains[row["id"]] if gains is not None else row.get("gain_path"))
                row["metadata"] = validate_metadata(overrides[row["id"]])
                for key, value in row["metadata"].items():
                    if value != row["original_metadata"][key]:
                        row["metadata_sources"][key] = "user override"
                angles = [t["angle_deg"] for t in row["tilts"]]
                row.update(angle_min=min(angles), angle_max=max(angles), status="imported", ctf=None)
                fresh.append(row)
            root = self._root(project_id)
            data = self._read(root)
            existing = {r["id"]: r for r in data["series"]}
            for row in fresh:
                row['enabled']=existing.get(row['id'],{}).get('enabled',True)
                if existing.get(row['id'],{}).get('storage_name'):row['storage_name']=existing[row['id']]['storage_name']
                if existing.get(row['id'],{}).get('slurm_array_index') is not None:
                    row['slurm_array_index']=existing[row['id']]['slurm_array_index']
            existing.update({r["id"]: r for r in fresh})
            data["series"] = list(existing.values())
            self._apply_shared(data, shared_metadata, shared_gain, gain_flip_y)
            self._write(root, data)
            del self.previews[preview_id]
            return self._view(project_id)

    def import_project(self, project_id, preview_id, metadata, gain_path=None, gain_flip_y=False):
        with self.lock:
            preview = self.previews.get(preview_id)
            if not preview or preview['project_id'] != project_id:
                raise InputError('Prepare the import again.')
            ids = [row['id'] for row in preview['rows']]
            return self.import_series(project_id, preview_id,
                                      {key:metadata for key in ids},
                                      {key:gain_path for key in ids}, gain_flip_y=gain_flip_y)

    def edit(self, project_id, series_id, metadata, **options):
        with self.lock:
            root = self._root(project_id)
            data = self._read(root)
            if not any(row['id'] == series_id for row in data['series']):
                raise InputError("Unknown tilt series.")
            return self.edit_project(project_id, metadata, **options)

    def _apply_shared(self, data, metadata, gain, gain_flip_y=False):
        if type(gain_flip_y) is not bool:raise InputError("Gain Flip Y must be true or false.")
        data['metadata'] = dict(metadata)
        data['gain_path'] = gain
        data['gain_flip_y'] = gain_flip_y
        for row in data['series']:
            for key, value in metadata.items():
                if value != row['metadata'].get(key):
                    row['metadata_sources'][key] = 'shared project value'
            row['metadata'] = dict(metadata)
            row['gain_path'] = gain
            row['gain_flip_y'] = gain_flip_y

    def edit_project(self, project_id, metadata, **options):
        with self.lock:
            root = self._root(project_id)
            data = self._read(root)
            validated = validate_metadata(metadata)
            gain = self.gain(options['gain_path']) if 'gain_path' in options else data.get('gain_path')
            self._apply_shared(data, validated, gain, options.get('gain_flip_y',data.get('gain_flip_y',False)))
            self._write(root, data)
            return self._view(project_id)
