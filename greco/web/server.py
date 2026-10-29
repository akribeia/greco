"""Foreground TCP server; managed Unix-socket instances use greco.launcher."""
from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .projects import InputError, ProjectService

STATIC = Path(__file__).with_name("static")


def make_handler(service, session_token):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json", *, cache_control="no-store", etag=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, allow_nan=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            compressed = False
            if len(body) >= 4096 and content_type.startswith(("application/json", "text/", "application/javascript")):
                encodings = [item.strip() for item in self.headers.get("Accept-Encoding", "").split(",")]
                if "gzip" in encodings:
                    body = gzip.compress(body, compresslevel=1)
                    compressed = True
            self.send_response(status)
            self.send_header("Vary", "Accept-Encoding")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            if etag is not None:self.send_header("ETag",etag)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            if content_type.startswith("text/html"):
                self.send_header("Set-Cookie", f"greco_session={session_token}; HttpOnly; SameSite=Strict; Path=/")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self,path):
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(path.stat().st_size))
            self.send_header('Content-Disposition','attachment; filename="'+path.name+'"')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            with path.open('rb') as stream:
                while chunk:=stream.read(1024*1024):self.wfile.write(chunk)

        def authenticated(self):
            cookie = self.headers.get("Cookie", "")
            tokens = dict(part.strip().split("=", 1) for part in cookie.split(";") if "=" in part)
            if not secrets.compare_digest(tokens.get("greco_session", ""), session_token):
                raise InputError("Open the interface to start a session.")
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise InputError("Origin is not allowed.")
            content_type=self.headers.get("Content-Type", "").split(";")[0]
            upload=urlsplit(self.path).path=='/api/picker-template-upload'
            if self.command == "POST" and content_type != ('application/octet-stream' if upload else 'application/json'):
                raise InputError('Use application/octet-stream.' if upload else 'Use application/json.')

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path.startswith("/api/"):
                    self.authenticated()
                    if parsed.path == '/api/update-status':
                        from ..updates import status
                        self.send(200,status(check=query.get('check',['0'])[0]=='1'))
                    elif parsed.path == "/api/browse":
                        self.send(200, service.browse(query.get("path", [None])[0], query.get("kind", ["folder"])[0]))
                    elif parsed.path == '/api/preprocessing-capabilities':
                        from ..ctf_coordinates import PROTOCOL
                        self.send(200,dict(ctf_axis_protocol=PROTOCOL,explicit_ctf_handedness=True))
                    elif parsed.path == '/api/particle-observation-report':
                        from .particle_observation_refinement import report_file
                        path=report_file(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0],query.get('kind',[''])[0],query.get('job',[''])[0])
                        self.send_file(path)
                    elif parsed.path == "/api/project":
                        self.send(200, service.get(query.get("id", [""])[0]))
                    elif parsed.path == '/api/project-status':
                        from .preprocessing import project_status
                        self.send(200,project_status(service,query.get('project_id',[''])[0]))
                    elif parsed.path == '/api/particle-postprocess-file':
                        from .particle_postprocess import artifact
                        path=artifact(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0],query.get('job',[''])[0],query.get('file',[''])[0])
                        self.send_file(path)
                    elif parsed.path == '/api/segmentation-status':
                        from .segmentation import status
                        self.send(200,status(service,query.get('project_id',[''])[0]))
                    elif parsed.path == '/api/segmentation-plane':
                        from .segmentation import plane
                        self.send(200,plane(service,query.get('project_id',[''])[0],query.get('segmentation_id',[''])[0],query.get('series_id',[''])[0],int(query.get('number',['0'])[0]),int(query.get('index',['0'])[0]),query.get('axis',['z'])[0],segment=query.get('segment',['0'])[0]=='1',autocontrast=query.get('autocontrast',['1'])[0]=='1',contrast=float(query.get('contrast',['1'])[0]),brightness=float(query.get('brightness',['0'])[0])),'image/png')
                    elif parsed.path == '/api/particle-jobs-status':
                        from .particle_jobs import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0]))
                    elif parsed.path == '/api/particle-view-history':
                        from .particle_view_history import history
                        self.send(200,history(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0],query.get('kind',[''])[0],query.get('job',[''])[0]))
                    elif parsed.path == '/api/particle-job-preset':
                        from .particle_job_preset import preset
                        self.send(200,preset(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0],query.get('job',[''])[0]))
                    elif parsed.path == '/api/particle-class-plane':
                        from .particle_classification import plane
                        self.send(200,plane(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0],int(query.get('index',['0'])[0]),int(query.get('class_id',['1'])[0]),axis=query.get('axis',['z'])[0],job_name=query.get('job',[None])[0]),'image/png')
                    elif parsed.path == '/api/particle-average-plane':
                        from .particle_jobs import plane
                        self.send(200,plane(service,query.get('project_id',[''])[0],query.get('assembly_id',[''])[0],int(query.get('index',['0'])[0]),axis=query.get('axis',['z'])[0],job_name=query.get('job',[None])[0],kind=query.get('kind',['refinement3d'])[0]),'image/png')
                    elif parsed.path == "/api/settings":
                        self.send(200, dict(browse_directory=str(service.browse_directory()), recent_projects=service.recent_store.list() if service.recent_store is not None else []))
                    elif parsed.path == '/api/preprocessing-status':
                        from .preprocessing import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0]))
                    elif parsed.path == '/api/picker-status':
                        from .picking import status
                        self.send(200,status(service,query.get('project_id',[''])[0],assembly_id=query.get('assembly_id',[None])[0]))
                    elif parsed.path == '/api/picker-asset-preview':
                        from .picking import asset_png
                        self.send(200,asset_png(service,query.get('project_id',[''])[0],query.get('asset',['template'])[0],query.get('assembly_id',[None])[0],index=int(query['index'][0]) if 'index' in query else None,axis=query.get('axis',['z'])[0],full=query.get('full',['0'])[0]=='1'),'image/png')
                    elif parsed.path == '/api/picker-examples':
                        from .picking import examples_png
                        self.send(200,examples_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query.get('assembly_id',[None])[0]),'image/png')
                    elif parsed.path == '/api/picker-viewer-info':
                        from .picking import viewer_info
                        self.send(200,viewer_info(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query.get('assembly_id',[None])[0]))
                    elif parsed.path == '/api/picker-plane':
                        from .picking import viewer_png
                        self.send(200,viewer_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],int(query.get('index',['0'])[0]),query.get('axis',['z'])[0],background=query.get('background',['.2'])[0],autocontrast=query.get('autocontrast',['0'])[0]=='1',contrast=query.get('contrast',['1'])[0],method=query.get('method',['original'])[0],brightness=query.get('brightness',['0'])[0],assembly_id=query.get('assembly_id',[None])[0]),'image/png')
                    elif parsed.path == '/api/handedness-status':
                        from .handedness import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0]))
                    elif parsed.path == '/api/ctf-redo-status':
                        from .ctf_redo import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0]))
                    elif parsed.path == '/api/reconstruction-landmark-densities':
                        from .landmark_density import densities
                        self.send(200,densities(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query.get('assembly_id',[None])[0]))
                    elif parsed.path == '/api/reconstruction-landmarks':
                        from .landmark_cloud import cloud
                        self.send(200,cloud(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query.get('assembly_id',[None])[0]))
                    elif parsed.path == '/api/reconstruction-tilt-plane':
                        from .reconstruction import refined_tilt_png
                        self.send(200,refined_tilt_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],int(query.get('index',['0'])[0]),query.get('mode',[None])[0]),'image/png')
                    elif parsed.path == '/api/reconstruction-status':
                        from .reconstruction import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0]))
                    elif parsed.path == '/api/reconstruction-capabilities':
                        from .reconstruction import capabilities
                        self.send(200,capabilities())
                    elif parsed.path == '/api/refinement-status':
                        from .refinement import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0]))
                    elif parsed.path == '/api/geometry-refinement-status':
                        if query.get('step_id'):
                            from .reconstruction_steps import status
                            self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query['step_id'][0]))
                        else:
                            from .refinement import status
                            self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],kind='geometry_refinement'))
                    elif parsed.path == '/api/angle-refinement-status':
                        from .refinement import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],kind='angle_refinement'))
                    elif parsed.path == '/api/volume-plane':
                        if query.get('step_id'):
                            from .reconstruction_steps import volume_png
                            result=volume_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query['step_id'][0],int(query.get('index',['0'])[0]),query.get('axis',['z'])[0],query.get('method',['original'])[0],autocontrast=query.get('autocontrast',['0'])[0]=='1',contrast=query.get('contrast',['1'])[0],brightness=query.get('brightness',['0'])[0])
                        else:
                            from .reconstruction import volume_png
                            result=volume_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],int(query.get('index',['0'])[0]),query.get('method',['original'])[0],query.get('axis',['z'])[0],autocontrast=query.get('autocontrast',['0'])[0]=='1',contrast=query.get('contrast',['1'])[0],brightness=query.get('brightness',['0'])[0])
                        self.send(200,result,'image/png')
                    elif parsed.path == '/api/curation-status':
                        from .manual_curation import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query.get('source',['reconstruction'])[0]))
                    elif parsed.path == '/api/manual-refinement-status':
                        from .manual_refinement import status
                        self.send(200,status(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],query.get('step_id',[''])[0]))
                    elif parsed.path == '/api/curation-plane':
                        from .manual_curation import plane_png
                        self.send(200,plane_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],int(query.get('index',['0'])[0]),query.get('source',['reconstruction'])[0],discarded=query.get('discarded',['0'])[0]=='1'),'image/png')
                    elif parsed.path == '/api/ctf-plane':
                        from .ctf_view import ctf_png
                        self.send(200,ctf_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],int(query.get('index',['0'])[0]),query.get('mode',['rings'])[0],rotation_deg=query.get('rotation_deg',['0'])[0],flip_y=query.get('flip_y',['0'])[0]=='1',df_hand=query.get('df_hand',[None])[0],axis_deg=query.get('axis_deg',[None])[0],preview_geometry=query.get('preview_geometry',['0'])[0]=='1'),'image/png')
                    elif parsed.path == '/api/tilt-plane':
                        from .preprocessing import plane_png
                        self.send(200,plane_png(service,query.get('project_id',[''])[0],query.get('series_id',[''])[0],int(query.get('index',['0'])[0]),query.get('rotation_deg',[None])[0],flip_y=query.get('flip_y',['0'])[0]=='1',landscape=query.get('landscape',['0'])[0]=='1',df_hand=query.get('df_hand',[None])[0],axis_deg=query.get('axis_deg',[None])[0],preview_geometry=query.get('preview_geometry',['0'])[0]=='1'),'image/png')
                    else:
                        self.send(404, dict(error="Unknown route."))
                    return
                from .static_assets import ROUTES as assets
                if parsed.path not in assets:
                    self.send(404, dict(error="Not found."))
                    return
                file = STATIC / assets[parsed.path]
                mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                etag=None
                if parsed.path!='/':
                    stamp=file.stat();etag=f'"{stamp.st_mtime_ns:x}-{stamp.st_size:x}"'
                    if self.headers.get('If-None-Match')==etag:
                        self.send_response(304)
                        self.send_header('ETag',etag)
                        self.send_header('Cache-Control','private, max-age=0, must-revalidate')
                        self.end_headers()
                        return
                payload=file.read_bytes()
                if parsed.path=='/':
                    from .. import __version__
                    payload=payload.replace(b'__GRECO_VERSION__',__version__.encode())
                self.send(200, payload, mime + "; charset=utf-8",etag=etag,
                          cache_control='private, max-age=0, must-revalidate' if etag else 'no-store')
            except (InputError, OSError, ValueError) as error:
                self.send(400, dict(error=str(error)))

        def do_POST(self):
            try:
                self.authenticated()
                length = int(self.headers.get("Content-Length", 0))
                if urlsplit(self.path).path=='/api/picker-template-upload':
                    from .picking import upload_template,MAX_TEMPLATE_BYTES
                    if not 0<length<=MAX_TEMPLATE_BYTES:raise InputError('Template upload must be between 1 byte and 128 MB.')
                    query=parse_qs(urlsplit(self.path).query)
                    result=upload_template(service,query.get('project_id',[''])[0],query.get('filename',['template.mrc'])[0],self.rfile.read(length))
                    self.send(200,result)
                    return
                if not 0 < length <= 8 * 1024 * 1024:
                    raise InputError("Empty request or request too large.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise InputError("Expected a JSON object.")
                route = urlsplit(self.path).path
                if route == '/api/update-install':
                    from ..updates import start_install
                    result=start_install(data.get('version'))
                elif route == '/api/update-restart':
                    from ..updates import schedule_restart
                    result=schedule_restart(data.get('version'))
                elif route == "/api/open":
                    result = service.open(data.get("path"), create=data.get("create") is True)
                elif route == "/api/picker-template-select":
                    from .picking import select_template
                    result = select_template(service,data['project_id'],data['path'],assembly_id=data.get('assembly_id'))
                elif route == '/api/reconstruction-assemblies':
                    from .reconstruction_assemblies import create
                    result=create(service,data['project_id'],data.get('name'),data.get('mode'))
                elif route == '/api/reconstruction-assembly-delete':
                    from .reconstruction_assemblies import delete
                    result=delete(service,data['project_id'],data['assembly_id'],data.get('confirmed',False))
                elif route == '/api/select-reconstruction-assembly':
                    from .reconstruction_assemblies import select
                    result=select(service,data['project_id'],data['assembly_id'])
                elif route == '/api/pick-assemblies':
                    from .pick_assemblies import create
                    result=create(service,data['project_id'],data.get('name'),data.get('reconstruction_id'))
                elif route == '/api/select-pick-assembly':
                    from .pick_assemblies import select
                    result=select(service,data['project_id'],data['assembly_id'])
                elif route == '/api/pick-assembly-delete':
                    from .pick_assemblies import delete
                    result=delete(service,data['project_id'],data.get('assembly_id'),data.get('confirmed',False))
                elif route == '/api/picker-mask-select':
                    from .picking import select_template
                    result=select_template(service,data['project_id'],data['path'],asset='mask',assembly_id=data.get('assembly_id'))
                elif route == '/api/segmentation-create':
                    from .segmentation import create
                    result=create(service,data['project_id'],data.get('segmentation',{}))
                elif route == '/api/segmentation-delete':
                    from .segmentation_deletion import delete
                    result=delete(service,data['project_id'],data.get('segmentation_id'),data.get('confirmed',False))
                elif route == '/api/segmentation-step':
                    from .segmentation import add_step
                    result=add_step(service,data['project_id'],data['segmentation_id'],data['series_id'],data['kind'],data.get('options',{}))
                elif route == '/api/segmentation-layer-name':
                    from .segmentation import rename_layer
                    result=rename_layer(service,data['project_id'],data['segmentation_id'],data['series_id'],data['number'],data.get('name'))
                elif route == '/api/segmentation-clear':
                    from .segmentation import clear_step
                    result=clear_step(service,data['project_id'],data['segmentation_id'],data['series_id'],data['number'],data.get('confirmed',False))
                elif route == '/api/particle-assemblies':
                    from .assemblies import create
                    result=create(service,data['project_id'],data.get('assembly',{}))
                elif route == '/api/submit-particle-extraction':
                    from .particle_jobs import submit
                    result=submit(service,data['project_id'],data.get('assembly_id'),kind='extraction',options=data.get('options'))
                elif route == '/api/particle-assembly-delete':
                    from .assemblies import delete
                    result=delete(service,data['project_id'],data.get('assembly_id'),data.get('confirmed',False))
                elif route == '/api/preprocessing-delete':
                    from .preprocessing_deletion import delete
                    result=delete(service,data['project_id'],data['series_id'],data.get('work'),data.get('confirmed',False))
                elif route == '/api/processing-job-delete':
                    from .preprocessing_deletion import delete
                    result=delete(service,data['project_id'],data['series_id'],data.get('work'),data.get('confirmed',False),kind=data.get('kind'))
                elif route == '/api/curation-test':
                    from .manual_curation import run_test
                    result=run_test(service,data['project_id'],data['series_id'],data.get('source','reconstruction'))
                elif route == '/api/submit-manual-refinement':
                    from .manual_refinement import submit
                    result=submit(service,data['project_id'],data['series_id'],data['step_id'],data.get('edits'),data.get('source_identity'),options=data.get('options'))
                elif route == '/api/submit-manual-export':
                    from .manual_refinement import submit_export
                    result=submit_export(service,data['project_id'],data['series_id'],data['step_id'],data.get('source_identity'),options=data.get('options'))
                elif route == '/api/particle-job-delete':
                    from .particle_job_deletion import delete
                    result=delete(service,data['project_id'],data.get('assembly_id'),data.get('job'),data.get('confirmed',False))
                elif route == '/api/submit-particle-classification':
                    from .particle_classification import submit
                    result=submit(service,data['project_id'],data.get('assembly_id'),options=data.get('options'))
                elif route == '/api/submit-particle-postprocessing':
                    from .particle_postprocess import submit
                    result=submit(service,data['project_id'],data.get('assembly_id'),options=data.get('options'))
                elif route == '/api/submit-particle-refinement3d':
                    from .particle_refinement import submit
                    result=submit(service,data['project_id'],data.get('assembly_id'),options=data.get('options'))
                elif route in ('/api/submit-particle-polishing','/api/submit-particle-ctf_refinement'):
                    from .particle_observation_refinement import submit
                    result=submit(service,data['project_id'],data.get('assembly_id'),kind=route.rsplit('-',1)[1],options=data.get('options'))
                elif route == "/api/scan":
                    result = service.scan(data["project_id"], data["mdoc_dir"], data["movie_dir"])
                elif route == "/api/preview":
                    result = service.preview(data["project_id"], data["scan_id"], data["mdocs"], data["movies"], data.get("gain_path"), data.get("gain_flip_y", False))
                elif route == "/api/import":
                    if 'project_metadata' in data:
                        result = service.import_project(data['project_id'], data['preview_id'], data['project_metadata'], data.get('gain_path'), data.get('gain_flip_y', False))
                    else:
                        result = service.import_series(data["project_id"], data["preview_id"], data["metadata"], data.get("gains"), data.get("gain_flip_y", False))
                elif route == '/api/series-enabled':
                    result=service.set_series_enabled_many(data['project_id'],data.get('series_ids',[data.get('series_id')]),data['enabled'])
                elif route == '/api/picking-source':
                    from .picking_source import set_source
                    result=set_source(service,data['project_id'],data['series_id'],data.get('source'),step_id=data.get('step_id'),reconstruction_id=data.get('reconstruction_id'))
                elif route == '/api/reconstruction-step-create':
                    from .reconstruction_steps import create
                    result=create(service,data['project_id'],data['series_id'],data['kind'],reconstruction_id=data.get('reconstruction_id'))
                elif route == '/api/reconstruction-step-delete':
                    from .reconstruction_steps import delete
                    result=delete(service,data['project_id'],data['series_id'],data['step_id'],data.get('confirmed',False))
                elif route == "/api/edit":
                    options = {key:data[key] for key in ("gain_path","gain_flip_y") if key in data}
                    result = service.edit(data["project_id"], data["series_id"], data["metadata"], **options)
                elif route == "/api/project-metadata":
                    options = {key:data[key] for key in ("gain_path","gain_flip_y") if key in data}
                    result = service.edit_project(data["project_id"], data["metadata"], **options)
                elif route == "/api/slurm-settings":
                    result = service.save_slurm(data["project_id"], data["template"], data.get("submission_host", ""))
                elif route in ('/api/submit-picking','/api/queue-picking'):
                    from .picking import submit
                    result=submit(service,data['project_id'],data['series_ids'],data.get('options',{}),queue_only=route.startswith('/api/queue-') or data.get('queue_only',False),only_missing=data.get('submit_all',False),assembly_id=data.get('assembly_id'))
                elif route == '/api/submit-all-queued':
                    from .queue import submit_all
                    result=submit_all(service,data['project_id'])
                elif route == '/api/clean-queued':
                    from .queue import clean
                    result=clean(service,data['project_id'])
                elif route == '/api/submit-handedness-test':
                    from .handedness import submit
                    result=submit(service,data['project_id'],data['series_id'],data['rotation_deg'],data['flip_y'])
                elif route == '/api/submit-ctf-redo':
                    from .ctf_redo import submit
                    result=submit(service,data['project_id'],data['series_id'],data.get('options'))
                elif route == '/api/orientation':
                    result=service.save_orientation(data['project_id'],data['orientation'])
                elif route in ('/api/submit-reconstruction','/api/queue-reconstruction'):
                    from .reconstruction import submit
                    mode=data.get('optimizer','greco_optimizer')
                    result=submit(service,data['project_id'],data['series_ids'],data.get('denoisers',[]),mode,data.get('entry_search_sigma_px',8.),queue_only=route.startswith('/api/queue-') or data.get('queue_only',False),only_missing=data.get('submit_all',False),tomogram_thickness_percent=data.get('tomogram_thickness_percent',50.),smoothness_strength=data.get('smoothness_strength',0.),rescue_tilts=data.get('rescue_tilts',True),final_binning=data.get('final_binning'),advanced_options=data.get('advanced_options'))
                elif route in ('/api/submit-refinement','/api/queue-refinement'):
                    from .refinement import submit
                    result=submit(service,data['project_id'],data['series_ids'],data.get('options',{}),queue_only=route.startswith('/api/queue-') or data.get('queue_only',False),only_missing=data.get('submit_all',False))
                elif route in ('/api/submit-geometry-refinement','/api/queue-geometry-refinement'):
                    if data.get('submit_all'):
                        if route.startswith('/api/queue-') or data.get('queue_only',False):raise InputError('Submit first geometry refinements directly; bulk queueing is not supported.')
                        if data.get('source_kind','reconstruction')!='reconstruction':raise InputError('Invalid geometry source.')
                        from .reconstruction_steps import submit_all
                        result=submit_all(service,data['project_id'],data['series_ids'],data.get('options',{}),step_id=data.get('step_id'))
                    elif data.get('step_id'):
                        if len(data['series_ids'])!=1:raise InputError('Submit additional steps per tilt series.')
                        from .reconstruction_steps import submit
                        result=submit(service,data['project_id'],data['series_ids'][0],data['step_id'],data.get('options',{}))
                    else:
                        from .refinement import submit
                        result=submit(service,data['project_id'],data['series_ids'],data.get('options',{}),kind='geometry_refinement',queue_only=route.startswith('/api/queue-') or data.get('queue_only',False),only_missing=data.get('submit_all',False),source_kind=data.get('source_kind','reconstruction'))
                elif route in ('/api/submit-angle-refinement','/api/queue-angle-refinement'):
                    from .refinement import submit
                    result=submit(service,data['project_id'],data['series_ids'],data.get('options',{}),kind='angle_refinement',queue_only=route.startswith('/api/queue-') or data.get('queue_only',False),only_missing=data.get('submit_all',False))
                elif route in ('/api/submit-preprocessing','/api/queue-preprocessing'):
                    from .preprocessing import submit
                    result=submit(service,data['project_id'],data['series_ids'],data.get('options',{}),queue_only=route.startswith('/api/queue-') or data.get('queue_only',False),only_missing=data.get('submit_all',False))
                else:
                    self.send(404, dict(error="Unknown route."))
                    return
                self.send(200, result)
            except (InputError, OSError, ValueError, KeyError, TypeError) as error:
                self.send(400, dict(error=str(error)))

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Greco cryo-ET web interface: movie preprocessing, tomogram reconstruction, particle picking, subtomogram averaging and segmentation.")
    parser.add_argument("--port", type=int, default=1212)
    args = parser.parse_args()
    from ..user_state import RecentProjects
    service = ProjectService(recent_store=RecentProjects())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service, secrets.token_hex(32)))
    print(f"Greco web: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
