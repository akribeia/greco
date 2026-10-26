"""One managed Greco server per user, with explicit release selection (Linux)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import getpass
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time

from .user_state import RecentProjects, private_directory, user_directory


class ControlError(Exception):
    pass


def instance_directory():
    root = user_directory()
    run = private_directory(root / 'run')
    # A short node identifier leaves room for Linux's 107-byte socket path limit.
    node = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]
    return private_directory(run / node)


@contextmanager
def locked(directory):
    descriptor = os.open(directory / 'control.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def read_state(directory):
    path = directory / 'server.json'
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ControlError(f'Invalid server state owner: {path}')
    return json.loads(path.read_text())


def write_state(directory, state):
    temporary = directory / ('state-' + secrets.token_hex(8))
    try:
        with temporary.open('x') as stream:
            temporary.chmod(0o600)
            json.dump(state, stream, indent=2)
            stream.write('\n')
        temporary.replace(directory / 'server.json')
    finally:
        temporary.unlink(missing_ok=True)


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect(self.host)
        except BaseException:
            self.sock.close()
            raise


def request(state, method='GET'):
    connection = UnixConnection(state['socket'], timeout=2)
    try:
        connection.request(method, '/_greco/control', headers={'X-Greco-Control': state['token']})
        response = connection.getresponse()
        data = json.loads(response.read(4096))
        if response.status != 200 or data.get('instance') != state['instance']:
            raise ControlError('Socket does not match the recorded Greco instance.')
        return data
    finally:
        connection.close()


def running(state):
    if not state:
        return False
    try:
        request(state)
        return True
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ECONNREFUSED):
            return False
        raise ControlError(f'Cannot determine server status: {error}') from error
    except (http.client.HTTPException, ValueError) as error:
        raise ControlError(f'Cannot verify Greco at {state["socket"]}: {error}') from error


def remove_stale_socket(path):
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise ControlError(f'Refusing to replace a non-socket or another user\'s file: {path}')
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(2)
    try:
        probe.connect(str(path))
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ECONNREFUSED):
            raise ControlError(f'Cannot check existing socket: {error}') from error
    else:
        raise ControlError(f'A server already owns {path}; refusing to replace it.')
    finally:
        probe.close()
    path.unlink(missing_ok=True)


def show_status(state, active):
    if not active:
        print('Greco: stopped on ' + socket.gethostname())
        return
    print(f'Greco {state.get("version", "unknown")}: running | PID {state["pid"]} | node {state["hostname"]}')
    print('Socket: ' + state['socket'])
    print('Log: ' + str(Path(state['socket']).with_name('server.log')))
    forward = f'127.0.0.1:{state["local_port"]}:{state["socket"]}'
    rule = '=' * 72
    print('\n' + rule)
    print('  CONNECT TO GRECO')
    print(rule)
    print('\n  Run on your PC:\n')
    print('    ' + shlex.join(['ssh', '-N', '-o', 'ExitOnForwardFailure=yes', '-L', forward, state['ssh_host']]))
    print('\n  Browser:\n')
    print(f'    http://localhost:{state["local_port"]}')
    print('\n' + rule + '\n')


def start(directory, previous, args, release=None):
    if running(previous):
        print('Greco is already running.')
        show_status(previous, True)
        return
    from .updates import current_release, release_environment
    release = release or current_release()
    path = directory / 'greco.sock'
    if len(os.fsencode(path)) > 107:
        raise ControlError('Socket path exceeds 107 bytes; set GRECO_USER_ROOT to a shorter private directory.')
    remove_stale_socket(path)
    state = dict(socket=str(path), hostname=socket.gethostname(), version=release['version'], release=release,
                 local_port=args.local_port or (previous or {}).get('local_port', 1212),
                 ssh_host=args.ssh_host or (previous or {}).get('ssh_host') or f'{getpass.getuser()}@{socket.getfqdn()}',
                 instance=secrets.token_hex(16), token=secrets.token_hex(32))
    write_state(directory, state)
    lease = os.open(user_directory()/'instance.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lease)
        raise ControlError('Another Greco instance is active for this user.')
    try:
        with (directory / 'server.log').open('ab') as log:
            process = subprocess.Popen([release['python'], '-m', 'greco.launcher', '_serve', str(directory)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       env=release_environment(release), cwd=release['root'], start_new_session=True, pass_fds=(lease,))
    finally:
        os.close(lease)
    state['pid'] = process.pid
    write_state(directory, state)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ControlError(f'Server exited during startup. See {directory / "server.log"}')
            if running(state):
                from .updates import atomic_json
                atomic_json(user_directory()/'active-server.json',state)
                show_status(state, True)
                return
            time.sleep(.1)
        raise ControlError(f'Server did not become ready. See {directory / "server.log"}')
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def stop(state):
    if not running(state):
        print('Greco is already stopped.')
        return
    # Stop through the authenticated socket, never through a potentially reused PID.
    request(state, 'POST')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not Path(state['socket']).exists() and not remote_active(state):
            print('Greco: stopped')
            return
        time.sleep(.1)
    raise ControlError('Server is still shutting down; check greco status and server.log.')


def serve(directory):
    from .web.projects import ProjectService
    from .web.server import make_handler
    state = read_state(directory)
    service = ProjectService(recent_store=RecentProjects())
    base = make_handler(service, secrets.token_hex(32))

    class Handler(base):
        def address_string(self):
            return 'unix'

        def control(self, shutdown=False):
            if not secrets.compare_digest(self.headers.get('X-Greco-Control', ''), state['token']):
                self.send(403, dict(error='Invalid control token.'))
                return
            self.send(200, dict(instance=state['instance']))
            if shutdown:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def do_GET(self):
            if self.path == '/_greco/control':
                self.control()
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == '/_greco/control':
                self.control(shutdown=True)
            else:
                super().do_POST()

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True
        request_queue_size = 128

    # The private parent directory protects binding before chmod is applied.
    server = Server(state['socket'], Handler)
    path = Path(state['socket'])
    identity = path.stat().st_ino
    path.chmod(0o600)
    def terminate(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        print(f'Greco socket: {path}', flush=True)
        server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        if path.exists() and path.stat().st_ino == identity:
            path.unlink()


def port(value):
    value = int(value)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError('Port must be between 1 and 65535.')
    return value


def local_state(state):
    return not state or state.get('hostname')==socket.gethostname()


def remote_active(state):
    if not state:return False
    with os.fdopen(os.open(user_directory()/'instance.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600),'w') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def require_local(state):
    if not local_state(state) and remote_active(state):
        raise ControlError('Greco '+state.get('version','')+' is running on '+state['hostname']+'. Run this command on that node.')


def managed_state(directory):
    # Prefer node state for compatibility with already running installations.
    local=read_state(directory)
    if local and running(local):return local
    path=user_directory()/'active-server.json'
    if path.is_file():
        saved=json.loads(path.read_text())
        if not local_state(saved) and remote_active(saved):return saved
    return local


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '_serve':
        serve(Path(sys.argv[2]))
        return
    # Keep explicit foreground launch arguments usable for existing deployments.
    if len(sys.argv) > 1 and sys.argv[1].startswith('--') and sys.argv[1] != '--help':
        from .web.server import main as foreground
        foreground()
        return
    parser = argparse.ArgumentParser(prog='greco', description="Run Greco's cryo-ET web interface, manage its server and install releases.")
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'restart'):
        command = commands.add_parser(name, help=name.capitalize() + ' your server')
        command.add_argument('--expected-instance', help=argparse.SUPPRESS)
        command.add_argument('--version', help='Installed release to run (start defaults to newest installed)')
        command.add_argument('--local-port', '--port', type=port, help='Laptop tunnel port (default: 1212)')
        command.add_argument('--ssh-host', help='SSH destination, e.g. user@node or a laptop SSH config alias')
    commands.add_parser('stop', help='Stop your server on this node; Slurm jobs continue')
    commands.add_parser('status', help='Check your server and print the laptop SSH tunnel')
    update = commands.add_parser('update', help='Check or install stable GitHub releases')
    update.add_argument('action', nargs='?', choices=('check',), help='Check without installing')
    update.add_argument('--version', help='Stable release to install')
    args = parser.parse_args()
    try:
        from .updates import select_release, status as update_status, install
        if args.command=='update':
            if args.action=='check':
                info=update_status(check=True)
                print('Greco '+info['current'])
                if info.get('error'):raise ControlError(info['error'])
                print('Latest release: '+info['latest']['version'])
            else:
                result=install(args.version,lambda message:print(message,flush=True))
                print('Greco '+result['version']+' installed. Use greco restart --version '+result['version']+' to activate.')
            return
        directory = instance_directory()
        # Common to every release and every node sharing this user directory.
        with locked(user_directory()):
            from .updates import register_current
            register_current()
            with locked(directory):
                state = managed_state(directory)
                if getattr(args,'expected_instance',None) and (not state or state.get('instance')!=args.expected_instance):
                    raise ControlError('The running instance changed; refresh the web page before restarting.')
                if args.command == 'status':
                    active = running(state) if local_state(state) else remote_active(state)
                    show_status(state, active)
                    if not active:raise SystemExit(1)
                elif args.command == 'stop':
                    require_local(state)
                    stop(state)
                else:
                    if args.command=='start' and ((local_state(state) and running(state)) or (not local_state(state) and remote_active(state))):
                        print('Greco is already running.')
                        show_status(state,True)
                        return
                    require_local(state)
                    version=args.version
                    if args.command=='restart' and version is None and state and running(state):
                        from .updates import current_release
                        release=state.get('release') or (select_release(state['version']) if state.get('version') else current_release())
                    else:
                        release=select_release(version)
                    if release is None:raise ControlError('That release is not installed. Use greco update --version '+str(version))
                    old_release=state.get('release') if state and running(state) else None
                    if args.command == 'restart':stop(state)
                    try:
                        start(directory, state, args, release)
                    except (ControlError,OSError) as error:
                        if args.command=='restart' and old_release and old_release!=release:
                            print('New release failed; restoring Greco '+old_release['version'])
                            start(directory,state,args,old_release)
                        raise error
    except (ControlError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Greco: {error}\n')


if __name__ == '__main__':
    main()
