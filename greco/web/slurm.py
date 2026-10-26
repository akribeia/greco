"""Import only Slurm configuration; pasted software commands are never reused."""
import re
import shlex

ALIASES = {'-J':'--job-name', '-o':'--output', '-e':'--error', '-n':'--ntasks',
           '-c':'--cpus-per-task', '-N':'--nodes', '-p':'--partition',
           '-A':'--account', '-t':'--time', '-C':'--constraint', '-G':'--gpus'}
DYNAMIC = {'--job-name':'job_name', '--output':'log_path', '--error':'error_path',
           '--nodes':'nodes', '--ntasks':'ntasks', '--cpus-per-task':'cpus_per_task'}
TOKEN = r"(?:XXX[A-Za-z0-9_]+XXX|\{\{\s*[A-Za-z0-9_]+\s*\}\}|<[A-Za-z0-9_]+>)"


def normalize_template(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Paste a Slurm script first.")
    if len(source) > 128 * 1024 or "\x00" in source:
        raise ValueError("Invalid template or template too large.")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"^\s*#SBATCH\s+(.+)$", line)
        if not match:
            continue
        # Some GUI templates reserve a whole directive for optional extras.
        # An unfilled slot contains no cluster configuration to import.
        if re.fullmatch(TOKEN, match[1].strip()):
            continue
        # Shell is not executed: parse option/value pairs, including quoted values.
        parts = shlex.split(match[1], comments=True)
        i = 0
        while i < len(parts):
            option = parts[i]
            if not option.startswith('-'):
                raise ValueError("Invalid SBATCH option: " + option)
            value = None
            if '=' in option:
                option, value = option.split('=', 1)
            elif len(option) > 2 and not option.startswith('--'):
                option, value = option[:2], option[2:]
            elif i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                i += 1
                value = parts[i]
            option = ALIASES.get(option, option)
            if option in DYNAMIC:
                value = '{{' + DYNAMIC[option] + '}}'
            directives[option] = value
            i += 1
    if not directives:
        raise ValueError("No #SBATCH directives found. Paste a cluster batch script.")
    # Add destinations and workload resources even if absent in the pasted script.
    for option, role in DYNAMIC.items():
        directives.setdefault(option, '{{' + role + '}}')
    warnings = []
    header = ['#!/usr/bin/env bash']
    for option, value in directives.items():
        if value is None:
            header.append('#SBATCH ' + option)
        else:
            if option not in DYNAMIC and re.search(TOKEN, value):
                warnings.append("Set the cluster value for " + option + ": " + value)
            if '\n' in value or '\r' in value:
                raise ValueError("Multiline SBATCH values are not supported.")
            encoded = value if re.fullmatch(r'[A-Za-z0-9_{}:.,/+%@=-]+', value) else shlex.quote(value)
            header.append('#SBATCH ' + option + '=' + encoded)
    body = ['set -euo pipefail', '{{environment}}',
            'export OMP_NUM_THREADS={{cpus_per_task}}',
            'cd "{{work_dir}}"', '{{command}}']
    return dict(template='\n'.join(header + [''] + body) + '\n',
                warnings=warnings, ready=not warnings)
