"""Run an installed Greco worker and record which shared sources it used."""
import argparse,os,runpy,sys
from pathlib import Path
from .provenance import record_code


def run(module,config):
    if not module.startswith('greco.web.') or not module.rsplit('.',1)[-1].endswith('_worker'):
        raise ValueError('Invalid Greco worker module.')
    # Set before importing workers; their subprocesses inherit this policy.
    os.environ['CUPY_CACHE_IN_MEMORY']='1'
    record_code(config,module,'started')
    original=sys.argv
    code=0
    try:
        sys.argv=[module,'--config',str(config)]
        runpy.run_module(module,run_name='__main__')
    except SystemExit as error:
        code=error.code if isinstance(error.code,int) else (1 if error.code else 0)
        raise
    except BaseException:
        code=1
        raise
    finally:
        sys.argv=original
        record_code(config,module,'finished',exit_code=code)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',required=True)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    run(args.module,args.config)

if __name__=='__main__':main()
