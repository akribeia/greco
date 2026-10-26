"""Command-line entry point for the production TinyCARE pipeline."""
import argparse
import json
from pathlib import Path


def run(config):
    # Keep the worker import lightweight for the web server and launchers.
    from greco._tinycare.pipeline import run as denoise
    return denoise(config)


def main():
    parser = argparse.ArgumentParser(description="Greco TinyCARE GPU denoising")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
