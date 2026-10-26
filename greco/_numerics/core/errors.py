"""Numerical exceptions retaining reference source-file and line context."""

from __future__ import annotations

import sys

__all__ = ['NumericalError', 'numerical_error', 'REFERENCE_SOURCE_ROOT']

# The GT binaries were compiled with __FILE__ = absolute source path, so
# their ``in: <file>, line <N>`` header carries the full source prefix.
REFERENCE_SOURCE_ROOT = "numerical-reference"


def _resolve_file(file: str) -> str:
    if file.startswith(("/", ".")):
        return file
    return REFERENCE_SOURCE_ROOT + "/" + file


class NumericalError(Exception):
    """Port of the GT exception; printing phase 1 happens in the
    constructor, exactly like the C++ one."""

    def __init__(self, file: str, line: int, msg: str) -> None:
        super().__init__(msg)
        self.file = _resolve_file(file)
        self.line = line
        self.msg = msg
        sys.stderr.write(f"in: {self.file}, line {line}\n")
        sys.stderr.write(f"ERROR: \n{msg}\n")
        sys.stderr.flush()


def numerical_error(file: str, line: int, msg: str) -> NumericalError:
    """``REPORT_ERROR(msg)`` equivalent: always raises."""
    raise NumericalError(file, line, msg)
