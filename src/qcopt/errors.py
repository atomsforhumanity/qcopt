"""qcopt's errors and the error kinds its results report.

The kinds are those shared by AFH's programs (axtb, linus), one vocabulary across
Python exceptions (``.kind``) and qcdata results (``error.kind``):

- ``input``: settings or a structure qcopt cannot run as given;
- ``unsupported``: a shared setting qcopt cannot honour yet (``accuracy`` targets);
- ``program``: a gradient or Hessian calculation failed (its error is attached);
- ``convergence``: the run did not reach what was asked: not converged in
  ``max_iter``, or converged to the wrong kind of stationary point (``reason`` says
  which);
- ``numerical``: a numerical failure inside qcopt (singular matrices, non-finite
  values);
- ``stopped``: the caller stopped the run (``runner.stop``); it keeps what it reached;
- ``timeout``: the run ran out of the caller's time; it keeps what it reached;
- ``internal``: a bug in qcopt.
"""

from __future__ import annotations

from typing import Literal

ErrorKind = Literal[
    "input",
    "unsupported",
    "program",
    "convergence",
    "numerical",
    "stopped",
    "timeout",
    "internal",
]


class QcoptError(Exception):
    """An error raised by qcopt; ``kind`` is one of :data:`ErrorKind`."""

    kind: ErrorKind = "internal"


class InputError(QcoptError, ValueError):
    """Settings, subprograms or a structure qcopt cannot run (``kind="input"``).

    Args:
        message: What is wrong, and what to give instead.
        code: AFH's code for it, when one applies: ``unknown_key`` (not a setting),
            ``not_applicable`` (a setting of another calculation type, or given
            results qcopt cannot use), ``no_effect`` (made irrelevant by the others).
    """

    kind: ErrorKind = "input"

    def __init__(self, message: str, code: str | None = None) -> None:
        """Hold the message and its code."""
        super().__init__(message)
        self.code = code


class UnsupportedError(InputError):
    """A shared setting qcopt cannot honour yet (``kind="unsupported"``).

    An :class:`InputError`, so every check of an input catches it; its ``kind`` is what
    results report.
    """

    kind: ErrorKind = "unsupported"
