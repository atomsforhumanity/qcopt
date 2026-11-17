"""The algorithms, as generators: minimization, transition-state search, IRC.

Each yields lists of :class:`~qcopt.driver.Ask` (the geometries where it needs a
gradient or a Hessian), receives one :class:`~qcopt.driver.Answer` per ask, and
returns a :class:`RunResult`. They never call a program: :func:`qcopt.driver.drive`
runs many of them together and computes each round's asks in one call. Everything is
in Cartesian coordinates (bohr, Hartree) at the boundary; see :mod:`qcopt.space` and
:mod:`qcopt.coordinates` for the spaces the steps are taken in.

- :func:`optimize`: minima (RS-RFO) and transition states (RS-P-RFO);
- :func:`irc`: the intrinsic reaction coordinate, both branches together;
- :func:`lockstep`: several algorithms run as one (an IRC's end minimizations).
"""

from .common import RunResult, lockstep
from .optimization import optimize
from .reaction_path import irc

__all__ = ["RunResult", "irc", "lockstep", "optimize"]
