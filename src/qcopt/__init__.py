"""qcopt: minima, transition states and IRCs, as an AFH program.

qcopt's Python interface is one function::

    import qcopt
    from qcdata import ProgramInput

    out = qcopt.compute(
        ProgramInput(
            program="qcopt",
            calctype="transition_state",  # or "optimization", "irc"
            structure=guess,
            keywords={"freq": {"check": True}},
            subprograms=[
                {"program": "axtb", "calctype": "gradient", "model": {"method": "gfn2"}},
                {"program": "axtb", "calctype": "hessian", "model": {"method": "gfn2"}},
            ],
        )
    )

A failure raises ``qcdata.ProgramError``. The settings are in docs/settings.md and
:mod:`qcopt.options`. Many inputs at once go through a runner
(``qccompute.compute`` with a list), which runs them in lockstep.
"""

from importlib import metadata as _metadata

from .program import compute

__version__ = _metadata.version(__name__)

__all__ = ["__version__", "compute"]
