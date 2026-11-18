"""Skip axtb's tests when the installed axtb cannot run on the installed qccompute."""

from __future__ import annotations

import functools

import pytest


@functools.cache
def _axtb_mismatch() -> str | None:
    """Why axtb cannot run here, or None.

    axtb 0.2.0a2 and earlier state ``threads`` in their plans, a field qccompute
    0.23.0rc1 removed: every axtb calculation then fails when it is planned.
    """
    try:
        import qccompute.program as protocol
        from qcdata import ProgramInput, Structure
    except ImportError:
        return None
    program = protocol.find_program("axtb")
    if program is None or "threads" in protocol.Resources.model_fields:
        return None
    h2 = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]])
    inp = ProgramInput(
        program="axtb", calctype="gradient", structure=h2, model={"method": "gfn2"}
    )
    try:
        protocol.plan_of(program, inp)
    except Exception as exc:  # noqa: BLE001 - the reason to skip
        return f"axtb cannot be planned on this qccompute: {exc}".splitlines()[0]
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark the axtb tests skipped when axtb and qccompute do not fit together."""
    reason = _axtb_mismatch()
    if reason is None:
        return
    for item in items:
        if item.get_closest_marker("axtb") is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
