# Choosing settings: the evidence behind the defaults

The defaults come from benchmarks on 100 reactions of the Reaction-QM set (Lee et al.,
*Sci. Data* 13, 983 (2026)) with g-xTB 2.0.1 gradients from axtb, against Sella,
geomeTRIC and OptKing on the same gradients (`benchmarks/`). This page says what each
choice was measured against, so you can tell when your system is different.

## The gradient program's SCF accuracy

At axtb's default SCF accuracy (1.0), g-xTB gradients carry about 1e-5 Eh/bohr of noise
(at most 4.4e-5), the size of qcopt's convergence threshold. With `accuracy: 0.01` in the
gradient program's keywords the noise is gone, for about 12% more time per gradient. Set
it explicitly: optimizers stall or fail their final checks on noisy gradients. Other
programs have the same issue with their own SCF thresholds.

## `convergence.g_tol`

The largest atomic gradient below 0.001 eV/angstrom (1.94e-5 Eh/bohr). It is the
threshold at which AFH's g-xTB transition states stopped carrying spurious imaginary
modes (mepd's KAIST benchmark, 2026-10-19). Looser thresholds converge in fewer steps but
leave transition states that a frequency check rejects.

## `hessian.initial` and the Hessian program

- **Minima** start from Lindh's model Hessian (`auto` without a Hessian program); exact
  Hessians rarely pay for themselves at the start of a minimization.
- **Transition states** need curvature along the reaction mode: `auto` uses the Hessian
  program's, else finite differences of gradients (6N gradients).
- **A cheaper Hessian program** is usually the best trade. On the 100 transition states
  at g-xTB, GFN2-xTB starting Hessians (`hessian_program=gfn2`) halved the total time and
  found a first-order saddle in all 100 (98 the same as from the exact start). A g-xTB
  Hessian costs about 38 gradients in axtb, a GFN2-xTB one about 8.
- Frames always hold the gradient program's energies and gradients. A cheaper method's
  frequencies are reported but decide nothing; with the method's own Hessians,
  `freq.check` makes them decide.

## `hessian.recompute_every`

Recomputing the exact Hessian every few steps did not pay at g-xTB: on the same
benchmark, where a Hessian costs about 28 gradients, every schedule tried (1, 3, 5) cost
more than the default (`0`: start, then quasi-Newton updates). With a lower-level Hessian
program it also replaces the updated Hessian, which loses the target method's curvature
(GFN2 Hessians every step: 67 of 100 g-xTB saddles). At GFN2-xTB itself, where a Hessian
is cheap, schedules roughly break even.

## `irc.step` and `irc.method`

On the 100 IRCs at g-xTB (default SCF accuracy), Gonzalez-Schlegel steps at 0.3
bohr amu^1/2 reached the same minima as Sella in all 100 with a median of 375 gradients
(ends included), against 528 for the local quadratic approximation at 0.1; LQA at 0.3
and GS2 at 0.5 each ended one IRC in another minimum. With qcopt 0.2.0 at SCF accuracy
0.01, GS2 at 0.3 matched Sella's minima in 94 (of the other 6, 4 ended in neighboring
minima of the same species) with a median of 349.5 gradients, ends included; in 16 a
branch reached `convergence.max_iter`.

## The IRC's end test (`convergence.g_tol`, `ends.minimize`)

An IRC branch ends once its largest atomic gradient is below 5e-4 Eh/bohr with positive
curvature: an IRC stops short of the minimum by design, and minimizing the ends
(`ends.minimize`, or separately) finishes the descent. In 21 of 83 finished IRCs, a
minimized end dropped more than 1 kcal/mol (Sella: 3 of 100).

From a very soft transition state the first step can already meet the test: from a
g-xTB saddle with a 27.5i cm^-1 mode (Reaction-QM rxn_043), both branches ended after one
step, about 20 kcal/mol above the minima their minimizations reached. Use
`ends.minimize` for such systems.
