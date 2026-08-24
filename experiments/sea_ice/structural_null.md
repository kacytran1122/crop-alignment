# The structural null, stated and proved

Every effect in this paper is measured against a null. For the artefact premium and
the transfer collapse that null is *estimated* — a control arm is trained, scored,
and subtracted, and the subtraction carries the control arm's sampling error. For
crop alignment it is not estimated. It is a property of the estimand, it holds
exactly, and it holds before any data is collected. This section says why, because
that is what the rest of the paper leans on.

## Setup

A generator produces, for each crop `c` and each pixel `p` inside it, a label
`L_c(p)`. When the generator reads only the pixels inside the crop — the closed
loop this paper audits — two crops covering the same source pixel can assign it
different labels.

A predictor `f` maps a pixel *and the crop it is read in* to a class, written
`f(p | c)`. Writing the crop into the signature is the whole point: a predictor
that ignores it is the null case, and one that does not is the thing being measured.

For a source pixel `p`, let `C(p) = {c : p in c}` be the crops covering it and
`m(p) = |C(p)|`. Only pixels with `m(p) >= 2` can carry any information here.

**Artefact set.** `A = { p : there exist c, c' in C(p) with L_c(p) != L_c'(p) }` —
the pixels where the covering crops disagree about the label. Off `A` the generator
is locally single-valued and there is nothing to detect.

## The estimand

For `p in A` and `c in C(p)`:

```
own(p, c)   = 1[ f(p | c) = L_c(p) ]

other(p, c) = 1/(m(p) - 1)  *  sum over c' in C(p), c' != c  of  1[ f(p | c) = L_c'(p) ]
```

`own` asks whether the prediction matches the label of the crop being read.
`other` asks how often it matches the label of a *different* crop covering the same
pixel, averaged over those crops so that pixels with more coverage are not
overweighted.

```
kappa = E[ own(p, c) - other(p, c) ]   over instances (p, c) with p in A
```

## Proposition

> Let `f` be **crop-invariant on `A`**: `f(p | c) = f(p)` for every `p in A` and
> every `c in C(p)`. Then `kappa = 0` exactly.

## Proof

Fix `p in A`, write `m = m(p)` and `y = f(p)`, which by hypothesis does not depend
on the crop. For each class `k` let

```
n_k = | { c in C(p) : L_c(p) = k } |,        sum over k of n_k = m
```

Sum `own` over the crops covering `p`:

```
sum over c of own(p, c)  =  sum over c of 1[ y = L_c(p) ]  =  n_y
```

Sum `other` over the same crops, and exchange the order of summation:

```
sum over c of other(p, c)
    = 1/(m-1)  *  sum over c  sum over c' != c  of  1[ y = L_c'(p) ]
    = 1/(m-1)  *  sum over c'  of  1[ y = L_c'(p) ] * |{ c : c != c' }|
    = 1/(m-1)  *  n_y * (m - 1)
    = n_y
```

The inner count is the step that matters: each `c'` appears once for every `c`
other than itself, so it is counted exactly `m - 1` times, and the normaliser
cancels it.

The two sums are equal at every `p in A`, so their difference vanishes pixel by
pixel, and therefore in any average over instances:

```
kappa = 0.                                                              QED
```

Nothing in the argument uses the number of classes, the distribution of labels, the
coverage `m(p)`, the size of `A`, or any property of the imagery. The two
probabilities in `kappa` are the same quantity computed twice whenever the
prediction cannot tell the crops apart, so the null is structural rather than
statistical: there is no sampling error on it and no control arm is needed to
estimate it.

**Corollary.** `kappa != 0` certifies that `f` is crop-dependent on `A`. The sign
says more: `kappa > 0` means the dependence is *aligned with the generator* — the
model reproduces the label belonging to the window it was shown, which is the
signature of having learned the generator rather than the phenomenon.

## Why the measured control is not zero

A convolutional network reading crops is **not** a crop-invariant predictor. Zero
padding at the crop border, receptive fields truncated at the window edge, and any
normalisation computed over the crop all make `f(p | c)` depend on `c` for pixels
near a boundary. The paper measures this directly as `Omega`, the probability that
two crops covering the same pixel yield different predictions, and it is nonzero in
every real arm: `0.032` for sea-ice models trained on repaired labels, `0.061` for
flood models at `alpha = 0`.

So the empirical control does not sit at `0` — it sits at the small positive value
this residual crop-dependence produces (`+0.0157` in sea ice, `+0.0438` in floods).
That value is a **resolution floor**, not a failure of the proposition: the
proposition says what a crop-invariant predictor does, and a CNN is not one.

Consequently every reported result is the **contrast** between the closed-loop arm
and its matched control, never raw `kappa`. The proposition is what makes that
contrast interpretable: it fixes the zero point, so a control that lands near it is
evidence the machinery works rather than an assumption being granted.

## Verifying the estimator against a known answer

The proposition also supplies something rarer than a null: a case where the correct
output is known in advance, and therefore a test of the *code* rather than of the
data. A per-pixel threshold on the chip-level smoothed VH reads one pixel value and
cannot see a window at all, so it is crop-invariant literally and not merely
approximately. Running the same estimator on it must return `0` — not a small
number, `0`.

Measured on four held-out flood events, on the same artefact set as every other arm
(26.64% of covered pixels for Bolivia, roughly 10^6 instances per event):

| event   | measured \|kappa\| | measured \|Omega\| |
|---------|--------------------|--------------------|
| Bolivia | 6.8e-16            | 0.0                |
| Ghana   | 1.3e-15            | 0.0                |
| India   | 1.0e-15            | 0.0                |
| Mekong  | 8.5e-16            | 0.0                |

`P(pred = own crop label | A)` and `P(pred = other crop label | A)` agree to every
printed digit (`0.5977` against `0.5977` for Bolivia); what remains is accumulated
floating-point roundoff over sums of millions of terms, not signal.

This is the sharpest check available anywhere in the paper. Every other arm returns
a number no one can verify independently. This one has a known answer, and any
error in the vote accounting — the `(m-1)` normaliser, the own-versus-other
bookkeeping, the artefact-set mask — would surface here as a nonzero result rather
than as a plausible one somewhere else.
