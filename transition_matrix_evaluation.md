# Evaluation of transition matrix goodness-of-fit

## Monte Carlo Pearson test against patient-level expected counts

### Setting

Production uses **subgroup routing**: each patient at source subspecialty `s` is assigned to an age/sex subgroup `g(i)`, and their routing vector is read from the corresponding `["subgroups"][g(i)]` table fitted by `TransferProbabilityEstimator` (see [`ISSUE_transfer_subgroup_routing.md`](ISSUE_transfer_subgroup_routing.md)). A single departure event from `s` is therefore a draw from `Categorical(p_i)`, where `p_i` is patient-specific, not from a single source-level multinomial.

The pooled `["services"]` row remains available via `estimator.get_transition_matrix(cohort)` as a diagnostic, but it is **not** what production applies and is **not** the object this test scores.

---

### Test object

For each active source `s`, the test asks the **conditional** question:

> Given that a patient departs from `s` during this window, with the subgroup mix that actually departed, do their destinations match the routing the model would have used for them?

The null hypothesis is that every observed departure event `i` from `s` was an independent draw from its own `Categorical(p_i)` with `p_i` taken from the patient's subgroup routing row.

---

### Data preparation

For each active source `s`, assemble a frame of **observed departure events** during the evaluation window. Each row carries:

- Source service `s`
- Cohort `c` (e.g. `elective` / `emergency`)
- The columns required by `estimator.subgroup_functions` to resolve `g(i)`
- Destination `d_i` (or `Discharge`)

For each event `i`, build its per-patient routing vector over the destination columns `D = columns(get_transition_matrix(c))` (transfer destinations plus `Discharge`):

- `q_transfer_i = get_transfer_prob(s, c, subgroup=g(i))`
- `q_dest_i(T) = get_destination_distribution(s, c, subgroup=g(i)).get(T, 0)`
- `p_i(T) = q_transfer_i × q_dest_i(T)` for `T ∈ D \ {Discharge}`
- `p_i(Discharge) = 1 − q_transfer_i`

Then aggregate to **expected counts** per destination:

- `E_d = Σ_i p_i(d)` for each `d ∈ D`
- `n_obs_d = #{events i : d_i = d}`
- `N = Σ_d n_obs_d` (total departures from `s`)

`E` is the patient-level expectation; it replaces the cohort-pooled `N × p` of a homogeneous-multinomial formulation.

A source is **active for evaluation** when `N ≥ 1` and `Σ_{d ≠ Discharge} E_d > 0`. Sources with all-discharge routing or no observed departures in the window are skipped (see Output).

---

### Running the test

```
for each active source s:
    E      = [Σ_i p_i(d) for d in D]                 # patient-level expectation
    n_obs  = observed destination counts for s

    T_obs = Pearson_X2(n_obs, E)

    for m in 1..M:                                   # M = 10_000 default
        for each event i:
            d_i_sim ~ Categorical(p_i)
        n_sim    = aggregate counts of d_i_sim
        T_sim[m] = Pearson_X2(n_sim, E)

    p_value[s] = (1 + count(T_sim >= T_obs)) / (M + 1)
```

Where the Pearson statistic is:

```
Pearson_X2(n, E) = sum_{d : E_d > 0} (n_d − E_d)^2 / E_d
```

with the convention that destinations where `E_d = 0` are dropped from the sum. If any observed count `n_d > 0` lands on a destination with `E_d = 0`, that is a **structural violation** (the model gave zero probability to a destination that actually received traffic) and is reported on the row alongside the test result, not folded into the statistic.

Simulation is **per patient**: each event draws its destination from its own `p_i`. There is no shared source-level `p̄` to draw a single `Multinomial(N, ·)` from once patients are heterogeneous.

---

### Output

For each active source, store:

- `T_obs` — observed Pearson X²
- `p_value` — Monte Carlo p-value
- `N` — total observed departures
- `E` and `n_obs` — full destination vectors, for drill-down
- Count of any structural violations (destinations with `E_d = 0` but `n_obs_d > 0`)

Ranking, filtering, and interpretation are out of scope for the implementation; results are stored per source.

---

### What the per-source test measures

The transition matrix is **row-stochastic**: each row is a probability vector over destinations (including `Discharge`). With subgroup routing the "row" that production actually applies is patient-specific; the test compares observed destinations against the aggregate expectation `E` built from those per-patient vectors.

**The expectation bundles two things.** Each per-patient `p_i` encodes:

1. The transfer-vs-discharge split — `1 − q_transfer_i` in the `Discharge` slot.
2. The destination distribution given transfer — the relative weights of the non-`Discharge` slots.

Pearson X² mixes both into one number. A source can be flagged because:

- The transfer-vs-discharge rate is wrong (too few or too many transfers, even if the routing among transferred patients is right), **or**
- The destination mix among transfers is wrong (transfer rate right, but routed to the wrong destinations), **or**
- both.

A single p-value cannot separate these. Use the observed-vs-expected drill-down to identify which component is miscalibrated.

**What the per-source test does not test:**

- The volume of departures from `s`. `N` is the conditioning variable, not the thing being tested. A specialty that releases twice as many patients as the model expects will not be flagged by this test — that is a separate departure-rate question.
- Target / column calibration. If destination `d` receives the right number of patients overall but they come from the wrong sources, every row can pass while the column is wrong. A per-target evaluation (column marginals: `observed_inflow(d)` vs `Σ_s E_d^s`) is the natural follow-up; it requires a different test on a different grain.
- Time-varying routing within the window. All departures from `s` are pooled into one aggregate expectation. A pathway change mid-window will look like miscalibration even if routing was correct before and after.
- Cross-source correlations. Sources are tested independently.
- The subgroup classifier itself. The test treats `g(i)` as given; if subgroups are misassigned at fit or predict time, that error is baked into both `E` and the `p_i` draws used for simulation, and the test cannot see it.

---

### Interpretation

**Overall calibration.** The primary summary is a table of active sources ranked by p-value, with `N` shown alongside, so the reader can separate significant results driven by genuine miscalibration from those driven by large `N` flagging trivial deviations.

**Role of N.**

- High `N` sources: a significant result is likely to reflect genuine miscalibration and warrants investigation.
- Low `N` sources: a significant result may be noise; a non-significant result does not confirm the row is correct, as the test has limited power.

**Drill-down for flagged sources.** For sources with significant p-values and sufficient `N`, plot `n_obs` against `E` per destination. This reveals which downstream services are being over- or under-predicted by the model and lets the reader distinguish a transfer-rate error (the `Discharge` slot is off) from a destination-mix error (non-`Discharge` slots are off).

**Implementation notes**

- Use `numpy.random.Generator.choice` per event, or build a single random uniform per event and bucket into cumulative `p_i`, to draw per-patient destinations.
- The +1 correction in the p-value formula ensures valid inference when `T_obs` is extreme.
- Store `E` and `n_obs` per source so drill-down does not require re-simulating.
- Multiple comparisons: with up to ~250 active sources tested independently, report raw p-values rather than applying a formal correction; the pattern across all rows is more informative than any binary threshold.

---

## Why Pearson X² and not the G-statistic under subgroup routing

An earlier draft of this spec used the G-statistic `G = 2 Σ n_d log(n_d / (N p_d))` against a single source-level row `p`. That choice no longer fits once subgroup routing is the live path. Three reasons:

1. **The null model is no longer a single multinomial.** With subgroup routing the observed counts are `n_d = Σ_i Y_{i,d}` with `Y_{i,d} ~ Bernoulli(p_i(d))` and patient-specific `p_i`. That is a Poisson-multinomial (a sum of independent but non-identically distributed categorical trials), not a multinomial. The G-statistic is `2 × log-likelihood-ratio` for a homogeneous multinomial; under heterogeneous `p_i` it no longer maps to any likelihood ratio and its main theoretical motivation disappears.

2. **There is no shared `p_d` to plug into `log(n_d / (N p_d))`.** Production does not use a source-level `p`; it uses per-patient `p_i` drawn from each patient's subgroup row. The only well-defined source-level quantity is the aggregate expectation `E_d = Σ_i p_i(d)`. Pearson X² is defined directly in terms of `E_d`. The G-statistic has to be reshaped as `2 Σ n_d log(n_d / E_d)`, which is no longer "the multinomial G test" but simply a function of `n` and `E`, with none of the small-cell justification that originally favoured G carrying over.

3. **Pearson decomposes cleanly at patient level.** The aggregate residual `n_d − E_d` equals `Σ_i (Y_{i,d} − p_i(d))`: a sum of per-patient signed residuals. Pearson is the standardised square of that sum. This gives a "the source-level residual is built from patient-level residuals, then standardised by expected count" interpretation that travels naturally to the heterogeneous setting. The `n_d log(n_d / E_d)` form has no analogous per-patient decomposition.

Both statistics can be plugged into the same Monte Carlo loop and will agree on most rows in practice. We standardise on Pearson because its construction is the one consistent with how production builds the expected count, and because that consistency lets the same statistic be re-aggregated cleanly to other levels (per cohort, per target, per subgroup) in any follow-up evaluation without reasoning about a likelihood model.
