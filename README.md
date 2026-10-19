# mgtwin

Staged, gated authoring of microgrid digital twins. v0.6.0 -- rungs 1-4 complete. Rung 4's adversarial identification battery measured a 33% false-confidence rate in the single-experiment protocol and drove the multi-amplitude consistency upgrade that closed it to 0%.

The wedge is microgrid controller pre-compliance screening, built against
reference networks rather than site data. Two properties of the domain make
automated authoring tractable here in a way it is not for arbitrary process
plants: the DER parameter surface is standardised by IEEE 1547, and the
validation oracle is a designed test rather than confounded production data.

## What this is

Six pipeline stages, each emitting a reviewable artifact:

| Stage | Module | Status |
|---|---|---|
| 1. ingest — tags to canonical entities | — | not built |
| 1b. benchmark ingest — MATPOWER `.m` to matrices | `matpower_io.py` | built |
| 1c. benchmark ingest — Simulink `.mdl` transformer/winding extraction | `mdl_io.py` | built |
| 2. compose — canonical entities to `Site` | `ontology.py`, `banshee.py` | built |
| 3. verify — `Site` to `GateReport` | `gates.py` | built |
| 4. identify — response data to parameters + honest confidence | `identify.py` | built |
| 5. test — `Site` + `TestSuite` to pass/fail | `testspec.py`, `testrun.py`, `runner.py` | built (RMS tier) |
| — unbalanced phase-frame solver (radial) | `unbalanced.py` | built, IEEE-4 verified |
| — SLG sequence fault analysis | `fault.py` | built |
| 6. envelope — declared validity envelope | — | not built |

Run the demonstration:

```bash
pip install pydantic numpy scipy pyyaml
python examples/demo.py
python -m pytest tests/ -q      # 29 tests
```

## Three design decisions worth arguing about

**Grounding is a first-class entity, not a transformer attribute.** It changes on
islanding and positive-sequence models are blind to it. `gate_grounding` fails
any island that retains no zero-sequence source — the standard case being a
grounded-wye/delta interconnection transformer whose only grounding reference
sits upstream of the point of interconnection. The microgrid is then ungrounded
the moment it islands, single-line-to-ground faults drive the unfaulted phases
toward ~1.73 pu, and neutral-current ground-fault detection stops working. That
is a safety finding, so it is a FAIL, not a warning.

**Converters are current-limited sources, not voltage sources behind
reactance.** `gate_protection` computes islanded fault duty as rated current
times the transient limit — typically 1.1–1.5 pu — and compares it against relay
pickup with a 1.5x dependability margin. Relay settings carried over from the
grid-connected study routinely fail this. The gate also warns when converters
supply more than 80% of islanded fault current, because magnitude-based grading
is structurally the wrong scheme there even when the arithmetic passes.

**The identifier refuses confidence it has not earned.** Fisher-information
intervals are valid only if the model form is correct. Under model-form mismatch
they are tight *and* wrong, which is worse than returning nothing because a false
certainty then propagates into a protection study. So residual whiteness
(Ljung-Box plus a runs test) is evaluated *first* and a failure downgrades the
verdict regardless of interval width.

`examples/demo.py` demonstrates all four verdicts against known truth:

| Case | Experiment | Verdict | Estimate error |
|---|---|---|---|
| A | PMU rate, 2 params, correct form | `IDENTIFIED` | 0.1% |
| B | PMU rate, 3 params incl. virtual inertia | `IDENTIFIED`, trade-off named | 2.7% |
| C | historian rate (1 s, 0.05 Hz), 3 params | `NOT_IDENTIFIABLE` | −95% |
| D | PMU rate, truth has an unmodelled ramp limit | `STRUCTURAL_MISMATCH` | +178% |

Case C and case D are the load-bearing ones. C is the same plant and the same
model as B, differing only in instrumentation — so the verdict is a property of
the *experiment*, and the gate's output is an instrumentation requirement rather
than a number. D is the trap: the fit is well-conditioned, the intervals read
1.5%, and the estimate is wrong by a factor of nearly three. Only the residual
structure reveals it.

Note on case B: the transfer function is
`ω/P = −rp / [(1 + tp·s)(1 + 2·H·rp·s)]`, so the two lags enter symmetrically and
at `2·H·rp = tp` they are near-interchangeable. The least-determined direction
comes back with the right shape (`tp` and `h_sec` opposed), but at PMU rate there
is still enough information to separate them. The honest verdict is therefore
`IDENTIFIED` with the trade-off named — not a refusal. A gate that cried wolf
here would be ignored when it mattered.

## What is deliberately absent

**No solvers.** Composition should target OpenDSS or pandapower for unbalanced
QSS and an external EMT tool for faults and inrush. The RMS model in `models.py`
exists only to exercise the identification machinery against known truth; it is
not a network solver and should not grow into one. An agent that writes its own
integrator produces bugs indistinguishable from physics errors.

**No invented limits.** 20 of the criteria in the shipped test spec carry no
numeric limit, because they are project or interconnection-agreement values.
Filling them with plausible-looking numbers would be the most dangerous thing
this harness could do.

**No normative standard text.** Clause fields are blank by design and must be
populated from your own licensed copy. Test descriptions are original summaries
written for this harness, not paraphrases, and they are not a substitute for
reading the standard.

**Verified, not validated.** Reference networks establish that the
implementation is correct — code-to-code agreement, published power-flow
solutions, cross-tool comparison. They cannot establish correspondence to a real
plant, because their converter controls are documented by construction. Every
artifact should say "verified against \<benchmark\>", never "validated".


## Rung 1 result: Banshee (EPHCC release)

`examples/banshee_rung1.py` (set `BANSHEE_DATA` to the extracted
"Supporting Data for Banshee Benchmark Paper/00 models and data for system
verification" folder; download from the EPHCC GitHub release named in the
Salcedo et al. 2019 paper).

**Verification (code-to-code): PASS.** The released MATPOWER case was
translated (`matpower_io.py`), rebuilt in pandapower as impedance elements in
pu on the 20 MVA base, and solved. Against the release's own five-tool
verification table (42 breaker currents from OPAL-RT, Typhoon HIL, SKM,
MATPOWER, RTDS): **28 of 32 scored meters reproduce the published MATPOWER
column to machine precision (<0.05%)**; mean deviation 0.49% against a
published five-tool spread of 4.5% mean / 7.1% max.

The residuals are attributed, not waved away:

* The four deviating meters (3.6–4.1%) all sit on the feeder-1 subtree feeding
  bus 9 — which the case file itself annotates as the induction-machine bus
  whose demand is a placeholder to be replaced with a simulated value. Scaling
  that one load by ~+5% collapses the 8–9 branch deviation to 0.7%. The
  published run used measured machine draw; the released file carries the
  placeholder. Left as-is deliberately.
* Running the v5 case (power-sink loads) instead of v4 (passive/admittance
  loads) shifts every large current by a systematic +3.5%: the CSV's provenance
  is the passive-load case, and **load model is part of network identity** —
  the same lesson that motivates regime-split validation everywhere else.

**Composition + gates: BLOCKED, and that is the finding.** Composing the same
release into a `Site` (40 buses, 39 branches incl. 22 transformer-equivalents,
18 loads, feeder-head relays taken from the release's own settings
spreadsheets at CT 1200:5) fails the gates:

* `grounding` FAILs outright — the machine-readable artifact is
  positive-sequence and cannot carry neutral treatment. Winding
  configurations exist only on the one-line drawing.
* `protection` FAILs on the PV-only feeder-2 island (available fault current
  0.35× and 0.26× of the release's own pickups) and on the CHP island at
  1.23× — below the dependability margin.
* `honesty` WARNs on every converter: no ride-through data exists in the
  release.

So the benchmark, composed from its machine-readable data alone, is fit for
power-flow verification and *not* fit for islanding or protection
conclusions — and the pipeline states that mechanically instead of leaving it
to be discovered in review. A regression test
(`test_banshee_composition_is_blocked_for_the_right_reasons`) pins this:
if composition from this data ever passes gates, the pipeline has started
inventing zero-sequence facts.


### Rung 1 extension: winding extraction and per-zone grounding

The winding configurations rung 1 originally lacked turned out to be in the
release after all — not in the one-line drawing, but in plain text inside the
OPAL-RT Simulink model: every transformer instance references the library block
`XFR DYg-30 HIL` (delta primary, grounded-wye secondary), with kVA, voltages,
%Z and X/R as block parameters. `mdl_io.py` extracts all 29 instances; 25 are
physical (the rest are HIL stub-line coupling artifacts), and the per-voltage-
class counts exactly match the 22 MATPOWER transformer branches plus the three
50 MVA substation banks.

That data exposed a defect in `gate_grounding` itself: it reasoned per island,
but delta windings partition a network into independent **zero-sequence
zones**, so an island can be perfectly grounded at 480 V while its 13.8 kV
level floats. The gate now builds the zone graph explicitly (transformer edges
never conduct zero sequence; an open switch severs every parallel edge between
its endpoints; an energized grounded-wye winding grounds the zone of the bus
it connects to; every zone of an energized island gets a verdict, including
pure transit zones with no local source or load — the case of an MV network
backfed through a DYg service transformer).

The resulting Banshee verdict is now a data-backed statement about the
benchmark, pinned by regression test: **every islanded 13.8 kV zone is
ungrounded** (4 FAILs: ISL_ALL, ISL_F1, ISL_F2, ISL_F3), while all 67 other
zone checks pass — every LV pocket is grounded by its own service transformer
Yg, and the grid-connected MV zone by the substation banks, whose Yg sits
upstream of the feeder-head breakers and is therefore lost on islanding. The
released machine models carry no zero-sequence path and no grounding
transformer exists anywhere in the model. This mirrors what the CHIL
literature on Banshee treats as an intentional protection-study feature; the
point is that the pipeline derived it mechanically from the release, with no
one-line reading and no assumptions.


## Rung 2 result: IEEE 4-node verification and SLG quantification

`examples/ieee4_rung2.py` — no external data needed.

**Both transformer connections PASS at the rounding floor of the published
solutions.** The phase-frame ladder solver (`unbalanced.py`, Kersting's
method with generalized transformer matrices — a deliberate, narrow library
component for radial studies, not agent-written ad hoc code) reproduces the
IEEE 4-node unbalanced step-down solutions for grounded-wye/grounded-wye and
delta/grounded-wye to within 0.6 V and 0.05°, against values published
rounded to 1 V / 0.1°. Regression tests additionally pin the exact 30°
delta/wye shift at no load and the physical property that matters most here:
zero-sequence current on the loaded wye side (hundreds of amperes) crosses
the delta winding at the 1e-6 level — the power-flow face of the same fact
`gate_grounding` checks structurally.

`examples/banshee_slg.py` then quantifies the rung-1 finding with
`fault.py` (sequence-network SLG, exact ungrounded limit) for the diesel
island, as a three-way grounding design comparison:

| Design | Fault current | Worst unfaulted phase |
|---|---|---|
| As released (ungrounded) | ~0 — undetectable | 1.732 pu |
| 12 Ω low-resistance zig-zag | 605 A — a 51G works | **1.87 pu** |
| Effective grounding (X0/X1 ≈ 2) | 798 A | 1.16 pu |

The middle row is the lesson worth the price of admission, and it is pinned
by a regression test: low-resistance grounding buys *detection*, not
overvoltage relief — a dominantly resistive Z0 against a reactive Z1 rotates
the unfaulted phasors *past* √3. "Add a grounding bank" is the beginning of
the design conversation, not the end. The released benchmark sits at the
worst corner for an islanded system: full overvoltage and an undetectable
fault. For the PV-only island the √3 conclusion is identical (it is a
property of the missing zero-sequence path, not of the source), but
quantifying the remediated fault current requires the converter's
current-limited model — the same gap `gate_model_honesty` keeps flagging,
now with an engineering consequence attached.


## Rung 3 result: the test runner and its honesty taxonomy

`examples/banshee_rung3.py` (needs `BANSHEE_DATA` with `BANSHEE.mdl`, i.e.
data pack v2). The runner (`runner.py`) executes the 2030.8-shaped suite
against a composed island; the result artifact (`testrun.py`) enforces four
rules that make the report worth signing:

1. **No invented limits.** Criteria whose limits belong to the
   interconnection agreement stay UNEVALUATED — measured value shown, verdict
   withheld — unless the caller supplies `project_limits`. Regression-pinned.
2. **Gate findings block cases.** PROT-01 on the as-released island is
   BLOCKED with the rung-1 grounding finding attached, because the gate has
   already answered it: undetectable fault, √3 overvoltage. Add the zig-zag
   from rung 2 and the case unblocks into an EXECUTED/INDETERMINATE
   screening — where the 1.88 pu LRG overvoltage reappears in the metrics,
   because it is the same verified machinery underneath.
3. **Tier limits are named, not blurred.** Black start, PQ (EMT questions),
   dispatch and reconnection (missing models) come back DEFERRED with the
   reason, never quietly "answered" at the wrong tier. The protection
   screening carries its requires-EMT condition even after executing.
4. **Provenance travels with verdicts.** Every PASS on assumed dynamic
   parameters prints its condition; rung 4's identification pipeline is what
   discharges it.

On Banshee ISL_F1 with example project limits, the run reads: 3 executed
(ISL-P-01 PASS at 0.29 Hz nadir; SS-01 PASS at a 2.30 Hz droop band — the
exact rp·ΔP·f characteristic, recovered after a fix where a moving reference
had smuggled in secondary control; **ISL-U-01 FAIL at 2.57 Hz / 4.9 Hz/s**,
because a droop-only 4 MVA diesel cannot absorb an 0.86 pu step and no
load-shed scheme exists in the model), 1 blocked, 5 deferred. The FAIL is
the deliverable: it is the engineering to-do list with a number on it.


## Rung 4 result: the adversarial identification battery

`examples/rung4_identifiability_evals.py` — no external data; ~4 min for the
full double battery (`mgtwin/truthgen.py` generates truths, `mgtwin/evals.py`
scores them).

Nine scenarios × N seeds, each truth plant either the library form exactly or
containing a mechanism it deliberately lacks (ramp limiter, governor
deadband, output clamp, 150 ms actuator lag, FFR-style knee droop), plus
false-alarm probes (5× noise, 6 mHz quantization, a genuinely minor 5 ms
parasitic lag). Each run lands in a confusion cell between verdict and
consequence; the forbidden cell is **false confidence** — a clean IDENTIFIED
on estimates >5% wrong.

**The finding: single-experiment identification has a structural blind
spot.** Residual whiteness catches *shape-class* mismatch (limiter, extra
lag: refused every seed) and is blind to *amplitude-class* mismatch —
deadband, saturation, knee-droop masquerade on any one trajectory as a
parameter shift with near-white residuals. Measured false-confidence rate:
**33%**, with biases of 25–60% certified as clean. This is not a threshold
bug; no single experiment contains the evidence.

**The fix the eval forced: `identify_multi` with an amplitude-consistency
gate.** A linear model form, if correct, must return the same parameters at
every amplitude; the amplitude-dependent mechanisms cannot. Two steps
(large + small) plus a spread-vs-combined-uncertainty test drives the
false-confidence rate to **0%** while every nominal and minor probe stays
IDENTIFIED (over-refusal ~2%). Both facts are regression-pinned — including
the blind spot itself, as `test_single_experiment_is_blind_to_amplitude_class_mismatch`,
so the README claim can never silently outlive the behaviour.

**Product consequence, fed back into the commissioning procedure:** the test
protocol must specify load steps at two amplitudes minimum, and the
certified-parameter artifact must record the cross-amplitude spread. One
clean load-rejection trace is marketing, not identification. This is also
what discharges rung 3's "PASS conditional on assumed parameters" notes:
provenance flips to IDENTIFIED only through this protocol.

## Standards position

- **IEEE 2030.8**: P2030.8 is an Active PAR, approved 2026-03-26, superseding
  2030.8-2018. Test definitions are therefore versioned YAML, not code, so the
  revision does not require a rewrite. Worth tracking the revision before
  committing a customer deliverable to the 2018 edition.
- **IEEE 2030.12-2025** (microgrid protection design, published 2025-10-23) is
  the normative reference for scheme selection when `gate_protection` fails.
- **WECC generic GFM models**: `ConverterControl` maps onto REGFM_A1 (droop),
  REGFM_B1 (VSM), REGFM_C1 (hybrid) and REGCA/REECA (grid-following) so output
  speaks a language interconnection reviewers already accept. The REGFM_A1
  specification documents that steady-state current limiting and advanced
  voltage ride-through are *not* included. Those two behaviours set islanded
  fault current, so `ConverterParams` carries them separately and
  `gate_model_honesty` warns whenever they were assumed rather than identified
  from measured response. Firmware version is recorded because ride-through
  behaviour is firmware-bound — a vendor update silently invalidates the model,
  which is the technical basis for continuous re-identification as a
  subscription rather than a one-off delivery.

## Next steps

Mapped to the eval ladder:

1. ~~Compose a public benchmark from documentation alone.~~ **Done: Banshee,
   see above.** Follow-ups it generated: extract winding configurations and CB
   placement from the SKM one-line (`BANSHEE_GRIDTIED.pdf`) to clear the
   grounding gate honestly, and encode the intertie switches so multi-feeder
   reconfiguration islands become enumerable.
2. ~~Unbalanced verification on IEEE 4-node.~~ **Done: rung 2, see above.**
   Follow-up: extend the solver's element set (delta loads, 3-wire lines,
   open-delta banks) or hand larger unbalanced cases (IEEE 13/34/123) to
   OpenDSS, keeping this solver as the verified reference for radial
   transformer/grounding questions.
3. ~~Build the test runner.~~ **Done: rung 3, see above.** Follow-ups it
   generated: a load-shed model (so ISL-U-01 has a remediation to re-judge),
   an EMS dispatch model (to undefer DISP-01/02), and a synchroniser model
   (REC-01).
4. ~~Inject model-form mismatch deliberately.~~ **Done: rung 4, see above.**
   Follow-ups: extend the battery with a switched-EMT averaged truth (needs an
   EMT toy plant), and carry the two-amplitude protocol into the bench-inverter
   procedure.
5. **Bench one real inverter.** A commercial grid-forming hybrid unit in the
   5–15 kW class plus a programmable load bank gives a genuinely undocumented
   converter whose ride-through and current-limit behaviour must be identified
   the hard way. That is the only thing that tests the claimed moat, and it costs
   tens of thousands rather than eighteen months of waiting for site access.

North-star metric: engineer-hours to a first gate-clean L2 twin.
