# mgtwin

Staged, gated authoring of microgrid digital twins. v0.6.9 -- rungs 1-4 complete; all six stages built (ingest and envelope in v0.6.6); seven gates, including loading (v0.6.4). Rung 4's adversarial identification battery measured a 33% false-confidence rate in the single-experiment protocol and drove the multi-amplitude consistency upgrade that closed it to 0%.

The wedge is microgrid controller pre-compliance screening, built against
reference networks rather than site data. Two properties of the domain make
automated authoring tractable here in a way it is not for arbitrary process
plants: the DER parameter surface is standardised by IEEE 1547, and the
validation oracle is a designed test rather than confounded production data.

## What this is

Six pipeline stages, each emitting a reviewable artifact:

| Stage | Module | Status |
|---|---|---|
| 1. ingest — tags to canonical entities | `ingest.py` | built (rules supplied per site) |
| 1b. benchmark ingest — MATPOWER `.m` to matrices | `matpower_io.py` | built |
| 1c. benchmark ingest — Simulink `.mdl` transformer/winding extraction | `mdl_io.py` | built |
| 2. compose — canonical entities to `Site` | `ontology.py`, `banshee.py` | built |
| 3. verify — `Site` to `GateReport` | `gates.py` | built |
| 4. identify — response data to parameters + honest confidence | `identify.py` | built |
| 5. test — `Site` + `TestSuite` to pass/fail | `testspec.py`, `testrun.py`, `runner.py` | built (RMS tier) |
| — unbalanced phase-frame solver (radial) | `unbalanced.py` | built, IEEE-4 verified |
| — SLG sequence fault analysis | `fault.py` | built |
| 6. envelope — declared validity envelope | `envelope.py` | built |

Run the demonstration:

```bash
pip install pydantic numpy scipy pyyaml
python examples/demo.py
python -m pytest tests/ -q      # 76 tests; 8 skip without the Banshee data (set BANSHEE_DATA)
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

## The loading gate and switch states (v0.6.4-v0.6.5)

The grounding, forming and energisation gates check that a section has a
zero-sequence path, a voltage reference and room for its inrush -- not that its
sources can carry its load. CAMPAIGN-003 showed what that misses: a black-start
state that left the halls' feeders closed on a 3 MVA genset (15-32 MVA of load)
passed every gate, as did a 3.5 MVA IT transformer feeding a 12.5 MW hall cord.
`gate_loading` takes each energised section of each islanded state and FAILs
when its load's apparent power exceeds the rating of every source energised in
it, storage included; WARNs when it exceeds the sustained sources' (the section
runs on its batteries); and FAILs a transformer feeding more load than its
rating on a side with no source of its own. It reasons on ratings, not a load
flow, so a PASS is necessary, not sufficient. `gate_units` now also checks a
transformer's reactance on its own rating (0.02-0.25 pu): a value given on the
site's base -- x 1.0 for 6 % on 3 MVA at a 50 MVA base -- is caught.

Every gate reads a state's switches the same way (`Site.open_switch_ids`): a
switch is open if the state lists it in `open_switches`, or if it is marked
`normally_closed: false` and the state does not list it in `closed_switches`.
A tie or alternate leg is therefore open unless the state closes it -- a
transfer that re-closes groups onto their second leg says so. Before v0.6.5
the gates read a normally open switch as closed in every state, which joined
Rev C's boards through their ties and second legs: its BLACKSTART_GEN put the
genset on the whole 30 MW site instead of the 15 MW it actually reaches. A
state cannot both open and close one switch. The energisation gate FAILs a
step whose transformer no energised source reaches ("the step energises
nothing"), so a state that forgets to close a leg is caught rather than passed.

## Protection: what mgtwin declares and screens, what a study decides (v0.6.8-v0.6.9)

The split: the Site is the single record of the breakers and the relay
settings installed on them; mgtwin screens those settings from ratings; a
protection study - fault currents at each relay in each state, grading,
coordination, settings, arc flash - is a network solver's job (Electrisim),
and its verdict comes back into the envelope.

**Declared.** A relay (`OvercurrentRelay`, alias `Relay`) carries a list of
`ProtectionFunction`s: ANSI code (50/51, 50N/51N, 50G/51G, 67/67N, 51V, 21,
87L/87B/87T, 27/59/59N, 81U/81O, 50BF), pickup in primary amperes, direction,
curve and time dial, the settings groups it is active in (none listed: all),
and for a unit element the buses of its zone. A 67 must have a direction and
an 87 its zone. The legacy fields - one pickup, curve, time dial, directional
flag - still declare one phase element (51, or 67 forward) active in every
group, so earlier Sites read unchanged. Each state names its
`settings_group` (default "A").

**Screened** (`gate_protection`), per islanded state, on the elements active
in its group: a closed breaker with no element active is unprotected (FAIL -
the black-start group a relay was never given); every 50/51/67 pickup against
the available converter-limited fault current with the 1.5x dependability
margin (FAIL); in a converter-dominated state, the breakers whose primary
protection rests on magnitude alone are named (WARN), and a PASS when none
does - a unit element counts for a breaker only when its zones take in both
sides of it (a feeder's 87L, a bus section between two 87B zones), not a
board's 87B on a breaker leaving the board (v0.6.9). A breaker fed from both
sides with only non-directional overcurrent sees the through-faults the far
side can drive (WARN), judged for phase and earth apart since v0.6.9: phase,
when that side's fault contribution exceeds the 1.5x margin over its rating -
a machine's 1/x'', not a converter's 1.1-1.5 pu, which a pickup above load
does not see (those breakers are counted in one PASS); earth, when each side
has an earthing source - the two neutrals in parallel a non-directional 51N
cannot locate. A 67N covers earth faults, not phase ones. On Banshee's ISL_ALL
the check finds each feeder head fed back by the diesel on another feeder.
Phase elements declared without a pickup - a design whose setting study is
still open - are named in one WARN per state rather than passed silently, and
a relay's CT ratio may be left undeclared. Earth-fault elements are not
screened by magnitude: their current is the zero-sequence path's.

**Decided by a study.** `build_envelope(..., protection_studies=[...])` takes
a `ProtectionStudyRecord` per state studied (`load_protection_studies` reads
them from JSON: state, tool, passed, whether the converter models were
identified, whether it covered the whole state, reference; fields it does not
know - a tool's working detail kept beside the record - are skipped). For
those states the study supersedes the screening: a failed study puts
protection OUTSIDE; a passed one on identified models INSIDE, on assumed ones
PROVISIONAL naming the study; a study that covered part of the state only
(`complete` False, v0.6.9) leaves it PROVISIONAL and says what it left out. A
passed, complete study also lifts a protection-gate screening FAIL, since it
computed the currents the screening only bounds. States no study covers keep
the screening's verdict.

## Banshee transformer ratings (v0.6.7)

The units gate's reactance check (v0.6.4) found that `compose_banshee` had
copied each MATPOWER branch's pu impedance - on the 20 MVA system base - into
`Transformer.x_pu`, which is on the transformer's own rating, and had paired
the branches with the .mdl's kVA ratings in list order ("they feed no gate";
since v0.6.4 they do). It now pairs the .m's transformer branches with the .mdl's instances by impedance (one-to-one within each voltage class, ties in order) and rebases their pu impedance to the paired unit's own rating. The .mdl's Z% and X/R make the
pairing exact for 19 of the 22 branches: XTX_3_4, for one, is the 500 kVA unit
its x 1.96 says, not the 2500 kVA it was given in turn. Three branches match
no instance - TX_30_40 and TX_80_110 (x 0.2977, 13.8/0.48 kV) and TX_50_60
(x 0.752, 4.16/0.48 kV) - and take their class's remaining unit, 32-51 %
apart; `transformer_pairing` lists every pair and its mismatch for review.
Every reactance is then 0.037-0.074 pu on its own rating and the units gate
passes Banshee; the Banshee tests, which skip without the data pack, had been
failing with it since v0.6.4.

## Stage 1: tag ingest (v0.6.6)

`ingest(tags, rules)` (exported as `mgtwin.ingest_tags`) maps a SCADA or historian tag list (`read_tags`: a CSV
with `tag` and optional `unit`, `description`, `sample_period_s`) to canonical
entities - kind and id as the Site lists them - and measurements (P, Q, S, V,
I, F, SOC, POS, TEMP). The rules are the site's naming convention written as
regular expressions (`TagRule`, or `load_rules` from JSON); none are built in,
because a convention guessed is a mapping invented. Every tag comes back
MATCHED, UNMATCHED (no rule, or a measurement code the rule does not map),
AMBIGUOUS (two rules disagree) or DUPLICATE, and every unit is checked against
the measurement's dimension and scaled to the canonical unit: a kW tag for P
is scaled by 1e-3, an A tag for P is flagged. `compare_with_site` then sets
the entities against the composed Site: telemetry for equipment the Site does
not hold, equipment no tag observes, and sources that cannot be identified
from operating data - no P or F tag, or, given the caller's
`identification_period_s`, sampled too slowly (rung 4's case C: a historian's
1 s does not identify what a PMU does). Ingest does not compose topology:
tags say what is measured, not what is connected.

## Stage 6: the validity envelope (v0.6.6)

`build_envelope(site, identifications)` declares where the twin is valid and
`Envelope.check(state, phenomenon, load_mva=, step_pu=)` answers INSIDE,
PROVISIONAL or OUTSIDE with the reasons. The phenomena this package
represents (loading, RMS frequency, protection screening, grounding, static
energisation) are SUPPORTED or PROVISIONAL - provisional while their
parameters are assumed; those it does not (bus voltages, EMT, dispatch,
reconnection) are OUTSIDE, with the tool that answers them. A state the Site
does not enumerate is OUTSIDE; a state a gate FAILs is OUTSIDE for that
gate's phenomena (a forming FAIL for all), and a site-wide FAIL - a units
error - puts everything OUTSIDE until it clears. Load above a state's
sustained sources is PROVISIONAL (it lasts as long as the storage), above all
its sources OUTSIDE. An `IdentificationRecord` per source carries
`identify_multi`'s verdict and the step amplitudes its experiments spanned: a
step outside them is an extrapolation and PROVISIONAL. Bounds no input
supplies - the frequency and voltage ranges the dynamic models hold over - are
listed as undeclared, not filled in. `render()` gives the reviewable Markdown,
`to_json()` the artifact.

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
