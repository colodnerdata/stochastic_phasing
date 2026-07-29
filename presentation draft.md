Stochastic Expenditure Phasing: Coupling Uncertain Duration with Uncertain Spend Profiles
45-Minute Presentation Draft — Slide-by-Slide with Speaker Notes
Audience: Cost estimators (DOE/NNSA community)
Timing plan: Motivation ~8 min | Methodology ~12 min | Monte Carlo coupling ~12 min | Interpretation ~10 min | Wrap-up/Q&A buffer ~3 min
Palette/figure notes reference the pipeline outputs (01–05 PNGs) plus new figures specced inline as [FIG-x].
SECTION 1 — MOTIVATION (Slides 1–7, ~8 min)
Slide 1: Title
Stochastic Expenditure Phasing
Modeling when the money moves — with uncertainty
[Name, org, date, venue]
Notes: One-liner framing: "We're good at estimating how much and how long. This talk is about when — and why treating 'when' as uncertain changes the answers to questions we're already being asked."
Slide 2: The Questions Phasing Must Answer
What budget authority is needed in each fiscal year?
How much year-of-execution escalation exposure do we carry?
Is expenditure-to-date consistent with plan, or a warning sign?
If the schedule slips, what happens to the outyear profile?
Notes: All four are temporal questions. A total-cost S-curve (the risk-analysis kind) answers none of them. Emphasize the audience already answers these — usually with a deterministic spread.
Slide 3: Current Practice and Its Blind Spot
Common approaches: fixed percentage spreads, analogy to one “representative” project, hand-built profiles
All share one assumption: the phasing shape is known
Historical data says otherwise — shape varies project to project, a lot
[FIG-1: 01_curve_overlays.png — all historical TPC curves overlaid.]
Notes: Point at the spread at mid-schedule: "At 50% schedule, our completed projects had spent anywhere from X% to Y%. A deterministic profile picks one line out of this fan and bets the budget request on it."
Slide 4: Why This Bites — Three Concrete Failure Modes
Budget formulation: deterministic profile → point annual requests → no basis for reserves phasing
Escalation: back-loaded reality vs. front-loaded plan = under-escalated estimate
Execution review: "behind plan" may be within normal shape variation — or not; no way to tell without a distribution
Notes: The third is the EVM hook for this audience: a phasing distribution gives you control limits for percent-spent-vs-percent-schedule, turning a vibes-based conversation into a statistical one.
Slide 5: The Coupling Problem
Duration is uncertain (we already model this)
Phasing shape is uncertain (we usually don't)
Annual spend depends on both at once: a 60% cumulative point means different FY dollars in a 4-year execution vs. a 6-year execution
These uncertainties compound in the outyears — exactly where budget decisions live
Notes: This is the thesis slide. The rest of the talk builds the machinery to do this compounding honestly and then — the part people get wrong — to read the outputs correctly.
Slide 6: What We Built (Preview)
Fit each completed project's normalized spend curve to a 2-parameter family
Model the fitted parameters as a population → sample new plausible curves
In Monte Carlo: draw duration + draw curve + draw cost → simulated FY-by-FY expenditure
Entire phasing model = 5 numbers per stratum (2 means, 2 SDs, 1 correlation)
Notes: Land the compactness point: auditable, publishable in one table row, no black box.
Slide 7: Roadmap
Motivation ✓ → Methodology → Monte Carlo coupling → How to read the results (the part with the traps)
SECTION 2 — METHODOLOGY (Slides 8–17, ~12 min)
Slide 8: Data and Normalization
[16] completed projects, [annual] expenditure snapshots, three cost accumulations (TEC / OPC / TPC); TPC is today's focus
Each project normalized to the unit square: cumulative % of realized schedule vs. cumulative % of realized cost
Deliberate choice: normalize by actuals, not baselines — the model's job in simulation is to spread a total across a duration
Notes: Field the inevitable question preemptively: growth is handled by the companion cost/duration models; phasing describes shape given the totals. State the completion convention (how "100% schedule" is defined) — it matters for tail behavior.
Slide 9: The Curve Family — Beta CDF
Model: %cost = F(%schedule; α, β), the Beta CDF
Why Beta: bounded [0,1] support matches the problem exactly; two parameters span front-loaded, back-loaded, symmetric, and linear shapes
[FIG-2: 2×2 grid of Beta CDF shapes — (α<β) front-loaded, (α>β) back-loaded, (α=β>1) symmetric S, (α=β=1) linear. Annotate each.]
Notes: This is established practice in the community (cite handbook lineage) — the new part comes two slides later.
Slide 10: The Interpretation That Makes It Talkable
The Beta density f(t; α, β) is the distribution of when a dollar is spent
Fraction of cost in any schedule window = F(t₂) − F(t₁)
Mean spend timing = α/(α+β) → one number for front/back-loading
Peak spend rate at (α−1)/(α+β−2)
Notes: This is the slide that makes the method reviewable by non-statisticians. "Project X has mean spend timing 0.58" is a sentence a program manager can argue with.
Slide 11: Stage 1 — Fit Every Project Individually
Nonlinear least squares of the Beta CDF to each project's interior points
Initialization by method of moments on spend increments (treat annual increments as probability mass → moment-match α, β)
Diagnostics per curve: R², RMSE, max absolute error, bound-contact flag
Notes: One sentence on why initialization matters (CDF least-squares is start-sensitive) and one on honesty (LS on cumulative data → autocorrelated residuals → SEs are indicative). With annual data, points per curve are few — which is exactly why Stage 2 pools.
Slide 12: Fit Results
[n] curves fit; [report convergence, median R², RMSE, worst max-abs-err]
[Show 2–3 representative fits: a front-loaded, a back-loaded, a typical]
[FIG-3: three small-multiple panels, actuals as dots + fitted curve.]
Notes: If a curve fit poorly, show it and say why (funding pause, closeout tail) — showing the worst case buys credibility for everything else.
Slide 13: Stage 2 — The Fitted Parameters Are the Data
Each project reduces to a point (α̂, β̂)
The cloud of points is the empirical distribution of phasing behavior
Modeling target: the joint distribution of (α, β) across projects
[FIG-4: 02_scatter_ellipses.png — (α, β) scatter with 95% ellipses, unit and log space.]
Notes: This is the methodological pivot and the actual contribution: instead of picking one curve, we characterize the population of curves.
Slide 14: Choosing the Population Model
Candidates: bivariate normal in unit space vs. log space (bivariate lognormal)
Log space preferred: guarantees positive parameters, symmetrizes right skew, multiplicative variation is the natural scale
Checked empirically: marginal normality tests + Mahalanobis–χ² plots in both spaces
[FIG-5: 04_qq_normality.png.]
Notes: Keep to 60 seconds. The message is "we tested the assumption, here's the evidence," not a stats lecture.
Slide 15: The α–β Correlation
Measured in unit and log space; [report r, ρ, significance]
Substantive meaning: moving along the α=β diagonal changes steepness; moving across it changes front/back-loading
Correlation retained in the sampling model regardless of significance — costs nothing, drops nothing
[FIG-6: 03_correlation_panels.png, TPC row.]
Notes: If correlation is high and positive: "our projects vary more in how sharply they ramp than in which direction they lean." That's a finding worth saying aloud.
Slide 16: The Complete Phasing Model
TPC: (ln α, ln β) ~ BVN(μ, Σ) — [table: μ_lnα, μ_lnβ, σ_lnα, σ_lnβ, ρ]
Sampling a future project's curve = one draw from this distribution
Vintage drift check: [report Spearman vs. start year] → pooling across [2002–2019] is [supported]
Notes: Pause here. "This table is the model. Everything in the next section consumes these five numbers."
Slide 17: Honest Caveats (Before Anyone Asks)
n = [16]: population parameters carry sampling error the generator doesn't propagate → bands modestly optimistic
Beta family can't do multi-modal profiles (stop-work, split funding)
Hierarchical Bayes is the principled upgrade — this method is its front end, and its results initialize that model
Notes: Sixty seconds, unapologetic. Naming the hierarchical extension signals you know where this sits on the rigor ladder.
SECTION 3 — MONTE CARLO: COUPLING DURATION AND PHASING (Slides 18–25, ~12 min)
Slide 18: The Simulation Recipe
Per iteration i:
Draw total cost C⁽ⁱ⁾ (cost model)
Draw duration D⁽ⁱ⁾ (schedule model)
Draw phasing (α⁽ⁱ⁾, β⁽ⁱ⁾) ~ exp[BVN(μ̂, Σ̂)]
Cumulative profile: c(m) = C⁽ⁱ⁾ · F(m/D⁽ⁱ⁾; α⁽ⁱ⁾, β⁽ⁱ⁾), m = 0…D⁽ⁱ⁾
Difference into periods → map to fiscal years → apply escalation by year of execution
Notes: Emphasize the separation of concerns: the phasing draw is on normalized time, so the same curve draw stretches over whatever duration was drawn. Steps 1–3 can be independent or correlated — a modeling choice, next slide.
Slide 19: Independence Is an Assumption — Test It
Default: (α, β) ⊥ D ⊥ C
Testable: regress (ln α̂, ln β̂) on log realized duration / size across the historical set → [report result]
If long projects systematically back-load: add correlation via a Gaussian copula over (ln α, ln β, ln D) — no structural change
Notes: This defuses the sharpest technical objection in the room. If your empirical check found no relationship, say so and show the scatter in backup.
Slide 20: What Coupling Does That Sequential Doesn't
Deterministic phasing on top of stochastic duration: bands in calendar time come only from stretching one shape
Coupled model: shape variation and stretching → wider, more honest bands, especially mid-execution
Side-by-side comparison:
[FIG-7: two cumulative fan charts in calendar months, same duration distribution; left = fixed mean (α, β), right = sampled (α, β). Annotate band widths at a mid-execution month.]
Notes: This figure is the "so what" of the whole methodology section. Quantify: "at month [18], the 90% band is [X] points wider once phasing uncertainty is included."
Slide 21: From Months to Fiscal Years
Iteration output = monthly (or annual) spend stream in execution time; budget questions live in fiscal years
Bin each iteration's stream into FYs given a start date; escalate each FY's spend by its own index
Duration uncertainty now shows up as which FYs exist at all in a given iteration
Notes: Plant the seed for Section 4: "notice that in a 4-year draw, FY+6 has zero spend — not small spend, zero. Hold that thought; it's about to matter a lot."
Slide 22: Escalation Exposure — A Free Deliverable
Longer draws push spend into later, more-escalated years → then-year cost is correlated with duration even with constant-year cost fixed
The simulation yields the full distribution of escalation dollars, not a point adjustment
[Illustrative: P80 duration scenario carries $[X]M more escalation than P20]
Notes: For a DOE audience this alone can justify the model. Keep to 90 seconds.
Slide 23: Simulation Outputs — The Raw Material
Per iteration: {C, D, α, β, FY spend vector}. From [10,000] iterations we can compute:
Cumulative spend fan charts (calendar time)
Annual spend distributions by FY
P(project active in FY y)
Joint statistics (spend-to-date vs. eventual total, etc.)
Notes: Transition: "We now have everything. The remaining problem — and it is a real problem — is that the obvious way to summarize these outputs produces numbers people will misread."
Slide 24: A Worked Example Setup
Notional project: cost model [$X ± Y], duration model [median 60 mo, P90 84 mo], TPC phasing population
All Section 4 figures come from this single simulation
Notes: Fix the example now so Section 4 is concrete, not hypothetical.
Slide 25: [Reserve/flex slide — live demo or additional example]
SECTION 4 — INTERPRETING THE RESULTS (Slides 26–33, ~10 min)
Slide 26: The Outyear Problem, Stated
FY spend in a late year is zero in every iteration where the project finished earlier
So annual spend in FY y is a mixture: a point mass at $0 (weight = P(D ends before y)) plus a continuous part
Ordinary percentiles behave strangely on mixtures — and late FYs are always mixtures
[FIG-8: histogram of FY+7 spend across iterations — visible spike at zero plus a right lobe. Annotate: "P(zero) = 1 − P(active in FY+7) = 0.XX".]
Notes: This is the intellectual core of the section. Everything that follows is a consequence of this one histogram.
Slide 27: Consequence 1 — The Median Profile Ends Too Early
q₅₀(FY y spend) = $0 whenever P(active in y) < 50%
So the "P50 annual profile" silently truncates at the median completion date — even though half of all runs spend money beyond it
The P50 profile is not the median scenario; no iteration looks like it
[FIG-9: annual bars of q10/q50/q90 by FY. Highlight the FY where q50 collapses to zero while q90 remains large.]
Notes: Say the trap out loud: "If you hand this chart to a budget office, they will read the P50 row as 'the realistic profile' and conclude the outyears are free. They are not free — they're contingent."
Slide 28: Consequence 2 — High Quantiles in Late Years Are Conditional Realities in Disguise
q₉₀(FY+7 spend) mixes two different questions:
Will the project still be running? (probability statement)
If it is, how much will it spend? (conditional magnitude)
A large unconditional P90 in FY+7 does not mean "10% chance we need $Z in FY+7" in the intuitive sense — it reflects the joint event {still active} ∧ {high spend given active}
Recommended decomposition, per FY: report P(active) and the conditional spend distribution given active, side by side
[FIG-10: dual-axis by FY — bars: conditional mean/quantiles of spend given active; line: P(active in FY). The line falls as bars stay meaningful.]
Notes: This is the slide the talk title promised. Walk one FY end-to-end: "FY+7: 22% of runs are still active; those runs spend a median of $9M there. Say it that way — never 'the FY+7 P50 is zero.'"
Slide 29: Consequence 3 — Quantile Profiles Aren't Scenarios (Don't Sum Them)
The vector of per-FY P80s is not an executable profile: sum of annual quantiles ≠ quantile of the total
Funding every year at its unconditional P80 generally over-programs — early and late FY spends are negatively related through duration (long draws shift money out of early years into late ones)
Means are the exception: the mean annual profile sums exactly to the mean total. Quantiles don't aggregate; means do.
Notes: Invite the audience to see the mechanism: one duration draw can't be simultaneously short (fat early years) and long (fat late years), but the marginal quantile table implicitly assumes it can.
Slide 30: What to Report Instead — A Recommended Package
Mean annual profile — additive, ties to mean total; the TOA-planning backbone
P(active in FY) — the schedule-risk content of the outyears, on every chart
Conditional annual distributions given active — honest outyear magnitudes
Cumulative fan chart in calendar time — the executive picture
Scenario profiles: phasing of iterations near the P20/P50/P80 total (coherent, executable profiles — unlike quantile vectors)
[FIG-11: one-page mock dashboard combining 1–4.]
Notes: Item 5 is the practical replacement for "the P80 profile": select iterations within a band of the P80 total (or duration) and show their median profile — a real profile from real draws.
Slide 31: Reading the Fan Chart Correctly
Band width in early years ≈ phasing-shape uncertainty; in late years ≈ duration uncertainty dominating
The upper band's long tail is "still spending because still executing," not "spending more overall"
Where the P50 cumulative flatlines = median completion; where P90 flatlines = P90 completion — the horizontal gaps are schedule risk made visible
[FIG-12: annotated cumulative fan chart with these three callouts drawn on.]
Notes: Give the audience the literacy to brief this chart without you in the room. The horizontal-vs-vertical reading (schedule risk vs. cost-timing risk) is the takeaway.
Slide 32: Execution-Phase Use — Phasing Control Limits
Invert the model: given %schedule today, the population of curves implies a distribution of %cost
Actuals outside the [10th–90th] band → shape anomaly worth investigating; inside → normal variation, resist re-planning
[Optional: Bayesian update of (α, β) from actuals-to-date → refreshed EAC-timing forecast]
Notes: Closes the loop for the EVM practitioners; one slide only, flag the update machinery as future work.
Slide 33: Summary
Phasing shape is empirically variable → model it as a distribution, not a template (5 numbers per stratum)
Couple it with duration and cost in Monte Carlo — coupling is where escalation and outyear risk actually live
Report means, P(active), and conditional distributions; never sum quantile profiles, never read the P50 profile as a plan
Path forward: hierarchical Bayes for small-n honesty; covariates; correlation with duration if data supports it
Notes: Land on the interpretation rules — that's the durable value even for someone who never runs the model.
BACKUP SLIDES
B1: Method-of-moments initializer derivation
B2: Unit-space vs. log-space population fit comparison (parameter table + sampled-curve overlay differences)
B3: (α̂, β̂) vs. realized duration/size scatters — the independence check evidence
B4: Sum-of-quantiles vs. quantile-of-sum numeric demo from the worked example ([table: ΣFY P80s = $X vs. total P80 = $Y])
B5: Annual-vs-monthly granularity: effect on per-fit SEs and why Stage-2 pooling compensates
B6: Hierarchical Bayesian formulation sketch (one slide of structure, no derivation)
Production notes (not slides)
Figures FIG-7 through FIG-12 are new; all are single-simulation products — extend the existing script with an FY-binning + reporting module (happy to write it)
Slide count ≈ 33 + backups suits 45 min at your pace (~80 sec/slide with two 2-min anchor slides: 20 and 28)
Anchor slides to rehearse: Slide 20 (coupling payoff figure) and Slide 28 (conditional decomposition) — the talk's two "remember this" moments