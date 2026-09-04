# Curcumin on Graphene Oxide — Molecular Dynamics Stability Study

**Does a favorable drug-nanocarrier binding energy actually survive real thermal motion in water?**

This is Project 2 in a two-project computational nanomedicine sequence. [Project 1](https://github.com/sareer555/curcumin-doxorubicin-GO-DFT) used DFT to show that curcumin binds a graphene oxide (GO) nanocarrier more strongly than doxorubicin does, across every level of theory tried (public dashboard: https://sareer555.github.io/curcumin-doxorubicin-GO-DFT/). That result is a single-point, static energy comparison. This project asks the next question: does curcumin actually **stay** adsorbed on a GO surface once you let it move under real 300 K thermal motion in explicit water, or does a favorable static binding energy not survive contact with reality?

**Live dashboard:** https://sareer555.github.io/curcumin-GO-MD-GROMACS/

## Headline result

Across a 500 ps unrestrained production MD run at 300 K / 1 atm, curcumin **stayed adsorbed on the GO nanoflake, and if anything settled into closer contact over time**, not further away:

| Metric | Start | End | Mean | Interpretation |
|---|---|---|---|---|
| Closest atom-atom contact distance | 0.345 nm | 0.230 nm | 0.246 ± 0.028 nm | Stayed inside van-der-Waals contact range the entire run; trend is toward closer contact |
| Center-of-mass distance (curcumin ↔ flake) | 0.79 nm | 0.62 nm | — | Curcumin moved toward the surface, not away from it |
| Curcumin heavy-atom RMSD (vs. t=0) | — | 0.227 nm (final) | 0.216 nm | The molecule flexes (expected at 300 K) but doesn't unfold |
| Curcumin radius of gyration | 0.568 nm | 0.607 nm | — | Overall shape stayed stable, std only 0.016 nm |

This is consistent with Project 1's DFT finding that curcumin binds GO favorably, and extends it: the favorable static binding energy does translate into dynamic stability, at least on the sub-nanosecond timescale sampled here.

**Honest scope caveat, stated plainly rather than glossed over:** 500 ps is short relative to real desorption/diffusion timescales. This result supports *"curcumin does not spontaneously desorb on a sub-nanosecond timescale under these conditions"* — it is not proof of long-term (µs–ms, biologically relevant) stability, which would need a literature-scale 50+ ns run this project's compute budget (a shared cloud sandbox, not a dedicated cluster) could not reach in one session. The checkpoint file needed to extend the run further without starting over is included (see "Extending this run" below).

## Why build a force field from scratch instead of using AmberTools/acpype

The standard way to parameterize a small molecule or a custom nanomaterial for GROMACS is `acpype`, backed by AmberTools' `antechamber` (GAFF2 atom-typing + AM1-BCC charges), normally installed via `conda-forge`. That path was not available in this project's compute environment: `conda.anaconda.org` was network-blocked, and a pip-only ML-based charge alternative (`espaloma_charge`) had a broken `dgl`/`torch` dependency chain against the current PyPI `torch` build (a real, currently-unresolved upstream compatibility issue).

Rather than fake a level of rigor that wasn't actually done, this project uses an explicit, documented **"parameterize by chemical analogy" fallback**:

- **Bonded geometry** (equilibrium bond lengths/angles): taken from the best available real geometry — curcumin's own DFT-optimized structure from Project 1 for the drug, and the lattice-built geometry for the GO flake (honestly weaker for the flake, since a ~600-atom DFT optimization was outside this project's compute budget).
- **Force constants** (bond/angle/torsion stiffness): generic, literature-typical values by chemical bond/angle type (aromatic C–C, C–H, C=O, etc.) — not fit to a vibrational Hessian, which would need a DFT frequency calculation.
- **Partial charges and van der Waals parameters**: curcumin's come from RDKit's built-in MMFF94 force field (a real, published, validated general organic force field, computed with zero external network calls). The GO flake's come from **literature-averaged values by chemical role** (epoxide-carbon, basal-carbon, hydroxyl-carbon/oxygen/hydrogen, etc.), pulled from the published, peer-reviewed GAFF2-AIM GO dataset of [Pinto et al. 2022, *J. Phys. Chem. B*](https://pubs.acs.org/doi/10.1021/acs.jpcb.2c02061) (`arvpinto/GO_sheets` on GitHub) — real literature numbers, not invented ones, even though that dataset's own topology (a periodic sheet, different atom count/connectivity) couldn't be used directly for this finite nanoflake.

This is stated here as plainly as it's stated in the code comments: **this is not acpype/GAFF2-equivalent rigor.** It's an honest, defensible substitute given the actual tooling constraints, not a claim of full first-principles parameterization. The GO nanoflake's size (418 carbons) and oxidation ratio (O:C = 1:8) follow the published protocol of [Wu et al. 2022, *Molecules* 27, 6742](https://www.mdpi.com/1420-3049/27/19/6742).

## Two real bugs found and fixed during force-field construction

Building a force field from scratch surfaced two genuine geometry bugs, both diagnosed from first principles rather than fixed by trial and error. Recording both here because the failure mode is a useful lesson for anyone doing similar from-scratch parameterization work.

**Bug 1 — degenerate 180° edge-carbon termination.** Cutting a rectangular flake out of a hexagonal graphene lattice leaves a handful of "corner" carbons with only 1 carbon-carbon neighbor, not the 2 a normal edge carbon has. The edge-hydrogen-placement code assumed 2 neighbors everywhere; for a 1-neighbor corner atom, the "outward direction" math degenerates and points the new H directly along the existing C–C bond, producing an unphysical, exactly-180° H–C–C angle. Fixed by adding `trim_dangling_carbons()` (`scripts/build_go_flake.py`), which removes any carbon with fewer than 2 carbon neighbors before edge-terminating. This alone took energy-minimization convergence from 1823 steps down to 61.

**Bug 2 — unrealistic hydroxyl geometry, in two parts (the real root cause of persistent NVT crashes).** The GO flake's hydroxyl oxygens were originally placed exactly perpendicular to the local ring plane, giving an exactly-90° C–C–O(H) angle. That's geometrically clean in a static picture — clean enough to survive energy minimization, since steepest descent just needs *a* local minimum, however artificially symmetric — but wrong for a real sp3-hydroxylated carbon (should be closer to the tetrahedral ~109.5°), and it broke down under real 300 K thermal motion, causing repeated LINCS constraint failures on the hydroxyl O–H bond. Fixing the C–C–O angle (tilting the C–O bond toward the tetrahedral value for one ring neighbor) surfaced a second, genuinely separate bug while placing the O–H bond: it ended up nearly collinear with the tilted C–O bond (~171° C–O–H angle instead of the correct ~106° bent alcohol geometry), which swung the hydrogen back close enough to its own parent carbon (1.49 Å) to register as a spurious direct C–H "bond" in the topology's distance-based connectivity classifier. Traced to an angle-reference-frame sign error and fixed by rederiving the O–H placement as a proper rotation by the real ~106° C–O–H angle around a perpendicular axis (`scripts/build_go_flake.py`, `decorate_basal_plane`). After both fixes: energy minimization converged in 72 steps, and NVT equilibration ran the full 50 ps at 300 K with **zero** LINCS warnings — a first, after multiple earlier attempts crashed progressively later (step 1 → step 32 → step 447) as intermediate fixes were tried.

**Lesson for anyone doing similar work:** a placeholder bonded geometry that looks fine in a static picture, or even survives energy minimization, can still be badly wrong once real thermal motion is added. NVT equilibration is exactly the stress test that catches this — which is why it exists as a distinct pipeline stage rather than being a formality.

## Pipeline

| Stage | Tool | Result |
|---|---|---|
| GO nanoflake construction | `scripts/build_go_flake.py` (ASE) | 418 C / 108 H / 26 O groups, O:C = 1:8, clash-free |
| Curcumin topology | `scripts/curcumin_topology.py` (RDKit + DFT geometry) | 47 atoms, 48 bonds, 76 angles, 26 torsions |
| GO flake topology | `scripts/go_flake_topology.py` (ASE connectivity + literature charges) | 552 atoms, 760 bonds, 1543 angles, 679 torsions, net charge = 0 |
| Complex assembly | `scripts/assemble_complex.py` | Curcumin placed using the real DFT-derived stacking geometry from Project 1, 3.6 Å clearance |
| Solvation + neutralization | `gmx solvate` + `gmx genion` (TIP3P water) | 13,949 atoms total, already net-neutral |
| Energy minimization | `gmx mdrun` (steepest descent, `mdp/minim_full.mdp`) | Converged in 72 steps |
| NVT equilibration | `gmx mdrun` (`mdp/nvt.mdp`, 50 ps, position restraints on) | Clean, 299.7 ± 2.8 K, zero LINCS failures |
| NPT equilibration | `gmx mdrun` (`mdp/npt.mdp`, 50 ps, Berendsen barostat) | Density flat at ~1018 kg/m³, no drift |
| Production MD | `gmx mdrun` (`mdp/production.mdp`, 500 ps, restraints OFF, Parrinello-Rahman barostat) | 300.1 K, ~4 bar, stable — see headline result above |

All of this ran directly in a cloud sandbox (no GPU, 2 OpenMP threads) — no need for a dedicated cluster. Wall-clock: NVT ~236 s, NPT ~352 s, production ~59 min.

## Repository layout

```
scripts/     — every Python script used to build the system, in the order they run
mdp/         — GROMACS .mdp parameter files for each simulation stage
structures/  — key structure files (initial flake, curcumin DFT geometry, assembled complex, and .gro snapshots after minimization/NVT/NPT/production)
topology/    — GROMACS .itp/.top topology files, position-restraint files, and the analysis index file
data/        — raw GROMACS analysis output (.xvg energy/distance/RMSD time series) and a solute-only production trajectory (production_solute_only.xtc, curcumin + GO flake atoms only — the full 13,949-atom trajectory including water is not included for size reasons)
figures/     — the summary analysis figure
docs/        — the public dashboard (GitHub Pages)
```

## Reproducing this

```bash
# 1. Build the GO flake and both topologies
python3 scripts/build_go_flake.py
python3 scripts/go_flake_topology.py
python3 scripts/curcumin_topology.py

# 2. Assemble the complex and the combined .gro/.top
python3 scripts/assemble_complex.py
python3 scripts/build_complex_top.py   # assembles complex.top from the pieces above

# 3. Solvate, neutralize, minimize (standard GROMACS commands)
gmx solvate -cp complex.gro -cs spc216.gro -o complex_solv.gro -p complex.top
gmx grompp -f mdp/ions.mdp -c complex_solv.gro -p complex.top -o ions.tpr -maxwarn 5
echo SOL | gmx genion -s ions.tpr -o complex_ion.gro -p complex.top -pname NA -nname CL -neutral
gmx grompp -f mdp/minim_full.mdp -c complex_ion.gro -p complex.top -o em.tpr -maxwarn 5
gmx mdrun -deffnm em -v

# 4. NVT -> NPT -> production
gmx grompp -f mdp/nvt.mdp -c em.gro -r em.gro -p complex.top -o nvt.tpr -maxwarn 5
gmx mdrun -deffnm nvt -v
gmx grompp -f mdp/npt.mdp -c nvt.gro -t nvt.cpt -r nvt.gro -p complex.top -o npt.tpr -maxwarn 5
gmx mdrun -deffnm npt -v
gmx grompp -f mdp/production.mdp -c npt.gro -t npt.cpt -p complex.top -o production.tpr -maxwarn 5
gmx mdrun -deffnm production -v

# 5. Analysis
python3 scripts/analyze_production.py
```

(`ions.mdp`, `minim_full.mdp`, `nvt.mdp`, `npt.mdp`, `production.mdp` are all in `mdp/`.)

## Extending this run

The 500 ps production run's checkpoint is not included in this repo (trajectories are large; the `.xtc` compressed trajectory in `data/` is included instead), but the same `mdp/production.mdp` with a larger `nsteps` and `gmx mdrun -deffnm production -cpi production.cpt -nsteps <new_total>` would extend the same trajectory further, strengthening the "stays adsorbed" claim's time horizon — a natural next step if more compute becomes available.

## Related work

- Project 1 (DFT binding energies): https://github.com/sareer555/curcumin-doxorubicin-GO-DFT ([dashboard](https://sareer555.github.io/curcumin-doxorubicin-GO-DFT/))

## Sources

- [Computational design of graphene oxide-based delivery of PD128763 PARP inhibitor: DFT, molecular dynamics, and molecular docking analysis](https://link.springer.com/article/10.1007/s11224-025-02692-3)
- [Theoretical Study on the Aggregation and Adsorption Behaviors of Anticancer Drug Molecules on Graphene/Graphene Oxide Surface (Wu et al. 2022)](https://www.mdpi.com/1420-3049/27/19/6742)
- Pinto, A. et al. (2022), *J. Phys. Chem. B* — GAFF2-AIM force field for graphene oxide, [`arvpinto/GO_sheets`](https://github.com/arvpinto/GO_sheets)

## License

MIT — see `LICENSE`.
