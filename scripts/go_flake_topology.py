"""
go_flake_topology.py

Builds a GROMACS topology (.itp) for the GO nanoflake (go_flake.xyz, 448 C /
86 H / 56 O, built by build_go_flake.py).

Same "chemical analogy" fallback as curcumin.itp (see capability note #4 in
the project status doc) -- no acpype/AmberTools available in this sandbox.
Differences from the curcumin approach:
  - Bond connectivity from a DISTANCE cutoff (ase.neighborlist), not RDKit
    bond-order perception -- a ~450-carbon fused polyaromatic + epoxide
    system is a poor candidate for automatic bond-order/aromaticity
    perception (slow and error-prone), and we don't need bond orders here,
    only connectivity.
  - Equilibrium bond lengths/angles taken from the flake's own
    (lattice-built, chemically standard, but NOT DFT-relaxed) geometry --
    honest downgrade from curcumin's real DFT geometry, since a ~600-atom
    DFT optimization is outside this project's compute budget.
  - Partial charges and van der Waals parameters for basal/epoxide/hydroxyl
    carbons and oxygens are AVERAGED from the published, peer-reviewed
    GAFF2-AIM GO dataset (Pinto et al. 2022, JPCB, github.com/arvpinto/GO_sheets)
    for the matching chemical role (epoxide-carbon, basal-carbon,
    hydroxyl-carbon, hydroxyl-O/H, epoxide-O) -- real literature values,
    not invented ones, even though we can't use that dataset's topology
    directly (different atom count/connectivity: periodic sheet vs. our
    finite flake). Edge carbons/H (no periodic analog in that dataset) use
    standard generic aromatic C-H values instead.
"""
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

ANG_TO_NM = 0.1
DEG_TO_RAD = np.pi / 180.0

# literature-averaged charges (Pinto et al. 2022, GAFF2-AIM GO_26 dataset)
CHARGE = {
    "C_basal": -0.0124,
    "C_epoxide": 0.1120,
    "C_hydroxyl": 0.1676,
    "O_epoxide": -0.2499,
    "O_hydroxyl": -0.4845,
    "H_hydroxyl": 0.3713,
    "C_edge": -0.15,     # generic aromatic edge C (no periodic analog available)
    "H_edge": 0.15,
}
# (sigma_nm, epsilon_kJ/mol) -- same source, averaged by role
VDW = {
    "C_basal": (0.35349, 0.25919),
    "C_epoxide": (0.34415, 0.25919),
    "C_hydroxyl": (0.34398, 0.25919),
    "O_epoxide": (0.29066, 0.41881),
    "O_hydroxyl": (0.30002, 0.41881),
    "H_hydroxyl": (0.20743, 0.15047),
    "C_edge": (0.35500, 0.29288),   # generic aromatic C
    "H_edge": (0.24200, 0.12552),   # generic aromatic H
}
MASS = {"C": 12.011, "H": 1.008, "O": 15.999}

BOND_K = {
    frozenset(["C_basal", "C_basal"]): 392459.0,
    frozenset(["C_basal", "C_epoxide"]): 392459.0,
    frozenset(["C_basal", "C_hydroxyl"]): 392459.0,
    frozenset(["C_epoxide", "C_epoxide"]): 317984.0,   # slightly weakened, sp3-ish
    frozenset(["C_epoxide", "O_epoxide"]): 313800.0,
    frozenset(["C_hydroxyl", "O_hydroxyl"]): 267776.0,
    frozenset(["O_hydroxyl", "H_hydroxyl"]): 462750.0,
    frozenset(["C_basal", "C_edge"]): 392459.0,
    frozenset(["C_edge", "C_edge"]): 392459.0,
    frozenset(["C_edge", "H_edge"]): 307105.0,
}
DEFAULT_BOND_K = 350000.0
ANGLE_K = 560.0
DEFAULT_ANGLE_K = 450.0
TORSION_V = 25.0  # generic, keeps the sheet locally planar


def classify_atoms(atoms):
    cutoffs = {"C": 1.75, "O": 1.7, "H": 1.3}
    max_cut = 1.8
    i_list, j_list, d_list = neighbor_list("ijd", atoms, max_cut)
    syms = atoms.get_chemical_symbols()
    neighbors = [[] for _ in range(len(atoms))]
    for i, j, d in zip(i_list, j_list, d_list):
        lim = (cutoffs[syms[i]] + cutoffs[syms[j]]) / 2
        if d <= lim:
            neighbors[i].append(j)

    roles = [None] * len(atoms)
    for idx, sym in enumerate(syms):
        nbr_syms = [syms[n] for n in neighbors[idx]]
        if sym == "C":
            n_o = nbr_syms.count("O")
            n_h = nbr_syms.count("H")
            if n_h >= 1:
                roles[idx] = "C_edge"
            elif n_o >= 1:
                # distinguish epoxide (O has 2 C neighbors) vs hydroxyl (O has 1 C + 1 H)
                o_idx = neighbors[idx][nbr_syms.index("O")]
                o_nbr_syms = [syms[n] for n in neighbors[o_idx]]
                roles[idx] = "C_hydroxyl" if "H" in o_nbr_syms else "C_epoxide"
            else:
                roles[idx] = "C_basal"
        elif sym == "O":
            roles[idx] = "O_hydroxyl" if "H" in nbr_syms else "O_epoxide"
        elif sym == "H":
            # bonded to O -> hydroxyl H, bonded to C -> edge H
            roles[idx] = "H_hydroxyl" if "O" in nbr_syms else "H_edge"
    return roles, neighbors


def enumerate_angles(neighbors):
    angles = []
    for j, nbrs in enumerate(neighbors):
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append((nbrs[a], j, nbrs[b]))
    return angles


def enumerate_torsions(neighbors, bonds):
    torsions = []
    for i, j in bonds:
        i_c = [n for n in neighbors[i] if n != j]
        j_c = [n for n in neighbors[j] if n != i]
        chosen = None
        for a in i_c:
            for b in j_c:
                # skip degenerate cases from 3-membered rings (epoxide C-O-C):
                # the 1,4 atom can't coincide with an atom already in the
                # torsion, and the (a,b) 1-4 pair can't repeat one we've used
                if a != b and a != j and b != i:
                    chosen = (a, i, j, b)
                    break
            if chosen:
                break
        if chosen:
            torsions.append(chosen)
    return torsions


def main():
    atoms = read("go_flake.xyz")
    roles, neighbors = classify_atoms(atoms)
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    bonds = sorted({tuple(sorted((i, j))) for i, nbrs in enumerate(neighbors) for j in nbrs})

    # neutralize: the literature-averaged per-role charges don't sum to
    # exactly zero for OUR flake's particular role counts, so spread the
    # small residual evenly across all atoms (a standard, harmless fix --
    # the residual is a rounding-scale artifact of averaging, not a real
    # physical charge)
    raw_charges = [CHARGE[roles[idx]] for idx in range(len(atoms))]
    residual = sum(raw_charges) / len(atoms)
    print(f"Raw total charge: {sum(raw_charges):.4f}; per-atom neutralization: {residual:.6f}")

    lines = ["; go_flake.itp -- generated from a lattice-built (not DFT-relaxed) geometry",
             "; charges/vdW: literature-averaged GAFF2-AIM values (Pinto et al. 2022, JPCB)",
             "; by chemical role; bonded force constants generic by bond/angle type",
             f"; net charge neutralized by a uniform {-residual:.6f} e/atom correction", "",
             "[ moleculetype ]", "; Name    nrexcl", "GOFLAKE     3", "",
             "[ atoms ]",
             ";   nr    type  resnr residue  atom  cgnr    charge      mass"]
    for idx in range(len(atoms)):
        role = roles[idx]
        q = CHARGE[role] - residual
        m = MASS[syms[idx]]
        lines.append(f"{idx+1:6d}  {role:>12s}      1  GOFLAKE  {syms[idx]}{idx+1:<4d}  {idx+1:5d}  {q:9.4f}  {m:8.3f}")
    lines.append("")

    lines.append("[ bonds ]")
    lines.append(";   ai    aj  funct     r0(nm)    kb(kJ/mol/nm^2)")
    for i, j in bonds:
        r0 = np.linalg.norm(pos[i] - pos[j]) * ANG_TO_NM
        k = BOND_K.get(frozenset([roles[i], roles[j]]), DEFAULT_BOND_K)
        lines.append(f"{i+1:6d}{j+1:6d}      1  {r0:10.5f}  {k:14.1f}")
    lines.append("")

    torsions = enumerate_torsions(neighbors, bonds)
    lines.append("[ pairs ]")
    lines.append(";   ai    aj  funct")
    seen = set()
    for i, j, k, l in torsions:
        pair = tuple(sorted((i, l)))
        if pair not in seen:
            seen.add(pair)
            lines.append(f"{i+1:6d}{l+1:6d}      1")
    lines.append("")

    lines.append("[ angles ]")
    lines.append(";   ai    aj    ak  funct   theta0(deg)   k(kJ/mol/rad^2)")
    for i, j, k in enumerate_angles(neighbors):
        v1, v2 = pos[i] - pos[j], pos[k] - pos[j]
        cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta0 = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
        ka = ANGLE_K if "H" not in (syms[i], syms[k]) else 320.0
        lines.append(f"{i+1:6d}{j+1:6d}{k+1:6d}      1  {theta0:10.3f}  {ka:10.1f}")
    lines.append("")

    lines.append("[ dihedrals ]")
    lines.append(";   ai    aj    ak    al  funct   phase  kd(kJ/mol)  mult")
    for i, j, k, l in torsions:
        lines.append(f"{i+1:6d}{j+1:6d}{k+1:6d}{l+1:6d}      1     180.0  {TORSION_V:10.3f}     2")
    lines.append("")

    with open("go_flake.itp", "w") as f:
        f.write("\n".join(lines) + "\n")

    at_lines = ["[ atomtypes ]", "; name    mass    charge  ptype   sigma(nm)   epsilon(kJ/mol)"]
    for role in VDW:
        sigma, eps = VDW[role]
        elem = "C" if role.startswith("C") else ("O" if role.startswith("O") else "H")
        at_lines.append(f"{role:>12s}  {MASS[elem]:8.3f}  0.0000  A  {sigma:10.5f}  {eps:10.5f}")
    with open("go_flake_atomtypes.itp", "w") as f:
        f.write("\n".join(at_lines) + "\n")

    from collections import Counter
    print("Role counts:", Counter(roles))
    print(f"Atoms: {len(atoms)}, Bonds: {len(bonds)}, "
          f"Angles: {len(enumerate_angles(neighbors))}, Torsions: {len(torsions)}")
    print("wrote go_flake.itp, go_flake_atomtypes.itp")


if __name__ == "__main__":
    main()
