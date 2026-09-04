"""
curcumin_topology.py

Builds a GROMACS topology (.itp) for curcumin from our own DFT-optimized
geometry (Project 1, curcumin.out), since the sandbox has no network path
to AmberTools/conda-forge (acpype's normal backend) or to a working
ML-charge tool (espaloma_charge's dgl/torch dependency chain is broken
against the newest torch on PyPI here).

Methodology (documented explicitly, not hidden):
  - Bond connectivity: perceived from the DFT 3D geometry via RDKit
    (rdDetermineBonds), then sanitized/aromaticity-perceived.
  - Equilibrium bond lengths and angles: taken DIRECTLY from the DFT
    geometry itself (our own B3LYP/6-31G* result), not a generic table --
    this is the one part of the topology that's genuinely first-principles.
  - Force constants: generic literature-typical values by bond/angle
    chemical type (aromatic C-C, C-H, C=O, C-O ether, O-H, etc.) -- fitting
    real force constants requires a vibrational Hessian, which is outside
    this project's scope, so these are honestly generic, not system-fit.
  - Partial charges and van der Waals (sigma/epsilon): from RDKit's native
    MMFF94 force field (a real, published, validated general organic force
    field), self-consistently computed with no external network calls.
    MMFF's R*/eps (buffered 14-7 form) are converted to LJ sigma/epsilon
    via the standard rmin->sigma relation (sigma = R* / 2^(1/6)).
  - Torsions: a single generic periodic term per bond, high barrier
    (keeps ring/conjugated systems planar) for aromatic/conjugated bonds,
    low barrier (free rotation) for other single bonds.

This is an explicit, honest "parameterize by chemical analogy" fallback --
not a claim of acpype/GAFF2-equivalent rigor. See the project status doc
for why.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

KCAL_TO_KJ = 4.184
ANG_TO_NM = 0.1
DEG_TO_RAD = np.pi / 180.0

# generic force constants, kJ/mol/nm^2 (bonds) and kJ/mol/rad^2 (angles)
BOND_K = {
    frozenset(["C_ar", "C_ar"]): 392459.0,
    frozenset(["C_ar", "H"]): 307105.0,
    frozenset(["C_sp3", "H"]): 284512.0,
    frozenset(["C_sp2", "O_carbonyl"]): 476976.0,
    frozenset(["C_ar", "O_ether"]): 267776.0,
    frozenset(["C_sp3", "O_ether"]): 267776.0,
    frozenset(["C_ar", "O_hydroxyl"]): 267776.0,
    frozenset(["O_hydroxyl", "H"]): 462750.0,
    frozenset(["C_sp2", "C_sp2"]): 449000.0,
    frozenset(["C_sp3", "C_sp2"]): 265265.0,
    frozenset(["C_sp2", "C_ar"]): 317984.0,
    frozenset(["C_sp3", "C_ar"]): 265265.0,
    frozenset(["C_sp3", "C_sp3"]): 224262.0,
}
DEFAULT_BOND_K = 300000.0

ANGLE_K_AR = 560.0     # sp2/aromatic-heavy angle, kJ/mol/rad^2
ANGLE_K_SP3 = 460.0
ANGLE_K_H = 320.0      # any angle involving an H terminus
DEFAULT_ANGLE_K = 400.0

TORSION_V_CONJUGATED = 25.0   # kJ/mol, keeps planarity
TORSION_V_FREE = 3.0          # kJ/mol, free-ish rotation (e.g. OCH3, alkyl)


def atom_type(atom):
    sym = atom.GetSymbol()
    if sym == "H":
        return "H"
    if sym == "C":
        if atom.GetIsAromatic():
            return "C_ar"
        hyb = atom.GetHybridization()
        return "C_sp2" if str(hyb) == "SP2" else "C_sp3"
    if sym == "O":
        # crude but adequate: carbonyl (double bond to C), hydroxyl (has H
        # neighbor), else ether
        for b in atom.GetBonds():
            if b.GetBondTypeAsDouble() == 2.0:
                return "O_carbonyl"
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == "H":
                return "O_hydroxyl"
        return "O_ether"
    return sym


def load_mol():
    with open("curcumin_dft.xyz") as f:
        xyz_block = f.read()
    raw = Chem.MolFromXYZBlock(xyz_block)
    mol = Chem.Mol(raw)
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    Chem.SanitizeMol(mol)
    return mol


def enumerate_angles(mol):
    angles = []
    for atom in mol.GetAtoms():
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append((nbrs[a], atom.GetIdx(), nbrs[b]))
    return angles


def enumerate_torsions(mol):
    torsions = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom_j, atom_k = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
        i_candidates = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != k]
        l_candidates = [n.GetIdx() for n in atom_k.GetNeighbors() if n.GetIdx() != j]
        if not i_candidates or not l_candidates:
            continue
        i, l = i_candidates[0], l_candidates[0]
        conjugated = bond.GetIsAromatic() or bond.GetIsConjugated()
        torsions.append((i, j, k, l, conjugated))
    return torsions


def geom_bond_length(conf, i, j):
    pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
    return np.linalg.norm(np.array([pi.x, pi.y, pi.z]) - np.array([pj.x, pj.y, pj.z]))


def geom_angle(conf, i, j, k):
    pi = np.array(conf.GetAtomPosition(i))
    pj = np.array(conf.GetAtomPosition(j))
    pk = np.array(conf.GetAtomPosition(k))
    v1, v2 = pi - pj, pk - pj
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_t, -1, 1))


def main():
    mol = load_mol()
    conf = mol.GetConformer()
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    types = [atom_type(a) for a in mol.GetAtoms()]

    lines = []
    lines.append("; curcumin.itp -- generated from DFT-optimized geometry (Project 1)")
    lines.append("; force field: generic bonded params by chemical analogy + MMFF94 charges/vdW")
    lines.append("")
    lines.append("[ moleculetype ]")
    lines.append("; Name            nrexcl")
    lines.append("CURC                3")
    lines.append("")
    lines.append("[ atoms ]")
    lines.append(";   nr  type  resnr residue  atom  cgnr    charge      mass")
    elem_mass = {"C": 12.011, "H": 1.008, "O": 15.999}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        q = props.GetMMFFPartialCharge(idx)
        lines.append(f"{idx+1:6d}  {types[idx]:>10s}      1    CURC   {sym}{idx+1:<3d}  {idx+1:4d}  {q:10.4f}  {elem_mass[sym]:8.3f}")
    lines.append("")

    lines.append("[ bonds ]")
    lines.append(";   ai    aj  funct     r0(nm)    kb(kJ/mol/nm^2)")
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0 = geom_bond_length(conf, i, j) * ANG_TO_NM
        k = BOND_K.get(frozenset([types[i], types[j]]), DEFAULT_BOND_K)
        lines.append(f"{i+1:6d}{j+1:6d}      1  {r0:10.5f}  {k:14.1f}")
    lines.append("")

    lines.append("[ pairs ]")
    lines.append(";   ai    aj  funct")
    torsions = enumerate_torsions(mol)
    seen_pairs = set()
    for i, j, k, l, _ in torsions:
        pair = tuple(sorted((i, l)))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            lines.append(f"{i+1:6d}{l+1:6d}      1")
    lines.append("")

    lines.append("[ angles ]")
    lines.append(";   ai    aj    ak  funct   theta0(deg)   k(kJ/mol/rad^2)")
    for i, j, k in enumerate_angles(mol):
        theta0 = geom_angle(conf, i, j, k) / DEG_TO_RAD
        if "H" in (types[i], types[k]):
            ka = ANGLE_K_H
        elif types[j] in ("C_ar", "C_sp2"):
            ka = ANGLE_K_AR
        elif types[j] == "C_sp3":
            ka = ANGLE_K_SP3
        else:
            ka = DEFAULT_ANGLE_K
        lines.append(f"{i+1:6d}{j+1:6d}{k+1:6d}      1  {theta0:10.3f}  {ka:10.1f}")
    lines.append("")

    lines.append("[ dihedrals ]")
    lines.append(";   ai    aj    ak    al  funct   phase  kd(kJ/mol)  mult")
    for i, j, k, l, conjugated in torsions:
        v = TORSION_V_CONJUGATED if conjugated else TORSION_V_FREE
        phase = 180.0 if conjugated else 0.0
        mult = 2 if conjugated else 3
        lines.append(f"{i+1:6d}{j+1:6d}{k+1:6d}{l+1:6d}      1  {phase:8.1f}  {v:10.3f}  {mult:4d}")
    lines.append("")

    with open("curcumin.itp", "w") as f:
        f.write("\n".join(lines) + "\n")

    # separate atomtypes block (goes in the top-level .top, not the .itp)
    seen_types = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        t = types[idx]
        if t in seen_types:
            continue
        rstar, eps, _, _ = props.GetMMFFVdWParams(idx, idx)
        sigma_nm = (rstar * ANG_TO_NM) / (2 ** (1 / 6))
        eps_kj = eps * KCAL_TO_KJ
        mass = elem_mass[atom.GetSymbol()]
        seen_types[t] = (mass, sigma_nm, eps_kj)

    at_lines = ["[ atomtypes ]",
                "; name    mass    charge  ptype   sigma(nm)   epsilon(kJ/mol)"]
    for t, (mass, sigma, eps) in seen_types.items():
        at_lines.append(f"{t:>10s}  {mass:8.3f}  0.0000  A  {sigma:10.5f}  {eps:10.5f}")
    with open("curcumin_atomtypes.itp", "w") as f:
        f.write("\n".join(at_lines) + "\n")

    print(f"Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}, "
          f"Angles: {len(enumerate_angles(mol))}, Torsions: {len(torsions)}")
    print("Distinct atom types used:", list(seen_types.keys()))
    print("wrote curcumin.itp, curcumin_atomtypes.itp")


if __name__ == "__main__":
    main()
