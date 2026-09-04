"""
build_go_flake.py

Builds a finite graphene-oxide nanoflake for the Project 2 MD system:
  - hexagonal graphene lattice, cut to a roughly rectangular flake
  - edge carbons H-terminated
  - basal-plane carbons randomly decorated with epoxide (bridging O) and
    hydroxyl (C-OH) groups at a target O:C ratio (~1:8, matching Wu et al.
    2022, Molecules 27, 6742)

Output: go_flake.xyz (plain XYZ, ready for a quick MM relaxation / for
antechamber to read as a mol2 after conversion).
"""
import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.neighborlist import neighbor_list

RNG = np.random.default_rng(42)

TARGET_O_TO_C = 1 / 8.0   # literature oxidation ratio
CC_BOND = 1.42
CH_BOND = 1.09
COH_BOND = 1.43   # C-O for hydroxyl
OH_BOND = 0.96
EPOXIDE_HEIGHT = 1.25  # O sits above the C-C bond midpoint


def build_sheet(nx=16, ny=14):
    """Cut a roughly nx*ny-hexagon rectangular graphene flake."""
    unit = graphene(formula="C2", a=CC_BOND * np.sqrt(3), size=(nx, ny, 1), vacuum=15.0)
    unit.set_pbc(False)  # treat as a finite flake, not a periodic sheet
    return unit


def trim_dangling_carbons(atoms, max_passes=5):
    """Remove carbons with 0 or 1 carbon-carbon neighbors (corner/dangling
    atoms from cutting a rectangle out of the lattice). A single H can only
    properly terminate a 2-coordinate edge carbon (bisecting its two C-C
    bonds gives a sensible ~120 degree outward direction); a 1-coordinate
    carbon's "outward direction" collapses onto the existing bond, which
    produces a degenerate ~180 degree H-C-C angle -- exactly the pathology
    that showed up as unresolvable steric strain during equilibration.
    Trimming these atoms (and re-checking, since removing one can expose a
    new dangling neighbor) gives a smaller but structurally clean flake."""
    cutoff = CC_BOND * 1.2
    for _ in range(max_passes):
        i_list, j_list = neighbor_list("ij", atoms, cutoff)
        coord = np.bincount(i_list, minlength=len(atoms))
        dangling = np.where(coord < 2)[0]
        if len(dangling) == 0:
            break
        keep = [i for i in range(len(atoms)) if i not in set(dangling.tolist())]
        atoms = atoms[keep]
    return atoms


def h_terminate_edges(atoms):
    """Find 2-coordinate (edge) carbons and cap each with one H along the
    outward-pointing direction (opposite the sum of its bonds to neighbors)."""
    cutoff = CC_BOND * 1.2
    i_list, j_list = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    pos = atoms.get_positions()
    new_positions = []
    for idx in range(len(atoms)):
        if coord[idx] < 3:
            neighbors = j_list[i_list == idx]
            if len(neighbors) == 0:
                continue
            bond_vecs = pos[neighbors] - pos[idx]
            outward = -bond_vecs.sum(axis=0)
            norm = np.linalg.norm(outward)
            if norm < 1e-6:
                continue
            outward /= norm
            h_pos = pos[idx] + outward * CH_BOND
            new_positions.append(h_pos)
    return new_positions


def decorate_basal_plane(atoms, n_h_edge):
    """Add epoxide + hydroxyl groups to interior (3-coordinate) carbons to
    hit the target O:C ratio. Returns list of (element, position)."""
    cutoff = CC_BOND * 1.2
    i_list, j_list = neighbor_list("ij", atoms, cutoff)
    coord = np.bincount(i_list, minlength=len(atoms))
    interior = np.where(coord == 3)[0]
    pos = atoms.get_positions()
    z0 = pos[:, 2].mean()

    n_carbons = len(atoms)
    n_o_target = int(round(n_carbons * TARGET_O_TO_C))

    # Build bond list restricted to interior atoms for epoxide placement
    interior_set = set(interior.tolist())
    interior_bonds = [(i, j) for i, j in zip(i_list, j_list)
                       if i < j and i in interior_set and j in interior_set]
    RNG.shuffle(interior_bonds)

    # adjacency map (interior atoms only) so we can block not just a used
    # carbon but its direct neighbors too -- keeps functional groups from
    # landing right next to each other, which caused OH/OH steric clashes
    adjacency = {}
    for i, j in zip(i_list, j_list):
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    used = set()      # carbons that already carry a functional group
    blocked = set()   # used + their direct neighbors -- off-limits for a NEW group
    extra_atoms = []
    o_placed = 0
    bond_iter = iter(interior_bonds)
    interior_list = list(interior)
    RNG.shuffle(interior_list)
    site_iter = iter(interior_list)

    def block(idx):
        used.add(idx)
        blocked.add(idx)
        blocked.update(adjacency.get(idx, ()))

    # Alternate epoxide (uses 2 carbons) / hydroxyl (uses 1 carbon) placements
    toggle = 0
    while o_placed < n_o_target:
        if toggle == 0:
            try:
                i, j = next(bond_iter)
            except StopIteration:
                break
            if i in blocked or j in blocked:
                continue
            mid = (pos[i] + pos[j]) / 2
            sign = 1 if mid[2] >= z0 else -1
            o_pos = mid + np.array([0, 0, sign * EPOXIDE_HEIGHT])
            extra_atoms.append(("O", o_pos))
            block(i)
            block(j)
            o_placed += 1
        else:
            try:
                i = next(site_iter)
            except StopIteration:
                break
            if i in blocked:
                continue
            sign = 1 if pos[i][2] >= z0 else -1

            # Puckered (tetrahedral-like) C-OH placement, not a pure
            # perpendicular one. A purely perpendicular C-O bond (straight
            # up the local normal) makes an exactly-90 degree C-C-O angle
            # to every one of the carbon's 3 ring neighbors -- geometrically
            # clean but physically wrong for a real sp3-hydroxylated carbon
            # (should be closer to the tetrahedral ~109.5 degrees), and that
            # exact 90 degree degeneracy is what let this bond survive
            # static energy minimization while still being badly strained
            # under real 300 K dynamics (diagnosed root cause of the NVT
            # LINCS failures). Fix: tilt the C-O bond off the local normal,
            # away from one specific ring neighbor, by the difference
            # between the tetrahedral angle and 90 degrees, so the C-O bond
            # makes ~109.5 degrees to that neighbor instead of exactly 90.
            nbrs = list(adjacency.get(i, ()))
            normal_vec = np.array([0.0, 0.0, float(sign)])
            if nbrs:
                ref_bond = pos[nbrs[0]] - pos[i]
                ref_bond[2] = 0.0  # project into the local basal plane
                ref_norm = np.linalg.norm(ref_bond)
                if ref_norm > 1e-6:
                    v_in = ref_bond / ref_norm
                    tilt = np.radians(109.47 - 90.0)  # tetrahedral - perpendicular
                    dir_vec = np.cos(tilt) * normal_vec - np.sin(tilt) * v_in
                    dir_vec /= np.linalg.norm(dir_vec)
                else:
                    dir_vec = normal_vec
            else:
                dir_vec = normal_vec

            o_pos = pos[i] + dir_vec * COH_BOND

            # O-H direction: this MUST bend away from the C-O direction by
            # roughly the tetrahedral angle (~104-109 degrees is the real
            # C-O-H angle in an alcohol) -- continuing nearly straight out
            # from dir_vec (as an earlier version of this function did, by
            # just adding a small wobble to dir_vec itself) gives a
            # ~170 degree C-O-H angle, i.e. nearly LINEAR, which is even
            # more unphysical than the original 90-degree C-C-O problem
            # this function was written to fix, and was in fact the real
            # cause of the LINCS failures that persisted after the C-C-O
            # angle fix above (confirmed: the failing O-H bonds' own C-O-H
            # angle was ~171 degrees in that version). Build the O-H bond
            # by rotating the C-O direction by the C-O-H angle around an
            # arbitrary perpendicular axis, then randomizing the rotation
            # around the C-O bond itself (free rotation, like a real
            # hydroxyl group) via a second rotation about dir_vec.
            arbitrary = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(arbitrary, dir_vec)) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0])
            perp = np.cross(dir_vec, arbitrary)
            perp /= np.linalg.norm(perp)

            # NOTE: the angle we actually want to fix is C-O-H at vertex O,
            # between vectors O->C (= -dir_vec) and O->H. Building oh_dir as
            # an angle measured from dir_vec (C->O, i.e. O->C reversed) means
            # the angle from dir_vec must be (180 - target C-O-H angle), or
            # the H swings back toward the carbon instead of pointing
            # outward (this was tried first and put H only 1.49 A from its
            # own parent carbon -- close enough to register as a spurious
            # C-H "bond" in the topology's distance-based classifier).
            coh_angle = np.radians(106.0)  # typical alcohol C-O-H angle
            angle_from_dirvec = np.pi - coh_angle
            bent = np.cos(angle_from_dirvec) * dir_vec + np.sin(angle_from_dirvec) * perp

            # randomize rotation around the C-O bond axis (dihedral freedom)
            azimuth = RNG.uniform(0, 2 * np.pi)
            perp2 = np.cross(dir_vec, perp)
            oh_dir = (np.cos(azimuth) * (bent - np.cos(angle_from_dirvec) * dir_vec)
                      + np.sin(azimuth) * np.sin(angle_from_dirvec) * perp2
                      + np.cos(angle_from_dirvec) * dir_vec)
            oh_dir /= np.linalg.norm(oh_dir)
            h_pos = o_pos + oh_dir * OH_BOND
            extra_atoms.append(("O", o_pos))
            extra_atoms.append(("H", h_pos))
            block(i)
            o_placed += 1
        toggle = 1 - toggle

    return extra_atoms, o_placed, n_carbons


def main():
    sheet = build_sheet()
    sheet = trim_dangling_carbons(sheet)
    edge_h_positions = h_terminate_edges(sheet)
    extra_atoms, o_placed, n_carbons = decorate_basal_plane(sheet, len(edge_h_positions))

    symbols = list(sheet.get_chemical_symbols())
    positions = list(sheet.get_positions())

    for p in edge_h_positions:
        symbols.append("H")
        positions.append(p)

    for elem, p in extra_atoms:
        symbols.append(elem)
        positions.append(p)

    flake = Atoms(symbols=symbols, positions=positions)
    flake.center(vacuum=15.0)

    n_h_edge = len(edge_h_positions)
    n_epoxide_o = sum(1 for e, _ in extra_atoms if e == "O") - sum(1 for e, _ in extra_atoms if e == "H")
    print(f"Carbons: {n_carbons}")
    print(f"Edge H: {n_h_edge}")
    print(f"O groups placed: {o_placed}  (target O:C = 1:{1/TARGET_O_TO_C:.0f})")
    print(f"Actual O:C = 1:{n_carbons/o_placed:.1f}")
    print(f"Total atoms: {len(flake)}")

    flake.write("go_flake.xyz")
    print("wrote go_flake.xyz")


if __name__ == "__main__":
    main()
