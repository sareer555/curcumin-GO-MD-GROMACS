"""
assemble_complex.py

Places curcumin above the (much bigger) GO flake using the REAL relative
geometry from the Project 1 DFT-converged complex (curcumin_on_go_FINAL.xyz)
-- not a guess. Concretely:

  1. Take curcumin's own conformation AS IT SITS IN the DFT complex (not the
     standalone-optimized conformation) -- this is the geometry that's
     actually adopted when adsorbed, slightly relaxed from the free molecule.
  2. Compute the small DFT GO cluster's surface-normal direction and the
     stacking distance from that cluster's plane to curcumin's centroid.
  3. Compute the new, much bigger GO flake's own surface-normal direction
     and center.
  4. Rotate curcumin's DFT-complex conformation by whatever rotation maps
     the small cluster's normal onto the big flake's normal (preserving the
     real stacking orientation), then place it at the flake's center,
     offset along the flake's normal by the same stacking distance found
     in the DFT structure.

This keeps the starting pose physically motivated by an actual DFT result,
rather than an arbitrary "put it somewhere above the surface" guess.
"""
import numpy as np
from ase.io import read, write


def fit_plane_normal(positions):
    center = positions.mean(axis=0)
    centered = positions - center
    _, _, vt = np.linalg.svd(centered)
    normal = vt[2]  # smallest-variance direction = plane normal
    return center, normal / np.linalg.norm(normal)


def rotation_aligning(a, b):
    """Rotation matrix mapping unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def main():
    complex_atoms = read("/tmp/repo/outputs/optimization/curcumin_on_go_FINAL.xyz")
    go_ref = complex_atoms[:28].get_positions()
    curc_ref = complex_atoms[28:].get_positions()
    curc_symbols = complex_atoms[28:].get_chemical_symbols()

    go_ref_center, go_ref_normal = fit_plane_normal(go_ref)
    curc_ref_center = curc_ref.mean(axis=0)
    offset = curc_ref_center - go_ref_center
    standoff = np.dot(offset, go_ref_normal)
    if standoff < 0:  # normal direction is arbitrary sign from SVD; point it at curcumin
        go_ref_normal = -go_ref_normal
        standoff = -standoff
    print(f"DFT reference: GO-plane-to-curcumin standoff = {standoff:.3f} Angstrom "
          f"(expected ~3.0-3.6 A for pi-stacking)")

    flake = read("go_flake.xyz")
    flake_pos = flake.get_positions()
    flake_symbols = flake.get_chemical_symbols()
    # fit the plane using only basal (sp2, non-functionalized) carbons for a
    # clean surface normal, unbiased by out-of-plane epoxide/hydroxyl groups
    is_carbon = np.array([s == "C" for s in flake_symbols])
    flake_center_full = flake_pos.mean(axis=0)
    z_extent = flake_pos[:, 2] - flake_center_full[2]
    basal_mask = is_carbon & (np.abs(z_extent) < 0.3)  # near-planar carbons
    basal_pts = flake_pos[basal_mask]
    flake_center, flake_normal = fit_plane_normal(basal_pts)
    # orient flake_normal towards the same side curcumin will be placed (+z of the built lattice)
    if flake_normal[2] < 0:
        flake_normal = -flake_normal
    print(f"Flake surface normal: {flake_normal}, using {basal_mask.sum()} basal carbons")

    R = rotation_aligning(go_ref_normal, flake_normal)

    curc_centered = curc_ref - curc_ref_center
    curc_rotated = curc_centered @ R.T

    # Place laterally at the flake's own center (in-plane), then set the
    # normal-direction height by CLEARANCE from the flake's actual surface
    # (including epoxide/hydroxyl protrusions), not by re-using the small
    # DFT cluster's plane-to-centroid distance -- that reference plane isn't
    # directly comparable to the new flake's own coordinate frame, and
    # trusting it literally put curcumin's rotated pose overlapping the
    # flake's own oxidized bumps (checked: several sub-1-Angstrom contacts).
    # Target clearance matches the DFT standoff (~3.6 A), measured from
    # whichever flake atom curcumin would otherwise sit closest to.
    target_clearance = standoff  # from the real DFT geometry, ~3.6 A
    # start with curcumin's centroid sitting exactly at the flake's own
    # plane center (zero clearance), then push it up along the normal
    curc_trial = curc_rotated - curc_rotated.mean(axis=0) + flake_center

    from scipy.spatial import cKDTree as _KDT
    tree_flake = _KDT(flake_pos)
    # binary search the normal-direction offset that gives the target clearance
    lo, hi = 0.0, 15.0
    for _ in range(40):
        mid = (lo + hi) / 2
        trial_pos = curc_trial + flake_normal * mid
        d, _ = tree_flake.query(trial_pos, k=1)
        if d.min() < target_clearance:
            lo = mid
        else:
            hi = mid
    curc_final = curc_trial + flake_normal * hi
    print(f"Solved normal-direction offset: {hi:.3f} A above flake-center reference "
          f"to achieve {target_clearance:.2f} A clearance")

    # write combined structure: flake atoms first (matches go_flake.itp
    # ordering), then curcumin (matches curcumin.itp ordering)
    from ase import Atoms
    combined = Atoms(
        symbols=flake_symbols + curc_symbols,
        positions=np.vstack([flake_pos, curc_final]),
    )
    combined.write("complex_raw.xyz")
    print(f"wrote complex_raw.xyz ({len(combined)} atoms: "
          f"{len(flake_symbols)} GO flake + {len(curc_symbols)} curcumin)")

    # quick clash check between the two fragments
    from scipy.spatial import cKDTree
    tree = cKDTree(flake_pos)
    dists, _ = tree.query(curc_final, k=1)
    print(f"Closest curcumin-to-flake-atom contact: {dists.min():.3f} A "
          f"(should be > ~2.5 A, no clash)")


if __name__ == "__main__":
    main()
