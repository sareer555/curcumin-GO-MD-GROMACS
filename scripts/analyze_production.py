"""
analyze_production.py

Reproduces the production-MD analysis figure (figures/production_analysis.png)
and the headline adsorption-stability numbers quoted in the README, directly
from the GROMACS analysis output files in data/ (all produced by the `gmx`
commands documented in the README's "Reproducing this" section -- this
script only does the plotting/statistics on top of that raw output, it does
not re-run GROMACS itself).

Run from the repo root:
    python3 scripts/analyze_production.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_xvg(path, ncols):
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= ncols:
                data.append([float(x) for x in parts[:ncols]])
    return np.array(data)


def main():
    curc = load_xvg("data/curc_com.xvg", 4)
    go = load_xvg("data/go_com.xvg", 4)
    rmsd = load_xvg("data/curc_rmsd.xvg", 2)
    gyrate = load_xvg("data/curc_gyrate.xvg", 2)
    mindist = load_xvg("data/curc_go_mindist.xvg", 2)

    t = curc[:, 0]
    com_dist = np.linalg.norm(curc[:, 1:4] - go[:, 1:4], axis=1)

    print(f"N frames: {len(t)}, time range {t.min():.1f}-{t.max():.1f} ps")
    d = mindist[:, 1]
    print(f"Closest-atom curcumin-GO distance (nm): start={d[0]:.3f}, end={d[-1]:.3f}, "
          f"mean={d.mean():.3f}, std={d.std():.3f}")
    print(f"  first 10% mean: {d[:len(d)//10].mean():.3f}, last 10% mean: {d[-len(d)//10:].mean():.3f}")
    print(f"COM-COM distance (nm): start={com_dist[0]:.3f}, end={com_dist[-1]:.3f}, mean={com_dist.mean():.3f}")
    print(f"Curcumin heavy-atom RMSD (nm, vs t=0): mean={rmsd[:,1].mean():.3f}, final={rmsd[-1,1]:.3f}")
    print(f"Curcumin Rg (nm): start={gyrate[0,1]:.3f}, end={gyrate[-1,1]:.3f}, std={gyrate[:,1].std():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(mindist[:, 0], mindist[:, 1], color="#2b6cb0", lw=1.2)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Closest atom-atom distance (nm)")
    ax.set_title("Curcumin <-> GO flake contact distance")
    ax.axhline(mindist[:, 1].mean(), color="gray", ls="--", lw=0.8,
               label=f"mean = {mindist[:,1].mean():.2f} nm")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, com_dist, color="#c05621", lw=1.2)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Center-of-mass distance (nm)")
    ax.set_title("Curcumin <-> GO flake COM distance")

    ax = axes[1, 0]
    ax.plot(rmsd[:, 0], rmsd[:, 1] * 10, color="#2f855a", lw=1.2)  # nm -> A
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Curcumin heavy-atom RMSD (angstrom, vs t=0)")
    ax.set_title("Curcumin internal conformational stability")

    ax = axes[1, 1]
    ax.plot(gyrate[:, 0], gyrate[:, 1] * 10, color="#805ad5", lw=1.2)  # nm -> A
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Radius of gyration (angstrom)")
    ax.set_title("Curcumin shape (Rg) over time")

    fig.suptitle("Production MD (500 ps, 300 K, 1 atm): curcumin on GO nanoflake", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figures/production_analysis.png", dpi=150)
    print("saved figures/production_analysis.png")


if __name__ == "__main__":
    main()
