import re

with open('go_flake_atomtypes.itp') as f:
    go_at = f.read()
with open('curcumin_atomtypes.itp') as f:
    curc_at = f.read()

# strip the "[ atomtypes ]" header lines from the second block, keep only data rows
def strip_header(text):
    lines = text.splitlines()
    out = []
    started = False
    for line in lines:
        if line.strip().startswith('[ atomtypes ]'):
            started = True
            continue
        if started and line.strip().startswith(';'):
            continue
        if line.strip():
            out.append(line)
    return out

go_rows = strip_header(go_at)
curc_rows = strip_header(curc_at)

manual_extra = [
    "          OW    15.99940  0.0000  A   0.315061   0.636386",
    "          HW     1.00800  0.0000  A   0.000000   0.000000",
]

with open('go_flake.itp') as f:
    go_itp = f.read()
with open('curcumin.itp') as f:
    curc_itp = f.read()
with open('/usr/share/gromacs/top/amber99sb-ildn.ff/tip3p.itp') as f:
    tip3p = f.read()

parts = []
parts.append("[ defaults ]\n; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n1         2          yes        0.5      0.8333\n")
parts.append("[ atomtypes ]\n; name    mass    charge  ptype   sigma(nm)   epsilon(kJ/mol)")
parts.extend(go_rows)
parts.extend(curc_rows)
parts.extend(manual_extra)
parts.append("")

# insert posres include right before "[ moleculetype ]" of the NEXT molecule
def insert_posres(itp_text, posre_file):
    idx = itp_text.rfind("[ moleculetype ]")
    # go_itp/curc_itp each contain exactly one moleculetype block, so no "next"
    # moleculetype inside them -- append posres block at the end of the text instead
    return itp_text.rstrip() + f"\n\n#ifdef POSRES\n#include \"{posre_file}\"\n#endif\n"

go_block = insert_posres(go_itp, "posre_goflake.itp")
curc_block = insert_posres(curc_itp, "posre_curc.itp")

parts.append(go_block)
parts.append(curc_block)
parts.append(tip3p.strip())
parts.append("")
parts.append("[ system ]\nCurcumin on GO nanoflake, explicit water\n")
parts.append("[ molecules ]\n; name       nmols\nGOFLAKE      1\nCURC         1\nSOL              4442\n")

with open('complex.top', 'w') as f:
    f.write("\n".join(parts))

print("wrote complex.top")
