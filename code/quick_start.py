"""Load a released state and evaluate its amplitudes and a small-sample energy.

    python code/quick_start.py tpm02_L16_pfaffian --nconfigs 16

Prints the projected log|psi| and sign of the first stored Monte Carlo configurations and
their mean local energy next to the paper's value. Runs best on a GPU.
"""

import argparse

import numpy as np

import states

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("state", nargs="?", default="tpm02_L16_pfaffian", choices=states.STATES)
p.add_argument("--nconfigs", type=int, default=16, help="configurations to evaluate")
args = p.parse_args()

import quantax as qtx
from quantax.sampler import Samples

state, H, info = states.load_state(args.state)
print(f"{info.name}: {state.nparams} parameters")

spins = qtx.utils.to_distributed_array(states.load_spins(info.name)[: args.nconfigs])
psi = state(spins)
logabs = qtx.utils.to_replicated_numpy(psi.logabs)
sign = qtx.utils.to_replicated_numpy(psi.sign)
for i, (la, sg) in enumerate(zip(logabs, sign)):
    print(f"{i:3d}  log|psi| = {la:12.4f}  sign = {sg:+.0f}")

eloc = qtx.utils.to_replicated_numpy(H.Oloc(state, Samples(spins, psi=psi)))
eloc = np.real(eloc).astype(np.float64)
E, err = eloc.mean() / info.N, eloc.std(ddof=1) / np.sqrt(len(eloc)) / info.N
print(f"E/N = {E:.5f} +- {err:.5f} from {len(eloc)} configurations")
print(f"paper: E/N = {info.paper['E_per_site']} +- {info.paper['E_err']}")
