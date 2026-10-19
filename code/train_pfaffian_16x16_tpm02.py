r"""Train the 16x16 Pfaffian backflow (U = 8, t' = -0.2, 1/8 hole doping).

The paper's recipe in two stages: 5000 steps without projection from a cold start, then
1000 steps projected onto Flip(0) @ Flip(1) @ SpinInverse with fresh optimizer buffers:

    python code/train_pfaffian_16x16_tpm02.py --outdir runs/stage1
    python code/train_pfaffian_16x16_tpm02.py --outdir runs/stage2 --symm \
        --init runs/stage1/params_4999.eqx

Rerunning a command resumes from the newest checkpoint in --outdir. Needs a GPU.
"""

import argparse
import functools
import os
import re
import time

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--outdir", default="runs/pfaffian_16x16_tpm02")
parser.add_argument("--nsamples", type=int, default=1024, help="paper: 8192")
parser.add_argument("--steps", type=int, help="default 5000, or 1000 with --symm")
parser.add_argument("--symm", action="store_true", help="symmetry projection")
parser.add_argument("--init", type=os.path.abspath, help="initial parameters (.eqx)")
parser.add_argument("--init-spins", type=os.path.abspath, help=".npy or _spins.npz")
parser.add_argument("--c0", type=float, default=1.0, help="step-norm cap C0")
parser.add_argument("--distributed", action="store_true", help="see distributed.py")
args = parser.parse_args()

if args.distributed:
    import distributed

    distributed.initialize()

import jax
import jax.numpy as jnp
import numpy as np
import quantax as qtx
from quantax.symmetry import Flip, SpinInverse

from backflow import build_backflow

L, N, NELEC = 16, 256, (112, 112)  # 1/8 hole doping
SWEEP, THERMAL = 3 * N, 20 * N  # Metropolis proposals per step and for thermalisation
T0, T_DECAY = 1000, 4000  # step cap c(t) = C0 / (1 + max(t - T0, 0) / T_DECAY)
ASHIFT, CLIP = 1e-3, 5.0  # SR diagonal shift, local-energy clip in standard deviations
SEED, CKPT_EVERY = 42, 50

NSTEPS = args.steps if args.steps is not None else (1000 if args.symm else 5000)
if args.nsamples % (2 * jax.device_count()):
    raise ValueError("--nsamples must be a multiple of twice the number of devices")
per_dev = args.nsamples // jax.device_count()
MAX_PARALLEL = (per_dev, min(256 if args.symm else 512, per_dev))  # forward, Jacobian
RANK0 = jax.process_index() == 0
log = functools.partial(print, flush=True) if RANK0 else lambda msg: None
os.makedirs(args.outdir, exist_ok=True)
os.chdir(args.outdir)


def saved(pattern):
    """Steps i of the files in the run directory that match pattern."""
    return {int(m[1]) for f in os.listdir() if (m := re.fullmatch(pattern, f))}


def load_spins(path):
    """int8 (nsamples, 2N) chains from .npy or _spins.npz, repeated if fewer."""
    if path.endswith(".npz"):
        z = np.load(path)
        spins = np.where(np.unpackbits(z["bits"], axis=-1, count=z["shape"][1]), 1, -1)
    else:
        spins = np.load(path)
    assert spins.shape[1] == 2 * N, path
    return np.resize(spins.astype(np.int8), (args.nsamples, 2 * N))


qtx.set_random_seed(SEED)
qtx.sites.Square(L, particle_type=qtx.PARTICLE_TYPE.spinful_fermion, Nparticles=NELEC)
H = qtx.operator.Hubbard(U=8.0, t=[1.0, -0.2], n_neighbor=[1, 2])

start, init, opt_file, thermal = 0, args.init, None, THERMAL
init_spins = args.init_spins
if ckpts := saved(r"params_(\d+)\.eqx") & saved(r"optimizer_(\d+)\.eqx"):
    ckpt = max(ckpts)
    start, init, opt_file = ckpt + 1, f"params_{ckpt}.eqx", f"optimizer_{ckpt}.eqx"
    js = [j for j in saved(r"spins_(\d+)\.npy") if j <= ckpt]
    init_spins, thermal = (f"spins_{max(js)}.npy", N) if js else (None, THERMAL)
log(f"step {start}: params {init or 'random'}, chains {init_spins or 'random'}")

model = build_backflow("pfaffian", L)
symm = Flip(axis=0) @ Flip(axis=1) @ SpinInverse() if args.symm else None
state = qtx.state.Variational(model, init, symm=symm, max_parallel=MAX_PARALLEL)
spins = None if init_spins is None else jnp.asarray(load_spins(init_spins))
samplers = [
    S(state, args.nsamples // 2, thermal_steps=0, n_neighbor=(1, 2))
    for S in (qtx.sampler.ParticleHop, qtx.sampler.SiteExchange)
]
sampler = qtx.sampler.MixSampler(
    samplers, thermal_steps=thermal, sweep_steps=SWEEP, initial_spins=spins
)
solver = qtx.optimizer.minnorm_shift_eig(ashift=ASHIFT)
opt = qtx.optimizer.AdamSR(state, H, solver=solver, file=opt_file, clip=CLIP)

names = ("energy", "VarE", "step_norm")
traces = {k: list(np.load(f"{k}.npy")[:start]) if start else [] for k in names}
n_bad = 0
for i in range(start, NSTEPS):
    t = time.perf_counter()
    samples = sampler.sweep()
    step = opt.get_step(samples)
    cap = args.c0 / (1 + max(i - T0, 0) / T_DECAY)  # learning rate 1, capped step
    norm = jnp.linalg.norm(step)
    if jnp.isfinite(norm):
        state.update(step * jnp.minimum(1.0, cap / norm))
        n_bad = 0
    else:  # skip the update, AdamSR keeps its buffers
        n_bad += 1
        log(f"step {i}: non-finite step, update skipped")
        if n_bad == 5:
            raise RuntimeError("5 consecutive non-finite steps")
    for k, v in zip(traces, (opt.energy, opt.VarE, norm)):
        traces[k].append(float(v))
    if i < start + 10 or i % 10 == 0 or i == NSTEPS - 1:
        E, VarE, nrm = (v[-1] for v in traces.values())
        dt = time.perf_counter() - t
        log(f"step {i}: E {E:.4f}, VarE {VarE:.3f}, |step| {nrm:.3f}, {dt:.1f} s/step")
    if i % CKPT_EVERY == 0 or i == NSTEPS - 1:
        spins = qtx.utils.to_replicated_numpy(samples.spins)  # chains before the update
        if RANK0:
            np.save(f"spins_{i}.npy", spins)
            for k, v in traces.items():
                np.save(f"{k}.npy", np.asarray(v))
        state.save(f"params_{i}.eqx")
        opt.save(f"optimizer_{i}.eqx")
