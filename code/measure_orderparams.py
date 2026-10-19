"""Measure the energy, spin, charge and d-wave correlations of a released state.

    python code/measure_orderparams.py --state tpm02_L16_pfaffian \\
        --out runs/tpm02_L16_pfaffian

The Monte Carlo chains start at the released chains and are thermalized for
--thermal x N steps; sweeps are --sweep-steps x N steps apart (both default to the
paper's values). Every sweep measures the selected channels on --nsamples
configurations. The output folder gets

    series.npz      per-sweep means, rewritten every few sweeps
    results.json    E/N, sigma^2/N, Delta_SC, M^2_str, ... (see measure/assemble.py)
    results.npz     C_p(r), C_s(r), C_c(r), S(k), N(q)
    spins_last.npz  final chains, in the format of weights/<state>_spins.npz

Sweeps with non-finite values are excluded. On several nodes, run one process per node
with --distributed (automatic when SLURM_NTASKS > 1).
"""

import argparse
import os
import time

import numpy as np

CHANNELS = ("energy", "spin", "charge", "dwave")
NREF = 8  # reference sites of the transverse spin correlations
SAVE_EVERY = 4  # sweeps between rewrites of series.npz


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--state", required=True, help="released state, see states.py")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--nsamples", type=int, default=8192, help="samples per sweep")
    p.add_argument("--nsweeps", type=int, default=64, help="number of sweeps")
    p.add_argument("--thermal", type=float, help="thermalization steps per chain / N")
    p.add_argument(
        "--sweep-steps", type=float, help="steps per chain between sweeps / N"
    )
    p.add_argument("--channels", default=",".join(CHANNELS), help="comma-separated")
    p.add_argument("--distributed", action="store_true", help="see distributed.py")
    args = p.parse_args()
    channels = set(args.channels.split(","))
    if not channels <= set(CHANNELS):
        p.error(f"--channels must be a subset of {','.join(CHANNELS)}")
    if args.distributed or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        import distributed

        distributed.initialize()

    import jax
    import jax.numpy as jnp
    import quantax as qtx
    from quantax.sampler import Samples

    import states
    from measure import assemble, observables as obs
    from measure.transverse_spin import reference_operator, transverse_terms

    rank0 = jax.process_index() == 0
    to_np = qtx.utils.to_replicated_numpy

    def log(msg):
        if rank0:
            print(msg, flush=True)

    state, H, info = states.load_state(args.state)
    Lx, Ly = info.shape
    N = Lx * Ly
    nbar = (info.n_up + info.n_dn) / N
    family = obs.paper_family(info.tprime)
    if {"spin", "charge"} & channels:
        obs.stripe_k(Lx, Ly, family)  # fail early if q is not on the momentum grid
    log(f"{info.name}: {Lx}x{Ly}, {jax.device_count()} devices")

    thermal = info.sampler["thermal"] if args.thermal is None else args.thermal
    sweep = info.sampler["sweep"] if args.sweep_steps is None else args.sweep_steps
    thermal_steps = round(thermal * N)
    sampler = states.make_sampler(
        state,
        info,
        args.nsamples,
        sweep_steps=round(sweep * N),
        thermal_steps=thermal_steps,
        seed=42,
    )

    series = dict(
        state=info.name,
        tprime=info.tprime,
        LX=Lx,
        LY=Ly,
        family=family,
        nsamples=args.nsamples,
        thermal_steps=thermal_steps,
    )
    if "spin" in channels:
        refs = obs.reference_sites(Lx, Ly, NREF)
        ref_ops = [reference_operator(a, Lx, Ly) for a in refs]
        series["REF_SITES"] = np.array(refs)
    if "charge" in channels:
        series["nbar"] = nbar
    if "dwave" in channels:
        dw = obs.build_packed_dwave(Lx, Ly)
        series["F_keys"] = obs.F_keys_to_int(dw.F_keys)
        series["G_keys"] = obs.G_keys_to_int(dw.G_keys)

    path = os.path.join(args.out, "series.npz")
    rec, kept, excluded = {}, [], []
    if rank0:
        os.makedirs(args.out, exist_ok=True)

    def save():
        if rank0:
            tmp = os.path.join(args.out, "series_tmp.npz")
            np.savez(
                tmp,
                **series,
                **{k: np.stack(v) for k, v in rec.items()},
                kept_sweeps=np.array(kept, dtype=np.int32),
                excluded_sweeps=np.array(excluded, dtype=np.int32),
            )
            os.replace(tmp, path)

    t0 = time.time()
    for n in range(args.nsweeps):
        s = sampler.sweep().spins
        samples = Samples(s, psi=state(s))  # psi once, reused by every channel
        cur = {}
        if "energy" in channels:
            e = to_np(jnp.real(H.Oloc(state, samples))).astype(np.float64)
            cur["E_mean"], cur["E_sqr"] = e.mean(), (e**2).mean()
        if {"spin", "charge"} & channels:
            gz, gn = obs.sz_grid(s, Lx, Ly), obs.dn_grid(s, Lx, Ly, nbar)
            cur["sz_site"] = to_np(jnp.mean(gz, axis=0))
            cur["n_site"] = to_np(jnp.mean(gn, axis=0)) + nbar
            if "spin" in channels:
                cur["Czz"] = to_np(obs.autocorr(gz))
            if "charge" in channels:
                cur["Cnn"] = to_np(obs.autocorr(gn))
        if "spin" in channels:
            T = np.stack(
                [
                    to_np(transverse_terms(state, samples, op, a, Lx, Ly))
                    for a, op in zip(refs, ref_ops)
                ]
            )
            cur["Tpm"], cur["Tmp"] = T[:, 0], T[:, 1]
        if "dwave" in channels:
            cur["F"], cur["G"] = map(to_np, dw.measure(state, samples))

        if not all(np.isfinite(v).all() for v in cur.values()):
            excluded.append(n)
            log(f"sweep {n}: non-finite values, excluded")
            continue
        kept.append(n)
        for k, v in cur.items():
            rec.setdefault(k, []).append(v)
        msg = f"sweep {n}: {time.time() - t0:.0f} s"
        if "energy" in channels:
            var = cur["E_sqr"] - cur["E_mean"] ** 2
            msg += f", E/N = {cur['E_mean'] / N:.6f}, sigma^2/N = {var / N:.4f}"
        log(msg)
        if (n + 1) % SAVE_EVERY == 0:
            save()

    save()
    spins = to_np(s)
    if rank0:
        bits = np.packbits(spins > 0, axis=-1)
        np.savez(os.path.join(args.out, "spins_last.npz"), bits=bits, shape=spins.shape)
        if kept:
            assemble.main(path)


if __name__ == "__main__":
    main()
