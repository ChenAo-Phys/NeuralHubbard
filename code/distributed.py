"""Multi-process JAX with all GPUs of each process.

Under SLURM, ``jax.distributed.initialize()`` gives each process only GPU ``SLURM_LOCALID``.
`initialize` lets each process use all GPUs it sees, or splits them evenly between the
tasks of a node if every task sees all GPUs of the node (``srun --gpu-bind=none``).
Prefer one process per node: GPUs isolated per task can break NCCL within a node.
"""

import os
import re

import jax


def _tasks_per_node():
    # task count of this step on the first node, e.g. "2(x3),1"
    for key in ("SLURM_STEP_TASKS_PER_NODE", "SLURM_TASKS_PER_NODE"):
        if m := re.match(r"\d+", os.environ.get(key, "")):
            return int(m[0])
    return 1


def local_device_ids():
    """GPU indices of this process in its CUDA_VISIBLE_DEVICES, None outside SLURM."""
    if "SLURM_LOCALID" not in os.environ:
        return None
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    n_node = int(os.environ.get("SLURM_GPUS_ON_NODE", len(visible)))
    n = len(visible) or n_node
    if n == 0:
        return None
    tasks = _tasks_per_node()
    if tasks > 1 and n == n_node:  # every task sees all GPUs of the node
        i = int(os.environ["SLURM_LOCALID"])
        return list(range(i * n // tasks, (i + 1) * n // tasks))
    return list(range(n))


def initialize():
    jax.distributed.initialize(local_device_ids=local_device_ids())
