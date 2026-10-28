"""Stage 4: kinematic validation of a trajectory on a simulated Panda."""

import dataclasses

import numpy as np

from egoalign import robot


@dataclasses.dataclass
class Verdict:
    """Validation outcome.

    Attributes:
        valid: True if every frame passed.
        reason: Failure kind ('empty', 'unreachable', 'self_collision',
            'table_collision', 'singularity', 'joint_velocity') or None.
        frame: First failing frame index, or None.
        joints: (N, 7) joint solutions for the frames that passed.
    """

    valid: bool
    reason: str | None
    frame: int | None
    joints: np.ndarray


def _seeds(panda, n, rng):
    """READY followed by n-1 random configurations within joint limits."""
    randoms = [rng.uniform(panda.lo, panda.hi) for _ in range(n - 1)]
    return [robot.READY] + randoms


def _solve_first(panda, pose, n_seeds):
    """Solves the first frame from several seeds; returns (q, converged).

    Prefers a collision-free solution, else any converged one.
    """
    q, fallback = robot.READY, None
    for seed in _seeds(panda, n_seeds, np.random.default_rng(0)):
        q, ok = panda.solve(pose, seed)
        if ok and not panda.collisions():
            return q, True
        if ok and fallback is None:
            fallback = q
    if fallback is None:
        return q, False
    panda.forward(fallback)
    return fallback, True


def _check_state(panda, min_sigma):
    """Returns a failure reason for the robot's current state, or None."""
    hits = panda.collisions()
    if 'self' in hits:
        return 'self_collision'
    if 'table' in hits:
        return 'table_collision'
    if panda.min_singular_value() < min_sigma:
        return 'singularity'
    return None


def validate(traj, min_sigma=0.02, max_joint_speed=2.0, n_seeds=8):
    """Replays a trajectory through IK and flags the first infeasible frame.

    Args:
        traj: `taskspace.Trajectory` in the robot base frame.
        min_sigma: Minimum Jacobian singular value before calling it singular.
        max_joint_speed: Joint speed limit [rad/s].
        n_seeds: IK restarts for the first frame.

    Returns:
        A `Verdict`.
    """
    if not len(traj):
        return Verdict(False, 'empty', 0, np.zeros((0, 7)))
    panda = robot.Panda()
    max_step = max_joint_speed / traj.fps
    joints = []
    for i, pose in enumerate(traj.poses):
        if i == 0:
            q, ok = _solve_first(panda, pose, n_seeds)
        else:
            q, ok = panda.solve(pose, joints[-1], iters=30)
        reason = 'unreachable' if not ok else _check_state(panda, min_sigma)
        if reason is None and i and np.abs(q - joints[-1]).max() > max_step:
            reason = 'joint_velocity'
        if reason:
            return Verdict(False, reason, i, np.array(joints).reshape(-1, 7))
        joints.append(q)
    return Verdict(True, None, None, np.array(joints))
