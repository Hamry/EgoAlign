"""Franka Panda kinematic model in MuJoCo with damped least-squares IK."""

import mujoco
import numpy as np

from egoalign import taskspace

# Capsule Panda from Franka's modified DH; table top at z=0.
PANDA_XML = """
<mujoco model="panda">
  <compiler angle="radian"/>
  <default>
    <joint type="hinge" axis="0 0 1" limited="true"/>
    <geom type="capsule" size="0.05" rgba="0.8 0.8 0.8 1"/>
  </default>
  <worldbody>
    <geom name="table" type="box" pos="0.4 0 -0.02" size="0.8 0.8 0.02"/>
    <body name="link0">
      <geom fromto="0 0 0.02 0 0 0.333" size="0.06"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="j1" range="-2.8973 2.8973"/>
        <geom type="sphere" size="0.06"/>
        <body name="link2" quat="1 -1 0 0">
          <joint name="j2" range="-1.7628 1.7628"/>
          <geom fromto="0 0 0 0 -0.316 0"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="j3" range="-2.8973 2.8973"/>
            <geom fromto="0 0 0 0.0825 0 0"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <joint name="j4" range="-3.0718 -0.0698"/>
              <geom fromto="0 0 0 -0.0825 0.28 0"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <joint name="j5" range="-2.8973 2.8973"/>
                <geom type="sphere" size="0.05"/>
                <body name="link6" quat="1 1 0 0">
                  <joint name="j6" range="-0.0175 3.7525"/>
                  <geom fromto="0 0 0 0.088 0 0" size="0.045"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <joint name="j7" range="-2.8973 2.8973"/>
                    <geom fromto="0 0 0 0 0 0.107" size="0.04"/>
                    <geom type="box" pos="0 0 0.135" size="0.02 0.1 0.03"/>
                    <geom fromto="0 0.04 0.165 0 0.02 0.19" size="0.01"/>
                    <geom fromto="0 -0.04 0.165 0 -0.02 0.19" size="0.01"/>
                    <site name="tcp" pos="0 0 0.2104"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="link0" body2="link1"/>
    <exclude body1="link0" body2="link2"/>
    <exclude body1="link1" body2="link3"/>
    <exclude body1="link2" body2="link4"/>
    <exclude body1="link3" body2="link5"/>
    <exclude body1="link4" body2="link6"/>
    <exclude body1="link5" body2="link7"/>
  </contact>
</mujoco>
"""

def _converged(err, pos_tol, rot_tol):
    return bool(np.linalg.norm(err[:3]) < pos_tol
                and np.linalg.norm(err[3:]) < rot_tol)


READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


class Panda:
    """Kinematics, IK and collision queries for a single robot instance."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(PANDA_XML)
        self.data = mujoco.MjData(self.model)
        self.lo, self.hi = self.model.jnt_range.T
        self._tcp = self.model.site('tcp').id
        self._table = self.model.geom('table').id

    def forward(self, q):
        """Sets joints and runs kinematics plus collision detection."""
        self.data.qpos[:] = q
        mujoco.mj_fwdPosition(self.model, self.data)

    def tcp_pose(self):
        """Returns (position, rotation) of the tool centre point."""
        return (self.data.site_xpos[self._tcp].copy(),
                self.data.site_xmat[self._tcp].reshape(3, 3).copy())

    def jacobian(self):
        """Returns the (6, 7) world-frame TCP Jacobian at the current state."""
        jacp, jacr = np.zeros((3, 7)), np.zeros((3, 7))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._tcp)
        return np.vstack([jacp, jacr])

    def pose_error(self, pos, rot):
        """Returns (6,) world-frame [position, rotvec] error to target."""
        cur_pos, cur_rot = self.tcp_pose()
        q_tgt, q_cur, err = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(q_tgt, rot.ravel())
        mujoco.mju_mat2Quat(q_cur, cur_rot.ravel())
        mujoco.mju_subQuat(err, q_tgt, q_cur)
        return np.concatenate([pos - cur_pos, cur_rot @ err])

    def solve(self, pose, seed, iters=100, pos_tol=0.005, rot_tol=0.05,
              damping=0.05):
        """Damped least-squares IK for a 6D pose, respecting joint limits.

        Args:
            pose: (6,) target x, y, z, roll, pitch, yaw.
            seed: (7,) initial joint configuration.
            iters: Maximum iterations.
            pos_tol: Position tolerance [m].
            rot_tol: Rotation tolerance [rad].
            damping: DLS damping factor.

        Returns:
            (q, converged); the robot state is left at q.
        """
        pos, rot = pose[:3], taskspace.euler_to_matrix(pose[3:])
        q = np.clip(seed, self.lo, self.hi)
        for _ in range(iters):
            self.forward(q)
            err = self.pose_error(pos, rot)
            if _converged(err, pos_tol, rot_tol):
                return q, True
            jac = self.jacobian()
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6),
                                         err)
            q = np.clip(q + dq, self.lo, self.hi)
        self.forward(q)
        return q, _converged(self.pose_error(pos, rot), pos_tol, rot_tol)

    def collisions(self):
        """Returns contacts at the current state as 'self' or 'table' labels."""
        labels = []
        for c in self.data.contact[:self.data.ncon]:
            labels.append('table' if self._table in (c.geom1, c.geom2)
                          else 'self')
        return labels

    def min_singular_value(self):
        """Smallest singular value of the TCP Jacobian at the current state."""
        return np.linalg.svd(self.jacobian(), compute_uv=False)[-1]
