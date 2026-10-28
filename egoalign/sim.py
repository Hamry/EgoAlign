"""MuJoCo scene: Menagerie Franka Panda, table, mug and a POV camera."""

import os
import pathlib
import urllib.request
from xml.etree import ElementTree

import mujoco
import numpy as np

MENAGERIE_SHA = '71f066ad0be9cd271f7ed58c030243ef157af9f4'
_RAW = ('https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/'
        f'{MENAGERIE_SHA}/franka_emika_panda/')

MUG_RADIUS, MUG_HALF_HEIGHT = 0.03, 0.05
TCP_OFFSET = 0.1034
# OpenCV camera (x right, y down, z forward) to MuJoCo camera (z backward).
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def _download(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _write(dst, data):
    tmp = dst.with_suffix(dst.suffix + '.part')
    tmp.write_bytes(data)
    tmp.replace(dst)


def panda_xml():
    """Returns the cached Menagerie panda.xml, downloading it if missing."""
    root = pathlib.Path(os.environ.get(
        'EGOALIGN_CACHE', pathlib.Path.home() / '.cache' / 'egoalign'))
    root = root / f'menagerie_panda_{MENAGERIE_SHA[:8]}'
    xml = root / 'panda.xml'
    if xml.exists():
        return xml
    (root / 'assets').mkdir(parents=True, exist_ok=True)
    text = _download(_RAW + 'panda.xml')
    for mesh in ElementTree.fromstring(text).iter('mesh'):
        dst = root / 'assets' / mesh.get('file')
        if not dst.exists():
            _write(dst, _download(_RAW + 'assets/' + mesh.get('file')))
    _write(root / 'LICENSE', _download(_RAW + 'LICENSE'))
    _write(xml, text)  # Written last: marks the cache complete.
    return xml


def camera_quat(rot_cv):
    """MuJoCo camera quaternion from an OpenCV-convention world rotation."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rot_cv @ CV_TO_MJ).ravel())
    return quat


def build(mug_xy=(0.5, 0.0), cam_pos=(1.0, 0.0, 0.6), cam_rot=None,
          hfov_deg=65.0, width=320, height=240):
    """Compiles the scene.

    Args:
        mug_xy: Mug position on the table, robot base frame [m].
        cam_pos: POV camera position, robot base frame [m].
        cam_rot: (3, 3) OpenCV-convention camera-to-world rotation; defaults to
            looking from `cam_pos` at the mug.
        hfov_deg: Camera horizontal field of view [deg].
        width: Render width [px], used for the vertical FOV.
        height: Render height [px].

    Returns:
        A compiled `mujoco.MjModel`.
    """
    spec = mujoco.MjSpec.from_file(str(panda_xml()))
    spec.body('hand').add_site(name='tcp', pos=[0, 0, TCP_OFFSET])
    world = spec.worldbody
    world.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.8] * 3)
    world.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0.5, 0, -0.025], size=[0.6, 0.6, 0.025],
                   rgba=[0.55, 0.4, 0.3, 1])
    mug = world.add_body(name='mug', pos=[*mug_xy, MUG_HALF_HEIGHT])
    mug.add_freejoint(name='mug')
    mug.add_geom(name='mug', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[MUG_RADIUS, MUG_HALF_HEIGHT, 0], mass=0.2,
                 friction=[1.5, 0.01, 0.001], rgba=[0.9, 0.2, 0.2, 1])
    if cam_rot is None:
        cam_rot = look_at(cam_pos, [*mug_xy, MUG_HALF_HEIGHT])
    fovy = 2 * np.degrees(np.arctan(
        np.tan(np.radians(hfov_deg) / 2) * height / width))
    world.add_camera(name='pov', pos=list(cam_pos),
                     quat=list(camera_quat(cam_rot)), fovy=fovy)
    model = spec.compile()
    model.vis.global_.offwidth = max(width, 640)
    model.vis.global_.offheight = max(height, 480)
    return model


def look_at(eye, target):
    """OpenCV-convention rotation for a camera at `eye` facing `target`."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:  # Looking straight down or up.
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def mug_height(data):
    """Height of the mug centre above its resting height [m]."""
    return data.body('mug').xpos[2] - MUG_HALF_HEIGHT
