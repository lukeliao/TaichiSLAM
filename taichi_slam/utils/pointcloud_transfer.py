"""
Point cloud data conversion utilities (No ROS dependencies)
"""
import numpy as np
import math

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: [x, y, z, w] quaternion

    Returns:
        4x4 rotation matrix
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def transform_to_numpy(translation, rotation):
    """Convert translation and rotation to transformation matrix.

    Args:
        translation: [x, y, z] translation
        rotation: [x, y, z, w] quaternion

    Returns:
        R: 3x3 rotation matrix
        T: 3-element translation vector
    """
    T = np.array(translation, dtype=np.float64)
    R = quaternion_matrix(rotation)[:3, :3]
    return R, T

def load_pointcloud_from_file(filepath):
    """Load point cloud from file.

    Supports .npy, .xyz, .pcd formats

    Args:
        filepath: path to point cloud file

    Returns:
        xyz: Nx3 array of points
        rgb: Nx3 array of colors (or None if no color)
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npy':
        data = np.load(filepath)
        if data.shape[1] >= 6:
            xyz = data[:, :3]
            rgb = data[:, 3:6].astype(np.uint8)
        else:
            xyz = data[:, :3]
            rgb = None
        return xyz, rgb
    elif ext in ['.xyz', '.txt']:
        data = np.loadtxt(filepath)
        xyz = data[:, :3]
        rgb = data[:, 3:6].astype(np.uint8) if data.shape[1] >= 6 else None
        return xyz, rgb
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_pointcloud_to_file(xyz, rgb=None, filepath="pointcloud.npy"):
    """Save point cloud to file.

    Args:
        xyz: Nx3 array of points
        rgb: Nx3 array of colors (optional)
        filepath: output file path
    """
    if rgb is not None:
        data = np.hstack([xyz, rgb])
    else:
        data = xyz
    np.save(filepath, data)
