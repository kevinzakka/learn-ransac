"""Estimate a rigid transform between 2 point clouds.
"""

import numpy as np
import open3d as o3d

from ransac import RansacEstimator
from walle.core import RotationMatrix


def gen_data(N=100, frac=0.1):
  # create a random rigid transform
  transform = np.eye(4)
  transform[:3, :3] = RotationMatrix.random()
  transform[:3, 3] = 2 * np.random.randn(3) + 1

  # create a random source point cloud
  src_pc = 5 * np.random.randn(N, 3) + 2
  dst_pc = Procrustes.transform_xyz(src_pc, transform)

  # corrupt
  rand_corrupt = np.random.choice(np.arange(len(src_pc)), replace=False, size=int(frac*N))
  dst_pc[rand_corrupt] += np.random.uniform(-10, 10, (int(frac*N), 3))

  return src_pc, dst_pc, transform, rand_corrupt


def transform_from_rotm_tr(rotm, tr):
  transform = np.eye(4)
  transform[:3, :3] = rotm
  transform[:3, 3] = tr
  return transform


def view_pc(xyzrgbs):
  """Displays a list of colored pointclouds.
  """
  pcs = []
  for xyzrgb in xyzrgbs:
    pts = xyzrgb[:, :3].copy().astype(np.float64)
    if xyzrgb.shape[1] == 3:
      clrs = np.zeros((xyzrgb.shape[0], 3)).astype(np.float64)
    else:
      clrs = (xyzrgb[:, 3:].copy() / 255).astype(np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(clrs)
    pcs.append(pc)
  pcs[0].paint_uniform_color([1, 0.706, 0])
  pcs[1].paint_uniform_color([0, 0.651, 0.929])
  pcs[2].paint_uniform_color([1, 0, 0])
  o3d.visualization.draw_geometries(pcs)


class Procrustes:
  """Determines the best rigid transform [1] between two point clouds.

  References:
    [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
  """
  def __init__(self, transform=None):
    self._transform = transform

  def __call__(self, xyz):
    return Procrustes.transform_xyz(xyz, self._transform)

  @staticmethod
  def transform_xyz(xyz, transform):
    """Applies a rigid transform to an (N, 3) point cloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # homogenize 3D pointcloud
    xyz_t_h = (transform @ xyz_h.T).T  # apply transform
    return xyz_t_h[:, :3]

  def estimate(self, X, Y):
    # find centroids
    X_c = np.mean(X, axis=0)
    Y_c = np.mean(Y, axis=0)

    # shift
    X_s = X - X_c
    Y_s = Y - Y_c

    # compute SVD of covariance matrix
    cov = Y_s.T @ X_s
    u, _, vt = np.linalg.svd(cov)

    # determine rotation
    rot = u @ vt
    if np.linalg.det(rot) < 0.:
      vt[2, :] *= -1
      rot = u @ vt

    # determine optimal translation
    trans = Y_c - rot @ X_c

    self._transform = transform_from_rotm_tr(rot, trans)

  def residuals(self, X, Y):
    """L2 distance between point correspondences.
    """
    Y_est = self(X)
    sum_sq = np.sum((Y_est - Y)**2, axis=1)
    return sum_sq

  @property
  def params(self):
    return self._transform


if __name__ == "__main__":
  src_pc, dst_pc, transform_true, rand_corrupt = gen_data(frac=0.2)

  # estimate without ransac, i.e. using all
  # point correspondences
  naive_model = Procrustes()
  naive_model.estimate(src_pc, dst_pc)
  transform_naive = naive_model.params
  mse_naive = np.sqrt(naive_model.residuals(src_pc, dst_pc).mean())
  print("mse naive: {}".format(mse_naive))

  # estimate with RANSAC
  ransac = RansacEstimator(
    min_samples=3,
    residual_threshold=(0.001)**2,
    max_trials=100,
  )
  ret = ransac.fit(Procrustes(), [src_pc, dst_pc])
  transform_ransac = ret["best_params"]
  inliers_ransac = ret["best_inliers"]
  mse_ransac = np.sqrt(Procrustes(transform_ransac).residuals(src_pc, dst_pc).mean())
  print("mse ransac all: {}".format(mse_ransac))
  mse_ransac_inliers = np.sqrt(
    Procrustes(transform_ransac).residuals(src_pc[inliers_ransac], dst_pc[inliers_ransac]).mean())
  print("mse ransac inliers: {}".format(mse_ransac_inliers))

  # plot
  src_pc_trimmed = src_pc[inliers_ransac]
  dst_pc_trimmed = dst_pc[inliers_ransac]
  outlier_pc = dst_pc[rand_corrupt]
  view_pc([dst_pc_trimmed, Procrustes.transform_xyz(src_pc, transform_naive), outlier_pc])
  view_pc([dst_pc_trimmed, Procrustes.transform_xyz(src_pc, transform_ransac), outlier_pc])