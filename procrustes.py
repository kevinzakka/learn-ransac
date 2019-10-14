"""Robust orthogonal procrustes problem.
"""

import numpy as np
import open3d as o3d

from walle.core import RotationMatrix


def transform_xyz(xyz, transform):
  """Applies a rigid transform to an (N, 3) point cloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # homogenize 3D pointcloud
  xyz_t_h = (transform @ xyz_h.T).T  # apply transform
  return xyz_t_h[:, :3]


def transform_from_rotm_tr(rotm, tr):
  transform = np.eye(4)
  transform[:3, :3] = rotm
  transform[:3, 3] = tr
  return transform


def estimate_rigid_transform_rotm(X, Y):
  """Determines the rotation and translation that best aligns X to Y.
  """
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
  return transform_from_rotm_tr(rot, trans)


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


if __name__ == "__main__":
  # create a random rigid transform
  transform = np.eye(4)
  transform[:3, :3] = RotationMatrix.random()
  transform[:3, 3] = 2 * np.random.randn(3) + 1

  # create a random source point cloud
  src_pc = 5 * np.random.randn(100, 3) + 2
  dst_pc = transform_xyz(src_pc, transform)

  # corrupt 10%
  rand_corrupt = np.random.choice(np.arange(len(src_pc)), replace=False, size=10)
  dst_pc[rand_corrupt] += np.random.uniform(-10, 10, (10, 3))

  # estimate the naive way
  transform_naive = estimate_rigid_transform_rotm(src_pc, dst_pc)

  # ================== #
  # RANSAC params
  # ================== #
  s = 3
  N = 100
  d = 0.001
  # ================== #

  scores = {}
  for ii in range(N):
    # randomly select `s` pairs of points
    rand_idxs = np.random.choice(np.arange(len(src_pc)), replace=False, size=s)
    src_pts = src_pc[rand_idxs]
    dst_pts = dst_pc[rand_idxs]

    # estimate rigid transform from pair
    transform_est = estimate_rigid_transform_rotm(src_pts, dst_pts)

    # apply estimated transform on source points
    dst_pts_est = transform_xyz(src_pc, transform_est)

    # compare with ground truth and count inliers
    distances = np.linalg.norm(dst_pts_est - dst_pc, axis=1)
    inliers = distances <= d
    num_inliers = np.sum(inliers) - s

    # store score
    scores[ii] = [num_inliers, inliers, rand_idxs]

  best_model = sorted(scores.items(), key=lambda x: x[1][0])[-1][1]
  best_inliers = best_model[1]
  sample_idxs = best_model[2]

  # refit model using all inliers for this set
  src_pc_trimmed = src_pc[best_inliers]
  dst_pc_trimmed = dst_pc[best_inliers]
  outlier_pc = dst_pc[rand_corrupt]
  transform_ransac = estimate_rigid_transform_rotm(src_pc_trimmed, dst_pc_trimmed)

  print("naive: {}".format(np.linalg.norm(transform_naive - transform)))
  print("ransac: {}".format(np.linalg.norm(transform_ransac - transform)))

  view_pc([dst_pc_trimmed, transform_xyz(src_pc, transform_naive), outlier_pc])
  view_pc([dst_pc_trimmed, transform_xyz(src_pc, transform_ransac), outlier_pc])
