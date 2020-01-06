"""RANSAC + plane-fitting for normal estimation of point cloud data.

Ref: https://cs.nyu.edu/~panozzo/gp/04%20-%20Normal%20Estimation,%20Curves.pdf
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.spatial import KDTree
from walle.pointcloud import PointCloud
from ransac import RansacEstimator


class NormalEstimator:
  """Estimate a vector that is normal to the best-fitting plane.

  Concretely, finds a plane that minimizes the sum of squared distances.
  This is equivalent to computing the SVD of the (K, 3) matrix
  and extracting the left singular vector corresponding to
  the least singular value.

  Alternatively, we can form the matrix `X @ X.T` and compute its
  eigenvalue decomposition after which the normal vector
  of the best fitting plane will be the eigenvector corresponding
  to the smallest eigenvalue.
  """
  def __init__(self, up_vec=None):
    self.up_vec = up_vec
    self._normal_vec = None

  def fit_plane(self, X):
    """Finds the best fitting plane to a set of 3-D points.

    The parametric equation of a line is:
      a*x + b*y + c*z + d = 0
    where a, b, c correspond the the coefficients
    of the normal vector to the plane.

    Thus, given a point on the plane and a normal vector,
    d can be simply computed as the negative of the dot
    product.
    """
    d = -(X[0] @ self._normal_vec)
    return np.array([*self._normal_vec, d])

  def estimate(self, X):
    """Estimates a normal vector to the best-fitting plane.
    """
    # subtract out centroid
    X -= np.mean(X, axis=0)

    # normal vector is the left singular
    # vector associated with the smallest
    # singular value
    u, s, v = np.linalg.svd(X.T)
    norm_vec = u[:, -1]

    # this runs faster but I like SVD more :)
    # S = X.T @ X
    # eigvals, eigvecs = np.linalg.eig(S)
    # norm_vec = eigvecs[:, np.argmin(eigvals)]

    # orient with respect to up gravity vector
    if self.up_vec is not None:
      if np.isclose(np.linalg.norm(norm_vec), 0.):
        norm_vec = self.up_vec
      elif (norm_vec @ self.up_vec) < 0:
        norm_vec *= -1

    self._normal_vec = norm_vec

  def residuals(self, X):
    """Sum of the distances to best-fitting plane.
    """
    plane = self.fit_plane(X)
    plane_norm = np.linalg.norm(plane)
    d = plane[-1]
    plane = plane[:3]
    distances = (X * plane).sum(axis=1) - d
    return distances / plane_norm

  @property
  def params(self):
    return self._normal_vec


def compute_no_ransac(pts, tree):
  up_vec = np.array([0, 0, 1.])
  estimator = NormalEstimator(up_vec)
  all_dists, all_idxs = tree.query(pts, k=30, distance_upper_bound=0.1)
  normals = np.empty((len(pts), 3))
  for i in range(len(pts)):
    if not (i + 1) % 1000:
      print("{}/{}".format(i+1, len(pts)))
    idxs = all_idxs[i][all_dists[i] < np.inf]
    X = point_cloud[idxs, :3]
    estimator.estimate(X)
    normals[i] = estimator.params
  return normals


def compute_with_ransac(pts, tree, residual_threshold=0.002, max_trials=30):
  up_vec = np.array([0, 0, 1.])
  estimator = NormalEstimator(up_vec)
  ransac = RansacEstimator(4, residual_threshold, max_trials)
  all_dists, all_idxs = tree.query(pts, k=30, distance_upper_bound=0.1)
  normals = np.empty((len(pts), 3))
  for i in range(len(pts)):
    if not (i + 1) % 1000:
      print("{}/{}".format(i+1, len(pts)))
    idxs = all_idxs[i][all_dists[i] < np.inf]
    X = point_cloud[idxs, :3]
    ret = ransac.fit(estimator, X)
    normals[i] = ret["best_params"]
  return normals


def viz_normals(pts, clrs, normals):
  pts = pts.copy().astype(np.float64)
  clrs = pts.copy().astype(np.float64)
  normals = normals.copy().astype(np.float64)
  o3d_pc = [o3d.geometry.PointCloud()]
  o3d_pc[0].points = o3d.utility.Vector3dVector(pts)
  o3d_pc[0].colors = o3d.utility.Vector3dVector(clrs)
  o3d_pc[0].normals = o3d.utility.Vector3dVector(normals)
  o3d_pc.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
  o3d.visualization.draw_geometries(o3d_pc)


if __name__ == "__main__":
  cam_intr = np.loadtxt("data/camera_intrinsics.txt", delimiter=' ')
  cam_pose = np.loadtxt("data/camera_pose.txt", delimiter=' ')
  depth_im = cv2.imread("data/depth.png", -1).astype(float) / 10000
  color_im = cv2.cvtColor(cv2.imread("data/color.png"), cv2.COLOR_BGR2RGB)
  up_vec = np.array([0, 0, 1.])

  pc = PointCloud(color_im, depth_im, cam_intr)
  pc.make_pointcloud(cam_pose, depth_trunc=1.0, trim=True)
  pc.downsample(voxel_size=0.009, inplace=True)
  pc.view_point_cloud()

  point_cloud = pc.point_cloud
  pts = point_cloud[:, :3]
  clrs = point_cloud[:, 3:]

  print("Building KDTree...")
  tree = KDTree(pts)

  print("Estimating without RANSAC")
  normals_no_ransac = compute_no_ransac(pts, tree)
  viz_normals(pts, clrs, normals_no_ransac)

  print("Estimating with RANSAC")
  normals_with_ransac = compute_with_ransac(pts, tree)
  viz_normals(pts, clrs, normals_with_ransac)