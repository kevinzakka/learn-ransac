import numpy as np

from ransac import RansacEstimator
from walle.core import RotationMatrix


def gen_data():
  # create a random rigid transform
  transform = np.eye(4)
  transform[:3, :3] = RotationMatrix.random()
  transform[:3, 3] = 2 * np.random.randn(3) + 1

  # create a random source point cloud
  src_pc = 5 * np.random.randn(100, 3) + 2
  dst_pc = Procrustes.transform_xyz(src_pc, transform)

  # corrupt 10%
  rand_corrupt = np.random.choice(np.arange(len(src_pc)), replace=False, size=10)
  dst_pc[rand_corrupt] += np.random.uniform(-10, 10, (10, 3))

  return src_pc, dst_pc, transform


class Procrustes:
  """Determines the best rigid transform [1] between two point clouds.

  References:
    [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
  """
  def __init__(self, transform=None):
    self._transform = transform

  def __call__(self, xyz):
    return Procrustes.transform_xyz(xyz, self._transform)

  @classmethod
  def transform_xyz(cls, xyz, transform):
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
    
    if self._transform is None:
      self._transform = np.eye(4)
    self._transform[:3, :3] = rot
    self._transform[:3, 3] = trans

  def residuals(self, X, Y):
    Y_est = self(X) 
    return np.linalg.norm(Y_est - Y, axis=1)

  @property
  def params(self):
    return self._transform


if __name__ == "__main__":
  src_pc, dst_pc, transform_true = gen_data()

  # estimate the naive way
  naive_model = Procrustes()
  naive_model.estimate(src_pc, dst_pc)
  mse_naive = (naive_model.residuals(src_pc, dst_pc)**2).mean()
  transform_naive = naive_model.params

  # estimate with RANSAC
  ransac = RansacEstimator(
      min_samples=3,
      residual_threshold=0.001,
      max_trials=1000,
  )
  ret = ransac.fit(Procrustes(), [src_pc, dst_pc])
  transform_ransac = ret["best_params"]
  mse_ransac = ret["best_residual"]
  inliers_ransac = ret["best_inliers"]

  print("mse naive: {}".format(mse_naive))
  print("mse ransac: {}".format(mse_ransac))
