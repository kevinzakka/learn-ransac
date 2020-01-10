"""2D linear regression with RANSAC.

Note: The RANSAC loop is explicitly computed here for
clarity of exposition rather than using the estimator
class in `ransac.py`.
"""

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
  np.random.seed(0)
  num_points = 100

  # create linear function
  x = np.linspace(0, 30, num=num_points)
  noise = 2.5 * np.random.randn(len(x)) + 10
  y = 2*x - 3 + noise

  # create outliers
  frac = int(0.2 * num_points)
  idxs = np.random.choice(np.arange(num_points), size=frac, replace=False)
  y[idxs] += np.random.randint(-40, 20, frac)

  # least squares without ransac
  x = x.reshape(-1, 1)
  x = np.hstack((x, np.ones((len(x), 1))))  # add column of ones
  y = y.reshape(-1, 1)
  linreg_params = np.linalg.lstsq(x, y, rcond=None)[0]

  # ======================================================================== #
  # alternatively, we could have used the normal equation to solve
  # for the parameters but it is less numerically stable than
  # `np.linalg.lstsq` which uses QR decomposition.
  # ======================================================================== #
  # linreg_params = np.linalg.inv(x_lin.T @ x_lin) @ x_lin.T @ y_lin
  # ======================================================================== #

  # ======================================================================== #
  # RANSAC parameters
  # ======================================================================== #
  s = 2  # the smallest number of points required to estimate model params
  N = 100  # the number of RANSAC iterations
  d = 2  # the threshold used to identify a point that fits well
  # ======================================================================== #

  best_model = None
  best_inliers = None
  best_num_inliers = 0
  best_residual_sum = np.inf
  for ii in range(N):
    # randomly pick s points from the data
    pt_idxs = np.random.choice(np.arange(num_points), size=s, replace=False)
    other_idxs = np.setdiff1d(np.arange(num_points), pt_idxs)

    # determine params from points
    pts_x = x[pt_idxs]
    pts_y = y[pt_idxs]

    # normal equation: x = (A.T A).inv A.T b
    params = np.linalg.lstsq(pts_x, pts_y, rcond=None)[0]

    # compute residual (l2 loss in this case)
    residuals =  (x @ params - y)**2
    residuals_sum = residuals.sum()

    # determine points within threshold
    # we need to loop through all other points
    # and compute their distance to the estimated line
    num = np.abs((pts_y[1, 0]-pts_y[0, 0])*x - \
        (pts_x[1, 0]-pts_x[0, 0])*y + \
            pts_x[1, 0]*pts_y[0, 0] - \
                pts_y[1, 0]*pts_x[0, 0])
    denum = np.sqrt((pts_y[1, 0]-pts_y[0, 0])**2 + \
        (pts_x[1, 0]-pts_x[0, 0])**2)
    distances = num / denum

    inliers = (distances <= d)[:, 0]
    num_inliers = np.sum(inliers)
    num_outliers = len(other_idxs) - num_inliers

    # decide if better
    if (best_num_inliers < num_inliers) or (best_residual_sum > residuals_sum):
      best_num_inliers = num_inliers
      best_residual_sum = residuals_sum
      best_inliers = inliers

  # refit model using all inliers for this set
  pts_x = x[best_inliers]
  pts_y = y[best_inliers]
  best_params = np.linalg.lstsq(pts_x, pts_y, rcond=None)[0]

  # plot
  y_pred_ransac = x @ best_params
  plt.plot(x[:, 0], y_pred_ransac[:, 0], label="ransac")
  y_pred_lin = x @ linreg_params
  plt.plot(x[:, 0], y_pred_lin[:, 0], label="linreg")
  plt.scatter(x[best_inliers][:, 0], y[best_inliers], c='b', label="inliers")
  plt.scatter(x[~best_inliers][:, 0], y[~best_inliers], c='r', label="outliers")
  plt.legend(loc='upper left')
  plt.savefig("./plots/toy.png", format="png", dpi=200)
  plt.show()