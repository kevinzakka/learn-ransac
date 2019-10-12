"""A simple RANSAC class implementation.

Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/fit.py
"""

import numpy as np


class RansacEstimator:
  """Random Sample Consensus.
  """
  def __init__(self, min_samples, residual_threshold, max_trials):
    self.min_samples = min_samples
    self.residual_threshold = residual_threshold
    self.max_trials = max_trials

  def fit(self, model, data):
    """Robustely fit a model to the data.

    Args:
      model: a class object that implements `estimate` and `residuals` methods.
      data: the data to fit the model to. Can be a list of data pairs.

    Returns:
      best_model: the model with the largest consensus set and lower residual error.
      inliers: a boolean mask indicating the inlier subset in the data.
    """
    best_model = None
    best_inliers = None
    best_num_inliers = 0
    best_residual_mse = np.inf

    if not isinstance(data, (tuple, list)):
      data = [data]
    num_data = len(data[0])

    for trial in range(self.max_trials):
      # randomly select subset
      rand_subset_idxs = np.random.choice(np.arange(num_data), size=self.min_samples, replace=False)
      rand_subset = [d[rand_subset_idxs] for d in data]

      # estimate with model
      model.estimate(*rand_subset)

      # compute residuals
      residuals = model.residuals(*data)
      residuals_mse = (residuals**2).mean()
      inliers = residuals <= self.residual_threshold
      num_inliers = np.sum(inliers)

      # decide if better
      if (best_num_inliers < num_inliers) or (best_residual_mse > residuals_mse):
        best_num_inliers = num_inliers
        best_residual_mse = residuals_mse
        best_inliers = inliers

    # refit model using all inliers for this set
    data_inliers = [d[best_inliers] for d in data]
    model.estimate(*data_inliers)
    residuals = model.residuals(*data_inliers)
    residuals_mse = (residuals**2).mean()

    ret = {
        "best_params": model.params,
        "best_residual": residuals_mse,
        "best_inliers": best_inliers,
    }

    return ret