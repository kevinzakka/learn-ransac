"""2D orthogonal regression with RANSAC.

In orthogonal regression, we minimize the orthogonal
distance of the data points to the estimated line.
This is in contrast to linear least squares, which
minimizes the vertical distances of the points to
the line.

Orthogonal regression is usually a more effective
modeling tool when it is assumed that error exists
both in `X` and in `y`. In contrast, recall that
linear regression assumes there is error only
in the `y` variable, i.e. `y = W@x + ε`.

Minimizing orthogonal distances is equivalent to
finding the first principal component of the data.
This means that can apply PCA to our data matrix
`X` and extract its first principal component.

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

  # TODO: to be continued