"""2D regression with RANSAC.
"""

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    num_points = 100
    
    # create linear function
    x = np.linspace(0, 30, num=num_points)
    noise = 0.7 * np.random.randn(len(x))
    y = 2*x - 3 + noise

    # create outliers
    frac = int(0.2 * num_points)
    idxs = np.random.choice(np.arange(num_points), size=frac, replace=False)
    y[idxs] += np.random.randint(-40, 20, frac)

    # fit regular linear regression
    x_lin = x.copy().reshape(-1, 1)
    x_lin = np.hstack((x_lin, np.ones((len(x_lin), 1))))
    y_lin = y.copy().reshape(-1, 1)
    linreg_params = np.linalg.inv(x_lin.T @ x_lin) @ x_lin.T @ y_lin

    s = 2  # the smallest number of points required to estimate model params
    e = 0.25  # probability that a point is an outlier
    N = 100 # the number of iterations 
    d = 1  # the threshold used to identify a point that fits well
    T = 5  # the number of nearby points required to assert model fits well

    scores = {}
    for ii in range(N):
        # randomly pick s points from the data
        pt_idxs = np.random.choice(np.arange(num_points), size=s, replace=False)
        other_idxs = np.setdiff1d(np.arange(num_points), pt_idxs)

        # determine params from points
        pts_x = x[pt_idxs].reshape(-1, 1)
        # add columns of 1 for intercept
        pts_x = np.hstack((pts_x, np.ones((len(pts_x), 1))))
        pts_y = y[pt_idxs].reshape(-1, 1)

        # normal equation: x = (A.T A).inv A.T b
        params = np.linalg.inv(pts_x.T @ pts_x) @ pts_x.T @ pts_y

        # determine points within threshold
        # we need to loop through all other points
        # and compute their distance to the estimated line
        num = np.abs((pts_y[1, 0]-pts_y[0, 0])*x - (pts_x[1, 0]-pts_x[0, 0])*y + pts_x[1, 0]*pts_y[0, 0] - pts_y[1, 0]*pts_x[0, 0])
        denum = np.sqrt((pts_y[1, 0]-pts_y[0, 0])**2 + (pts_x[1, 0]-pts_x[0, 0])**2)
        distances = num / denum

        inliers = distances <= d
        num_inliers = np.sum(inliers)
        num_outliers = len(other_idxs) - num_inliers

        scores[ii] = [num_inliers, inliers, pt_idxs]
    
    best_model = sorted(scores.items(), key=lambda x: x[1][0])[-1][1]
    best_inliers = best_model[1]
    sample_idxs = best_model[2]

    # refit model using all inliers for this set
    pts_x = x[sample_idxs]
    pts_y = y[sample_idxs]
    pts_x = np.hstack([pts_x, x[best_inliers]]).reshape(-1, 1)
    pts_x = np.hstack((pts_x, np.ones((len(pts_x), 1))))
    pts_y = np.hstack([pts_y, y[best_inliers]]).reshape(-1, 1)
    best_params = np.linalg.inv(pts_x.T @ pts_x) @ pts_x.T @ pts_y


    plot_x = x.copy().reshape(-1, 1)
    plot_x = np.hstack((plot_x, np.ones((len(plot_x), 1))))
    plot_y = plot_x @ best_params
    plt.plot(plot_x[:, 0], plot_y[:, 0], label="ransac")
    plot_y_lin = plot_x @ linreg_params
    plt.plot(plot_x[:, 0], plot_y_lin[:, 0], label="linreg")
    # plt.plot(plot_x[:, 0], plot_x[:, 0]*2 -3, label="ground truth")
    plt.scatter(x[best_inliers], y[best_inliers], c='b', label="inliers")
    plt.scatter(x[~best_inliers], y[~best_inliers], c='r', label="outliers")
    plt.legend(loc='upper left')
    plt.savefig("./plots/toy.png", format="png", dpi=200)
    plt.show()
