# learn-ransac

Learning about the different uses of [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus): Ransom Sample Consensus.

RANSAC is an interative method for estimating the parameters of a model from data that contains outliers.

## API

```python
from ransac import RansacEstimator

# estimate with RANSAC
ransac = RansacEstimator(
    min_samples=3,
    residual_threshold=0.001,
    max_trials=1000,
)
ret = ransac.fit(Procrustes(), [src_pc, dst_pc])

# extract best values
transform4x4_ransac = ret["best_params"]
mse_ransac = ret["best_residual"]
inliers_ransac = ret["best_inliers"]
```

## 2D Regression

As a toy example, we'll be corrupting a linear function of `x` with random outliers. Then we'll plot the predicted line from RANSAC and linear regression.

<p align="center">
<img src="./plots/toy.png" width="75%" alt="Drawing">
</p>

## Rigid Transform Estimation

Suppose we are given two point clouds and we would like to estimate the rigid transformation that aligns the first point cloud to the other.

In the regular setting, we assume the point correspondences between both point clouds are correct and compute the optimal rigid transform using SVD. This is known as the [Orthogonal Procrustes Problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem).

But what happens if we have some outliers? This can happen for instance when calibrating a camera and a bad checkerboard center is detected. The answer ... RANSAC!

<p align="center">
<img src="./plots/naive_pro.png" width=48% alt="Drawing">
<img src="./plots/ransac_pro.png" width=48% alt="Drawing">
</p>

In the figure above, the red points correspond to corrupted correspondences. On the left, we have the rigid transform computed the regular way, without considering outliers. The blue and orange points are supposed to be superimposed. Notice how the outliers have affected the transformation and there is a slight offset between all points. On the right is the result of RANSAC + Procrustes. The blue points now completely superimpose the orange ones (not visible anymore) and are not affected by the red outliers.

## References

- [Kris Kitani's slides for 16-385](http://www.cs.cmu.edu/~16385/s17/Slides/10.3_2D_Alignment__RANSAC.pdf)
- [RANSAC for Dummies by Marco Zuliani](http://www.cs.tau.ac.il/~turkel/imagepapers/RANSAC4Dummies.pdf)
- [Robert Collins' slides for CSE486](http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf)
- [`skimage` implementation](https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/fit.py)
