"""Pointcloud registration using RANSAC.

Ref: http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def viz_registration(src_pc, targ_pc, rigid_transform):
    pass


def voxel_downsample(pc, voxel_size):
    """Uniformly downsample a point cloud using a regular voxel grid.
    """
    # determine bounds in each dimension
    x_min, x_max = np.min(pc[:, 0]), np.max(pc[:, 0])
    y_min, y_max = np.min(pc[:, 1]), np.max(pc[:, 1])
    z_min, z_max = np.min(pc[:, 2]), np.max(pc[:, 2])






if __name__ == "__main__":
    # read pcd files
    source_pc = o3d.io.read_point_cloud("./data/cloud_bin_0.pcd")
    target_pc = o3d.io.read_point_cloud("./data/cloud_bin_1.pcd")

    # ========================================================== #
    # pseudocode
    # ========================================================== #
    # 0a. possibly downsample point clouds for faster results
    # 0b. compute FPFH features of src and target pc
    # 1. randomly select 3 points from the src pc
    # 2. compute nearest neighbor of each src point in target
    # 3. using pairs, estimate rigid transform
    # 4. transform src using estimated transform
    # 5. determine inliers
    # 6. rinse and repeat
    # ========================================================== #

    source_pc.paint_uniform_color([1, 0.706, 0])
    target_pc.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_pc, target_pc])
