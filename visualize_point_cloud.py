import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud, figsize=(10, 8), marker_size=5, alpha=1.0, cmap='jet', dpi=100):
    """
    Visualize a point cloud.

    Args:
    - point_cloud (list of tuples): The point cloud, where each tuple contains 3D coordinates (x, y, z).
    - figsize (tuple): Size of the figure (width, height) in inches.
    - marker_size (float): Size of markers used for points in the scatter plot.
    - alpha (float): Opacity of markers (0.0 for fully transparent, 1.0 for fully opaque).
    - cmap (str): Colormap to use for coloring the points.
    - dpi (int): Dots per inch for the resolution of the output image (if saved).
    """

    # Convert the list of tuples into a numpy array
    point_cloud_array = np.array(point_cloud)

    # Extract x, y, and z coordinates
    x = point_cloud_array[:, 0]
    y = point_cloud_array[:, 1]
    z = point_cloud_array[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=marker_size, alpha=alpha, c=z, cmap=cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization')

    # Show plot
    plt.show()
