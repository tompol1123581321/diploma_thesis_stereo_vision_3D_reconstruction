import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud, figsize=(10, 8), marker_size=5, alpha=1.0, cmap='jet', dpi=100):
    """
    Visualize a point cloud.

    Args:
    - point_cloud (numpy.ndarray): The point cloud with shape (height, width, 3).
                                   Each point cloud contains 3D coordinates (x, y, z).
    - figsize (tuple): Size of the figure (width, height) in inches.
    - marker_size (float): Size of markers used for points in the scatter plot.
    - alpha (float): Opacity of markers (0.0 for fully transparent, 1.0 for fully opaque).
    - cmap (str): Colormap to use for coloring the points.
    - dpi (int): Dots per inch for the resolution of the output image (if saved).
    """

    # Extract x, y, and z coordinates
    x = point_cloud[:, :, 0]
    y = point_cloud[:, :, 1]
    z = point_cloud[:, :, 2]

    # Flatten the coordinates for plotting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Create a 3D scatter plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_flat, y_flat, z_flat, s=marker_size, alpha=alpha, c=z_flat, cmap=cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization')

    # Show plot
    plt.show()

# Example usage
# Load the generated point cloud

def generate_point_cloud(image_left, image_right, stereo_params):
    # Load stereo parameters
    camera_matrix_left, _, camera_matrix_right, _, R, T, _, _ = stereo_params
    
    # Load images
    img_left = cv2.imread(image_left)
    img_right = cv2.imread(image_right)

    # Undistort and rectify images
    img_left_rect = cv2.undistort(img_left, camera_matrix_left, None)
    img_right_rect = cv2.undistort(img_right, camera_matrix_right, None)
    
    # Convert images to grayscale
    img_left_gray = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)

    # Stereo rectification
    _, _, _, _, Q, _, _ = cv2.stereoRectify(camera_matrix_left, None, camera_matrix_right, None, img_left_gray.shape[::-1], R, T)

    # Compute disparity map
    stereo = cv2.StereoBM().create(numDisparities=16, blockSize=11)  # Adjust blockSize to meet requirements
    disparity = stereo.compute(img_left_gray, img_right_gray)

    # Compute point cloud
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    return points_3d

# Example usage
image_left_path = 'im0.png'
image_right_path = 'im1.png'

# Camera calibration parameters (from ETH3D)
camera_matrix_left = np.array([[544.861, 0, 433.468],
                               [0, 544.861, 232.786],
                               [0, 0, 1]])
camera_matrix_right = np.array([[544.861, 0, 433.468],
                                [0, 544.861, 232.786],
                                [0, 0, 1]])
R = np.eye(3)  # Identity matrix for rotation
T = np.array([60.2559, 0, 0])  # Baseline

# Load stereo parameters
stereo_params = (camera_matrix_left, None, camera_matrix_right, None, R, T, None, None)

# Generate point cloud
point_cloud = generate_point_cloud(image_left_path, image_right_path, stereo_params)

# Save point cloud (you may want to save it to a file format like .ply)
np.save('point_cloud.npy', point_cloud)
point_cloud = np.load('point_cloud.npy')

# Visualize the point cloud with increased quality
visualize_point_cloud(point_cloud, figsize=(12, 10), marker_size=10, alpha=0.6, cmap='viridis', dpi=150)
