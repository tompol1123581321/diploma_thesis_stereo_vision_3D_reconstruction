import cv2
import numpy as np
from generate_point_cloud.feature_matching import feature_matching

baseline=59.8896

def generate_point_cloud(image_left, image_right, stereo_params):
    camera_matrix_left, _, camera_matrix_right, _, R, T, _, _ = stereo_params
    focal_length = camera_matrix_left[0, 0]
    cx = camera_matrix_left[0, 2]
    cy = camera_matrix_left[1, 2]

    # Load stereo images
    img_left = cv2.imread(image_left)
    img_right = cv2.imread(image_right)

    # Stereo rectification
    img_left_rect = cv2.undistort(img_left, camera_matrix_left, None)
    img_right_rect = cv2.undistort(img_right, camera_matrix_right, None)

    # Convert images to grayscale
    img_left_gray = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)

    # Compute disparity map using stereo block matching
    stereo = cv2.StereoBM.create(numDisparities=64, blockSize=21)
    disparity = stereo.compute(img_left_gray, img_right_gray)

    # Perform bidirectional feature matching
    matched_keypoints_left,matched_keypoints_right = feature_matching(img_left_gray, img_right_gray)

    # Triangulate 3D points from disparity map and matched features
    points_3d = []
    for i in range(len(matched_keypoints_left)):
        # Get matched keypoints from the left and right images
        x_left, y_left = matched_keypoints_left[i][0]
        x_right, y_right = matched_keypoints_right[i][0]

        # Average x and y coordinates
        x_avg = (x_left + x_right) / 2
        y_avg = (y_left + y_right) / 2

        # Use disparity value from left image
        disparity_value = disparity[int(y_left), int(x_left)]

        # Convert disparity value to depth (Z)
        depth = baseline * focal_length / disparity_value

        # Compute 3D coordinates using the averaged coordinates
        X = (x_avg - cx) * depth / focal_length
        Y = (y_avg - cy) * depth / focal_length
        Z = depth

        points_3d.append((X, Y, Z))

    return points_3d
