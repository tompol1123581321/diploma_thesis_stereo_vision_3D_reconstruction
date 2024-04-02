import cv2
import numpy as np

def feature_matching(img_left_gray, img_right_gray):
    # Create AKAZE detector and descriptor
    akaze = cv2.AKAZE.create()

    # Find keypoints and descriptors in the left image
    keypoints_left, descriptors_left = akaze.detectAndCompute(img_left_gray, None)

    # Find keypoints and descriptors in the right image
    keypoints_right, descriptors_right = akaze.detectAndCompute(img_right_gray, None)

    # Convert descriptors to np.float32
    descriptors_left = descriptors_left.astype(np.float32)
    descriptors_right = descriptors_right.astype(np.float32)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher()

    # Match descriptors
    matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

    # Apply ratio test to filter matches
    good_matches = []
    for m, n in matches:
        good_matches.append(m)

    # Extract matched keypoints
    matched_keypoints_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_keypoints_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    print(matched_keypoints_left.__len__(),matched_keypoints_right.__len__())

    return matched_keypoints_left, matched_keypoints_right
