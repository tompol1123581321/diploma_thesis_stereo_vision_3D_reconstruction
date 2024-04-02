import cv2
import numpy as np
from generate_point_cloud.load_parameters import load_stereo_parameters
from generate_point_cloud.generate_point_cloud import generate_point_cloud
from visualize_point_cloud import visualize_point_cloud

def main():
    # Folder containing calibration and image files
    folder_name = 'delivery_area_2l'

    # Load stereo parameters
    try:
        stereo_params  = load_stereo_parameters(folder_name)
    except Exception as e:
        print("Error:", e)
        return

    # Select example images
    image_left_path = folder_name+'/im0.png'
    image_right_path = folder_name+'/im1.png'

    # Generate point cloud
    point_cloud = generate_point_cloud(image_left_path, image_right_path, stereo_params)

    # Visualize the point cloud
    visualize_point_cloud(point_cloud)

if __name__ == "__main__":
    # Test stereo parameters
    main()