import os
import numpy as np

def load_stereo_parameters(folder_name):
    calib_file_path = os.path.join(folder_name, 'calib.txt')

    try:
        # Load calibration parameters from calib.txt
        with open(calib_file_path, 'r') as f:
            lines = f.readlines()

        # Extract camera matrices and baseline from calib.txt
        camera_matrix_left = None
        camera_matrix_right = None
        baseline = None
        for line in lines:
            if line.startswith('cam0='):
                camera_matrix_left = np.array([[float(val) for val in part.split()] for part in line.split('[', 1)[1].split(']')[0].split(';')])
            elif line.startswith('cam1='):
                camera_matrix_right = np.array([[float(val) for val in part.split()] for part in line.split('[', 1)[1].split(']')[0].split(';')])
            elif line.startswith('baseline='):
                baseline = float(line.split('=')[1])



        R = np.eye(3)  # Identity matrix for rotation
        T = np.array([baseline, 0, 0])  # Baseline

        stereo_params = (camera_matrix_left, None, camera_matrix_right, None, R, T, None, None)

        return stereo_params

    except FileNotFoundError:
        raise FileNotFoundError("Files not found. Make sure the paths are correct.")
    except Exception as e:
        raise e