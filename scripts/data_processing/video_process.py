#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pickle


def extract_position(video_file, intrinsics, extrinsics, save_path=None):
    print "Loading video..."

    video = cv2.VideoCapture(video_file)
    status = video.isOpened()

    if status:
        print "Starting to process..."

        # collect metadata about the file.
        fps = video.get(cv2.CAP_PROP_FPS)
        dt = 1/(fps/1000)

        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

        # initialize variables
        traj = []

        mtx, dist = intrinsics
        z, T_cam = extrinsics

        alp = mtx[0, 0]
        beta = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]

        current_frame = 0

        # process all frames
        while current_frame < total_frames:
            if current_frame % 60 == 0:
                print "Processing frame ", current_frame

            success, image = video.read()
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)

            if not success:
                break

            # undistort the image first
            image = cv2.undistort(image, mtx, dist)

            # blur and convert color space
            blurred = cv2.GaussianBlur(image, (21, 21), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # track color
            green_lower = (45, 86, 46)
            green_upper = (60, 255, 255)

            mask = cv2.inRange(hsv, green_lower, green_upper)
            mask = cv2.erode(mask, None, iterations=3)
            mask = cv2.dilate(mask, None, iterations=3)

            mask_show = cv2.resize(mask, None, fx=0.5, fy=0.5)
            cv2.imshow("video", mask_show)
            cv2.waitKey(10)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]

            # compute center
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                center = (M["m10"] / M["m00"], M["m01"] / M["m00"])

                # convert to camera frame
                x_cam = z / alp * (center[0] - cx)
                y_cam = z / beta * (center[1] - cy)

                # convert to world frame
                p_cam = np.array([x_cam, y_cam, 0.0, 1.0])
                p_world = np.dot(T_cam, p_cam)

                traj.append(p_world[:3])
            else:
                pass

        video.release()
        cv2.destroyAllWindows()
    else:
        pass

    if save_path is not None:
        with open(save_path + "/extracted_traj.pkl", "w") as f:
            pickle.dump(np.asarray(traj), f)


def visualize_path(data_file):
    fig, axes = plt.subplots()

    with open(data_file) as f:
        traj = pickle.load(f)

    axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    plt.show()


def compute_err(data_file):
    pass


if __name__ == "__main__":
    # load the intrinsics of camera
    root_path = "/home/yuhang/Documents/camera_calibration"
    data = np.load(root_path + "/calibration_data.npz")
    intrisics = (data["intrinsic_matrix"], data["distCoeff"])

    # define the extrinsics
    Tcam = np.eye(4)
    Tcam[1, 1] = -1
    Tcam[2, 2] = -1

    z = 65

    extract_position(root_path + "/position_tracking.mov", intrisics, (z, Tcam), root_path)

    visualize_path(root_path + "/extracted_traj.pkl")
