import cv2
import numpy as np
import csv

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Define the camera matrix and distortion coefficients (need to be calibrated for your camera)
camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                          [0.000000, 919.018377, 351.238301],
                          [0.000000, 0.000000, 1.000000]])
dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# Define the length of the marker's side in meters
marker_length = 0.05


class ArucoDetector:
    def __init__(self,
                 input_file,
                 csv_output_file='aruco_detection_results.csv',
                 video_output_file='output_video.mp4'):
        self.input_file = input_file
        self.video_output_file = video_output_file

        self.csv_file = open(csv_output_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Frame ID', 'ArUCo ID', 'ArUCo 2D', 'ArUCo 3D (dist yaw pitch roll)'])

    def video_processing(self, running_rate=1):

        frame_id = 0

        cap = cv2.VideoCapture(self.input_file)

        # Get the video's frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(int(1000 / fps) / running_rate)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(self.video_output_file, fourcc, fps, (frame_width, frame_height))

        # Check if the video file opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            frame = self.frame_processing(frame, frame_id)

            # Display the frame
            cv2.imshow('ArUco Detection', frame)
            # Write the frame to the output video file
            out.write(frame)

            key = cv2.waitKey(frame_delay)
            # Pause using 'p' -> continue with any key
            if key == ord('p'):
                cv2.waitKey(-1)

            # Exit on 'q' key
            elif key == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

    def export_aruco_data(self, frame_id, aruco_id, aruco_2d, aruco_3d):
        # Write to CSV
        self.csv_writer.writerow([frame_id, aruco_id, aruco_2d, aruco_3d])
        print([frame_id, aruco_id, aruco_2d, aruco_3d])

    def draw_aruco_info(self, corner, frame, aruco_id):
        # Draw a green rectangle around the marker and put the ID
        pts = corner.reshape(-1, 2)
        for j in range(pts.shape[0]):
            pt1 = tuple(pts[j].astype(int))
            pt2 = tuple(pts[(j + 1) % pts.shape[0]].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(frame, str(aruco_id), tuple(pts[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

    def frame_processing(self, frame, frame_id):
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 255, 255))

        if ids is not None:
            for i in range(len(ids)):
                obj_points = np.array([
                    [-marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ])

                # Estimate pose of each marker
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners[i],
                    camera_matrix,
                    dist_coeffs
                )

                if not success:
                    continue

                # Draw axis for each marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

                # Extract the corners
                corner = corners[i][0]
                aruco_id = ids[i][0]
                aruco_2d = corner.tolist()
                tvec = tvec.flatten().tolist()
                rvec = rvec.flatten()

                # Calculate rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)

                # Convert rotation matrix to Euler angles
                dist = np.linalg.norm(tvec)
                yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
                pitch = np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
                roll = np.arctan2(rmat[2, 1], rmat[2, 2])

                aruco_3d = [dist, np.degrees(yaw), np.degrees(pitch), np.degrees(roll)]

                self.export_aruco_data(frame_id, aruco_id, aruco_2d, aruco_3d)

                frame = self.draw_aruco_info(corner, frame, aruco_id)

        return frame


if __name__ == '__main__':
    aruco_detector = ArucoDetector('challengeB.mp4')
    aruco_detector.video_processing(running_rate=1)
