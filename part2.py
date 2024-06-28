import cv2
import numpy as np
import csv

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Define the camera matrix and distortion coefficients

# tello camera metrix
# camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
#                           [0.000000, 919.018377, 351.238301],
#                           [0.000000, 0.000000, 1.000000]])
# dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# Macbook Pro camera metrix
camera_matrix = np.array([[1430.4797487055312, 0., 955.95824923704834],
                          [0., 1431.0020567966558, 538.32765562244504],
                          [0., 0., 1.]])
dist_coeffs = np.array([-0.0077421668067861908, 0.29422522351747443, 0.00054344989814312128,
                        -0.00050951878359647518, -0.61876923892130387])

# Define the length of the marker's side in meters
marker_length = 0.1  # 10 cm


class ArucoDetector:
    def __init__(self,
                 csv_output_file='aruco_detection_results.csv',
                 video_output_file='output_video.mp4'):
        self.video_output_file = video_output_file

        self.csv_file = open(csv_output_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Frame ID', 'ArUCo ID', 'ArUCo 2D', 'ArUCo 3D (dist yaw pitch roll)'])

    def video_processing(self, running_rate=2):

        frame_id = 0

        cap = cv2.VideoCapture('challengeB.mp4')

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
        out.release()
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

    def draw_instuctions_for_closest(self, image, closest_data):
        frame_id, aruco_id, corners, aruco_3d = closest_data
        dist, yaw, pitch, roll = aruco_3d

        # Add text for distance, yaw, pitch, and roll
        text = f'Distance: {dist:.2f}'
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        text = f'Yaw: {yaw:.2f} Pitch: {pitch:.2f} Roll: {roll:.2f}'
        cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        image_center = (image.shape[1] // 2, image.shape[0] // 2)

        marker_center = np.mean(corners, axis=0).astype(int)

        # Calculate the difference between the marker center and the image center
        dx = image_center[0] - marker_center[0]
        dy = image_center[1] - marker_center[1]

        # Draw lines indicating the required movement
        # Horizontal movement
        if dx != 0:
            color = (3, 169, 252) if dx < 0 else (3, 252, 210)
            cv2.arrowedLine(image, image_center, (image_center[0] - dx, image_center[1]), color, 2, tipLength=0.5)

        # Vertical movement
        if dy != 0:
            color = (3, 169, 252) if dy < 0 else (3, 252, 210)
            cv2.arrowedLine(image, image_center, (image_center[0], image_center[1] - dy), color, 2, tipLength=0.5)

        # Draw the marker center
        cv2.circle(image, tuple(marker_center), 5, (255, 0, 0), -1)

        # Draw round arrow for roll
        radius = 50  # Radius of the circular arrow

        # Define the angles for the arc (0 to roll angle)
        start_angle = 0
        end_angle = roll if roll > 0 else 360 + roll  # Adjust end angle for negative yaw

        # Draw the arc
        cv2.ellipse(image, image_center, (radius, radius), 0, start_angle, end_angle, (255, 255, 0), 2)

        # Calculate the arrowhead position
        angle = np.deg2rad(end_angle)
        arrow_tip = (int(image_center[0] + radius * np.cos(angle)), int(image_center[1] + radius * np.sin(angle)))
        arrow_base1 = (int(arrow_tip[0] - 10 * np.cos(angle - np.pi / 6)), int(arrow_tip[1] - 10 * np.sin(angle - np.pi / 6)))
        arrow_base2 = (int(arrow_tip[0] - 10 * np.cos(angle + np.pi / 6)), int(arrow_tip[1] - 10 * np.sin(angle + np.pi / 6)))
        # Draw the arrowhead
        cv2.line(image, arrow_tip, arrow_base1, (255, 255, 0), 2)
        cv2.line(image, arrow_tip, arrow_base2, (255, 255, 0), 2)

        yaw_text = None
        if yaw < -2:
            yaw_text = f'yaw: left'
        elif yaw > 2:
            yaw_text = f'yaw: right'

        if yaw_text is not None:
            cv2.putText(image, yaw_text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        return image

    def frame_processing(self, frame, frame_id):
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(255, 255, 255))
        closest_data = None
        closest_dist = float('inf')
        if ids is not None:
            for i in range(len(ids)):
                obj_points = np.array([
                    [-marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ])

                # Estimate pose of each marker
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)

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
                roll = np.arctan2(rmat[1, 0], rmat[0, 0])
                yaw = np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
                pitch = np.arctan2(rmat[2, 1], rmat[2, 2])

                aruco_3d = [dist, np.degrees(yaw), np.degrees(pitch), np.degrees(roll)]
                if dist < closest_dist:
                    closest_dist = dist
                    closest_data = [frame_id, aruco_id, aruco_2d, aruco_3d]

                self.export_aruco_data(frame_id, aruco_id, aruco_2d, aruco_3d)

                frame = self.draw_aruco_info(corner, frame, aruco_id)

            if closest_data is not None:
                frame = self.draw_instuctions_for_closest(frame, closest_data)

        return frame


if __name__ == '__main__':
    aruco_detector = ArucoDetector()
    aruco_detector.video_processing()
