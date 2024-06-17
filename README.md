# Indoor Autonomous Drone Competition

This code gets a recorded video from Tello drone and detects ArUco markers and calculate the position in the frame and the 3D data of marks.
The 3D data is: distance, yaw, pitch and roll.

The code exports a csv file with all the parsed data and the video that every detected ArUco mark in a green rectangular frame with its ID and 3D axes.

The code enable to pause the video during the run with pressing the key 'p'.

The video can be process faster by changing the parameter 'running_rate' to bigger numer.

This repo have 3 example videos, csv export for the 'challengeB.mp4' and it's output video.

