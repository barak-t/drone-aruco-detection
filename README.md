# Indoor Autonomous Drone Competition
## Part A:

This code gets a recorded video from Tello drone and detects ArUco markers and calculate the position in the frame and the 3D data of marks.
The 3D data is: distance, yaw, pitch and roll.

The code exports a csv file with all the parsed data and the video that every detected ArUco mark in a green rectangular frame with its ID and 3D axes.

The code enable to pause the video during the run with pressing the key 'p'.

The video can be process faster by changing the parameter 'running_rate' to bigger numer.

This repo have 3 example videos, csv export for the 'challengeB.mp4' and it's output video.

<img width="961" alt="image" src="https://github.com/barak-t/drone-aruco-detection/assets/64011788/924b8b26-3aee-4cad-982a-83ed7d278934">

<img width="962" alt="image" src="https://github.com/barak-t/drone-aruco-detection/assets/64011788/be507414-1930-4e65-88c7-e0433d6fdc72">

## Part B:

In part B, we add directions instuctions to the video that the goal is to center the Aruco marker in the middle and paraller to the camera.

The movement to **right-left** :left_right_arrow:  and **up-down** :arrow_up_down: is marked with arrows and the **yaw** rotations :arrow_right_hook:	 display on the screen.

In addition the **distance** bettwen the cammera and the Aruco is display, and all the others raw metrics.

For bonus the roll also display on sceen with turquoise circular arrow :arrows_counterclockwise:	

<img width="1504" alt="image" src="https://github.com/barak-t/drone-aruco-detection/assets/64011788/cc6eaf0b-5a50-4059-8caf-28c87512f34a">

**Related files:**
- part2.py : the main code for part 2
- live-direction-test.mp4 : live test with one Aruco marker, moving the cammera according the instuctions on the screen
- challengeB-with-directions.mp4 : the video from the part 1 with the new display
