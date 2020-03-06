We record images from a camera which contain an apriltag and detect corners of those tags. We then estimate the camera pose using the following two methods. Once we have the camera pose, we render a bird mesh over the center of the apriltag. 

1) PnP.py - this solves the Perspective N point problem by assuming that all the 4 corners of the AprilTag are coplanar. This simplifies the problem to just solving an SVD to get the camera pose.  

2) P3P.py - here we dont make the above assumption, therefore to solve the Three Point Perspective Pose Estimation Problem we use [Grunert's Solution](https://haralick-org.torahcode.us/journals/three_point_perspective.pdf). From this we get the correspondence of 3 points in the world frame and the camera frame, and we solve for camera pose using the Procrustes Method. 

4) The recorded images are present inside the frames folder and the corners of the apriltags in corners.npy
5) You can directly execute run_P3P.py for seeing the results of step(1)
6) You can directly execute run_PnP.py for seeing the results of step(2)


![alt text](bird_collineation.gif)
