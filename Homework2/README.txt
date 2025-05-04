Diego Hergert
811: 811281898

This homework implements basic Visual Odometry (VO) using
OpenCV and the 200 images from the KITTI dataset.
The program tracks features using SIFT and FLANN,
then estimates the essential matrix to recover the camera pose.
The poses are then stored to build the trajectory for the 
camera. The point matches from the tracked features are
also triangulated to build a sparse 3D point cloud. The 
output video is saved as "tracking_result.mp4".

The program is written c++ and uses the OpenCV, iostream,
sstream, iomanip, and vector libraries. I used the MSVC compiler
because openCV was having errors with g++ when I tried using that. 

1. feature matching
Use SIFT to detect keypoints and then FLANN to matche
using k-NN matcher with a ratio test and distance filter.

2. Estimate pose
The essential matrix is computed when applying RANSAC to 
get R and t to track the trajectory of the camera.

3. Traingulation
3D points are triangulated between matching points in consecutive
frames. The points are then transformed to the world coordinate frames
and points behind the camera or too far out are filtered out.

4. Plotting
The 2D point cloud is then plotted with the camera trajectory with the current
image and feature matches on the top. This is saved as "tracking_result.mp4".

To compile I used ctrl shift b to build on vscode but it is the same as the following
line of code. (this was generated from my tasks.json using chatGPT)
cl.exe /EHsc /I C:/opencv/build/include /I C:/opencv/build/include/opencv2 Homework2/homework2.cpp /link /LIBPATH:C:/opencv/build/x64/vc16/lib opencv_world4110.lib /OUT:Homework2/homework2.exe

To run the executable generated after compiling I ran the following line:
.\homework2.exe

References:
KITTI Odometry dataset (and I looked at some of the work submitted there to better understand)

OpenCV documentation to use the library

A lot of youtube tutorials on VO for the steps to take