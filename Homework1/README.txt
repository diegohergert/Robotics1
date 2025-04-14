Diego Hergert
811281898

This homework tracks features on a sequence of 200 images using SIFT features and FLANN matcher 
(Uses Kd-tree and other algorithms). To remove outliers i selected the closest point out of the two 
returned by the FLANN matcher then applied ransac to further remove errors. For ransac I used an 
affine estimate to have less calculations for my program. The left fram in the "tracking_result.mp4"
is the previous frame while the right image is the current frame. The dots are all the features found,
and the lines connect features with valid matches. I have the video recorded for 15 frames per second
because otherwise it would go through them very fast. I mostly used libraries from openCV and the compiler
I used was cl (MSVC (microsoft c++)).

to compile I would run (in terminal):
cl.exe /EHsc /I "C:/opencv/build/include" /I "C:/opencv/build/include/opencv2" Homework1/homework1.cpp /link /LIBPATH:"C:/opencv/build/x64/vc16/lib" opencv_world4110.lib /OUT:Homework1/homework1.exe

to run:
.\homework1.exe
(the video will be stored as "tracking_result.mp4")

I used SIFT to detect keypoints and as a descriptor.
FLANN as the descriptor matcher.
Ransac using an estimated affine transformation with a reprojection threshold of 5.5.


A potential enhancement would be to cross check the FLANN matching to find better matches and then apply Ransac.
I attempted to remove dimensions for the FLANN matcher by calculating the top 25 dimensions by largest variance,
but it did not work unless I allowed every dimension to be used.