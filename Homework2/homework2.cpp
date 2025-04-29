#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

cv::Mat K = (cv::Mat_<double>(3, 3) << 
7.070493e02, 0, 6.040814e02, 
0, 7.070493e02, 1.805066e02, 
0, 0, 1);

// Function to plot the 3D point cloud and trajectory
cv::Mat plot3DTrajectory(const std::vector<cv::Point3f>& trajectory, const std::vector<cv::Point3f>& globalCloud, const cv::Size& plotSize) {
    // Create a black canvas for the plot
    cv::Mat plot = cv::Mat::zeros(plotSize, CV_8UC3);
    
    // Define the plot area and scale factors
    float scale = 2.0f;  // Scale factor for visualization
    cv::Point2f center(plotSize.width / 2, plotSize.height / 2);
    
    // Draw the 3D points (point cloud) if available
    for (const auto& point : globalCloud) {
        // Project 3D point to 2D for visualization (simple orthographic projection)
        cv::Point2f pt2d(
            center.x + point.x * scale,
            center.y + point.z * scale  // Using z for vertical dimension in plot
        );
        
        if (pt2d.x >= 0 && pt2d.x < plotSize.width && pt2d.y >= 0 && pt2d.y < plotSize.height) {
            // Draw the point in dark green (smaller points)
            cv::circle(plot, pt2d, 1, cv::Scalar(0, 100, 0), -1);
        }
    }
    
    // Draw the trajectory in cyan color
    if (trajectory.size() > 1) {
        for (size_t i = 1; i < trajectory.size(); i++) {
            cv::Point2f pt1(
                center.x + trajectory[i-1].x * scale,
                center.y + trajectory[i-1].z * scale  // Using z for vertical dimension
            );
            
            cv::Point2f pt2(
                center.x + trajectory[i].x * scale,
                center.y + trajectory[i].z * scale
            );
            
            // Draw trajectory line in cyan
            cv::line(plot, pt1, pt2, cv::Scalar(255, 255, 0), 2);
        }
        
        // Draw the current position as a larger cyan circle
        if (!trajectory.empty()) {
            cv::Point2f currentPos(
                center.x + trajectory.back().x * scale,
                center.y + trajectory.back().z * scale
            );
            cv::circle(plot, currentPos, 5, cv::Scalar(255, 255, 0), -1);
        }
    }
    
    return plot;
}

/*
* This method is used to track the SIFT features in a sequence of 200 images for homework 1 of robotics.
* The images are stored in the "200_images" folder and are named from 000000.png to 000200.png.
* The program uses OpenCV to read the images, detect SIFT features, and then match the features between consecutive images.
* The matches are filtered using the ratio test (because FLANN returns the 2 best matches) and RANSAC to find inliers.
* The matches are then drawn on the images and saved as a video file named "tracking_result.mp4".
*/
int main() {
    //print the start message
    std::cout << "OpenCV image test started!" << std::endl;

    //set up image path and video writer
    std::string imagePath = "200_images/";
    const int frames = 201;
    const cv::Size imageSize(1500, 400);

    cv::namedWindow("3D Cloud & Trajectory", cv::WINDOW_NORMAL);
    cv::resizeWindow("3D Cloud & Trajectory", 1500, 400*2);

    // Create a VideoWriter object to save the output video
    cv::VideoWriter videoWriter("tracking_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 12, cv::Size(imageSize.width, imageSize.height*2));
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }

    // Create a SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(3000, 3.1, .05, 12, 2);

    // Read the first image and detect keypoints and descriptors
    std::stringstream ss;
    ss << imagePath << std::setw(6) << std::setfill('0') << 0 << ".png";
    cv::Mat prevImg = cv::imread(ss.str());
    if (prevImg.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }
    cv::resize(prevImg, prevImg, imageSize);
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Mat prevDescriptors;
    
    sift->detectAndCompute(prevImg, cv::noArray(), prevKeypoints, prevDescriptors);
    if (prevDescriptors.empty()) {
        std::cerr << "Error: No descriptors found in the first image." << std::endl;
        return -1;
    }

    //create a vector to store the trajectory
    std::vector<cv::Point3f> trajectory;
    // Add initial position to trajectory
    trajectory.emplace_back(0.0f, 0.0f, 0.0f);

    //create a global pose matrix
    cv::Mat globalPose = cv::Mat::eye(4, 4, CV_64F);

    //create a vector to store the 3D points
    std::vector<cv::Point3f> globalCloud;

    // Create initial plot for the first frame
    cv::Mat plot = plot3DTrajectory(trajectory, globalCloud, imageSize);
    
    // Create initial combined image
    cv::Mat matchResized, plotResized, combined;
    cv::resize(prevImg, matchResized, imageSize);
    cv::resize(plot, plotResized, imageSize);
    cv::vconcat(matchResized, plotResized, combined);
    
    // Display and write the first frame
    cv::imshow("3D Cloud & Trajectory", combined);
    videoWriter.write(combined);

    // Loop through the remaining images
    for (int i = 1; i < frames; i++) {
        std::stringstream ssCurrent;
        ssCurrent << imagePath << std::setw(6) << std::setfill('0') << i << ".png";
        cv::Mat currentImg = cv::imread(ssCurrent.str());
        if (currentImg.empty()) {
            std::cerr << "Error: Could not read the image." << std::endl;
            continue;  // Skip this frame and try the next one
        }
        cv::resize(currentImg, currentImg, imageSize);
        std::vector<cv::KeyPoint> currentKeypoints;
        cv::Mat currentDescriptors;
        sift->detectAndCompute(currentImg, cv::noArray(), currentKeypoints, currentDescriptors);
        if (currentDescriptors.empty()) {
            std::cerr << "Error: No descriptors found in the current image." << std::endl;
            continue;  // Skip this frame and try the next one
        }

        // Match desriptors using FLANN (has kd tree like algorithm for matching)
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(prevDescriptors, currentDescriptors, knnMatches, 2);

        // Filter matches using a ratio test to check the distance between the best and second-best matches
        std::vector<cv::DMatch> goodMatches;
        const float maxDistance = 150.0; // ratio test threshold
        for (size_t j = 0; j < knnMatches.size(); j++) {
            if (knnMatches[j].size() >= 2 && knnMatches[j][0].distance < 0.55 * knnMatches[j][1].distance
                && knnMatches[j][0].distance < maxDistance) {
                goodMatches.push_back(knnMatches[j][0]);
            }
        }

        cv::Mat essential;
        std::vector<cv::Point2f> prevPoints, currentPoints;
        std::vector<uchar> inliersMask;
        //ransac to remove outlier matches by estimating the affine transformation
        std::vector<cv::DMatch> inlierMatches;
        if (goodMatches.size() > 8) {
            for (size_t j = 0; j < goodMatches.size(); j++) {
                prevPoints.push_back(prevKeypoints[goodMatches[j].queryIdx].pt);
                currentPoints.push_back(currentKeypoints[goodMatches[j].trainIdx].pt);
            }
            essential = cv::findEssentialMat(prevPoints, currentPoints, 
                K, cv::RANSAC, 0.99999, 1, inliersMask);
            for (size_t j = 0; j < inliersMask.size(); j++) {
                if (inliersMask[j]) {
                    inlierMatches.push_back(goodMatches[j]);
                }
            }
        } else {
            inlierMatches = goodMatches;
        }

        // get the camera rotation and translation
        cv::Mat R, t;
        cv::Mat maskMat(inliersMask);
        cv::recoverPose(essential, prevPoints, currentPoints, K, R, t, maskMat);
        
        // Create transformation matrix
        cv::Mat Rt = cv::Mat::zeros(4, 4, CV_64F);
        Rt.at<double>(3, 3) = 1.0;
        R.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
        t.copyTo(Rt(cv::Rect(3, 0, 1, 3)));
        
        // Store previous pose for triangulation
        cv::Mat posePrev = globalPose.clone();

        // Update global pose
        globalPose = globalPose * Rt;
        
        // Add to trajectory (using cyan color in plot)
        trajectory.emplace_back(
            static_cast<float>(globalPose.at<double>(0,3)),
            static_cast<float>(globalPose.at<double>(1,3)),
            static_cast<float>(globalPose.at<double>(2,3))
        );

        // 2) triangulate
        cv::Mat P1 = K * cv::Mat::eye(3,4,CV_64F);
        cv::Mat P2 = K * Rt(cv::Rect(0,0,4,3));
        cv::Mat pts4D;
        cv::triangulatePoints(P1, P2, prevPoints, currentPoints, pts4D);

        std::vector<cv::Point3f> pts3D;
        cv::convertPointsFromHomogeneous(pts4D.t(), pts3D);
        // invert posePrev to map from previous-camera → world
        cv::Mat posePrevInv = posePrev.inv();

        // for every camera-frame point:
        for (auto &p : pts3D) {
            // homogenous point in prev-camera coords
            cv::Mat pCam = (cv::Mat_<double>(4,1) << p.x, p.y, p.z, 1.0);
            // world coords:
            cv::Mat pWorldH = posePrevInv * pCam;
            double w = pWorldH.at<double>(3,0);
            if (fabs(w) < 1e-6) continue;
            cv::Point3f pw(
            float(pWorldH.at<double>(0,0)/w),
            float(pWorldH.at<double>(1,0)/w),
            float(pWorldH.at<double>(2,0)/w)
            );
            // (optional) z-filter to cut behind/backside points
            if (pw.z > 0 && pw.z < 50.0f)
            globalCloud.push_back(pw);
        }

        // Draw matches on the current image
        for (const auto& match : inlierMatches) {
            cv::circle(currentImg, currentKeypoints[match.trainIdx].pt, 4, cv::Scalar(0, 255, 0), -1); // green dot
        }

        // Generate 3D trajectory plot
        plot = plot3DTrajectory(trajectory, globalCloud, imageSize);

        // Combine the images
        cv::Mat plotResized, matchResized;
        cv::resize(plot, plotResized, imageSize);
        cv::resize(currentImg, matchResized, imageSize);
        cv::Mat combined;
        cv::vconcat(matchResized, plotResized, combined);
        
        // Display and save
        cv::imshow("3D Cloud & Trajectory", combined);
        cv::waitKey(1);
        videoWriter.write(combined);

        std::cout << "Processed frame " << i << std::endl;

        // update which image is which
        prevImg = currentImg.clone();
        prevKeypoints = currentKeypoints;
        prevDescriptors = currentDescriptors.clone();
    }

    // release video writer
    videoWriter.release();
    std::cout << "Video saved as tracking_result.mp4" << std::endl;
    return 0;
}