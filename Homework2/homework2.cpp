#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

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
    const cv::Size imageSize(1300, 480);

    // Create a VideoWriter object to save the output video
    cv::VideoWriter videoWriter("tracking_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 12, imageSize);
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }

    // Create a SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(5000, 4.2, .03, 8, 1.8);

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
        for (size_t j = 0; j < knnMatches.size(); j++) {
            if (knnMatches[j].size() >= 2 && knnMatches[j][0].distance < 0.95 * knnMatches[j][1].distance) {
                goodMatches.push_back(knnMatches[j][0]);
            }
        }

        //ransac to remove outlier matches by estimating the affine transformation
        std::vector<cv::DMatch> inlierMatches;
        if (goodMatches.size() > 6) {
            std::vector<cv::Point2f> prevPoints, currentPoints;
            for (size_t j = 0; j < goodMatches.size(); j++) {
                prevPoints.push_back(prevKeypoints[goodMatches[j].queryIdx].pt);
                currentPoints.push_back(currentKeypoints[goodMatches[j].trainIdx].pt);
            }
            std::vector<uchar> inliersMask;
            cv::Mat affineH = cv::estimateAffine2D(prevPoints, currentPoints, inliersMask, cv::RANSAC, 5.5);

            for (size_t j = 0; j < inliersMask.size(); j++) {
                if (inliersMask[j]) {
                    inlierMatches.push_back(goodMatches[j]);
                }
            }
        } else {
            inlierMatches = goodMatches;
        }

        //draw matches on the images
        for (const auto& match : inlierMatches) {
            cv::circle(currentImg, currentKeypoints[match.trainIdx].pt, 4, cv::Scalar(0, 255, 0), -1); // green dot
        }

        // write the modified current frame to video
        videoWriter.write(currentImg);

        // show it (optional)
        cv::imshow("Good Matches", currentImg);
        cv::waitKey(1);
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