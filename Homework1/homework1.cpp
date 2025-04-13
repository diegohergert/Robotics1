#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

/*
* This method is used to track the SIFT features in a sequence of 200 images for homework 1 of robotics.
* The images are stored in the "200_images" folder and are named from 000000.png to 000200.png.
*/
int main() {
    std::cout << "OpenCV image test started!" << std::endl;

    std::string imagePath = "200_images/";
    const int frames = 201;
    const cv::Size imageSize(760, 480);
    const cv::Size frameSize(1520, 480);

    cv::VideoWriter videoWriter("tracking_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 12, frameSize);
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(1200);

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

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(prevDescriptors, currentDescriptors, knnMatches, 2);

        // Filter matches using a ratio test to check the distance between the best and second-best matches
        std::vector<cv::DMatch> goodMatches;
        for (size_t j = 0; j < knnMatches.size(); j++) {
            if (knnMatches[j].size() >= 2 && knnMatches[j][0].distance < 0.7 * knnMatches[j][1].distance) {
                goodMatches.push_back(knnMatches[j][0]);
            }
        }

        //ransac
        std::vector<cv::DMatch> inlierMatches;
        if (goodMatches.size() > 6) {
            std::vector<cv::Point2f> prevPoints, currentPoints;
            for (size_t j = 0; j < goodMatches.size(); j++) {
                prevPoints.push_back(prevKeypoints[goodMatches[j].queryIdx].pt);
                currentPoints.push_back(currentKeypoints[goodMatches[j].trainIdx].pt);
            }
            std::vector<uchar> inliersMask;
            cv::Mat H = cv::findHomography(prevPoints, currentPoints, cv::RANSAC, 3, inliersMask);
            for (size_t j = 0; j < inliersMask.size(); j++) {
                if (inliersMask[j]) {
                    inlierMatches.push_back(goodMatches[j]);
                }
            }
        } else {
            inlierMatches = goodMatches;
        }

        cv::Mat imgMatches;
        cv::drawMatches(prevImg, prevKeypoints, currentImg, currentKeypoints, inlierMatches, imgMatches,
                      cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>());
        
        videoWriter.write(imgMatches);

    // Uncomment to see display while processing (p to pause and any key to continue)
    /*cv::imshow("Matches", imgMatches);
    int key = cv::waitKey(1) & 0xFF;
    if (key == 'p') {  // If 'p' is pressed, pause indefinitely.
        std::cout << "Paused. Press any key to continue..." << std::endl;
        cv::waitKey(0);
    }
            */

        prevImg = currentImg.clone();
        prevKeypoints = currentKeypoints;
        prevDescriptors = currentDescriptors.clone();
    }

    videoWriter.release();
    std::cout << "Video saved as tracking_result.mp4" << std::endl;
    return 0;
}