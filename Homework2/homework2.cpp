#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

// Create a SIFT detector
cv::Ptr<cv::SIFT> sift = cv::SIFT::create(2500);

//camera intrinsic matrix
cv::Mat K = (cv::Mat_<double>(3, 3) <<
 7.070493e+02, 0.0, 6.040814e+02, 
 0.0, 7.070493e+02, 1.805066e+02, 
 0.0, 0.0, 1.0);

// Load an image given its index
cv::Mat loadImage(int i, const std::string& basePath) {
    std::stringstream ss;
    ss << basePath << std::setw(6) << std::setfill('0') << i << ".png";
    cv::Mat img = cv::imread(ss.str());
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << std::endl;
    }
    return img;
}

// Detect and compute SIFT features
void detectAndCompute(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

// Match features using FLANN + Lowe's ratio test
std::vector<cv::DMatch> matchFeatures(const cv::Mat& desc1, const cv::Mat& desc2) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(desc1, desc2, knnMatches, 2);
    std::vector<cv::DMatch> goodMatches;
    for (const auto& knn : knnMatches) {
        if (knn.size() >= 2 && knn[0].distance < 0.8f * knn[1].distance) {
            goodMatches.push_back(knn[0]);
        }
    }
    return goodMatches;
}

// Filter matches using RANSAC
std::vector<cv::DMatch> filterMatchesRANSAC(
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches) 
{
    if (matches.size() <= 6) {
        return matches;
    }
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    std::vector<uchar> inliersMask;
    cv::Mat affine = cv::estimateAffine2D(pts1, pts2, inliersMask, cv::RANSAC, 4);
    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); ++i) {
        if (inliersMask[i]) {
            inliers.push_back(matches[i]);
        }
    }
    return inliers;
}

/*
* This method is used to track the SIFT features in a sequence of 200 images for homework 1 of robotics.
* The images are stored in the "200_images" folder and are named from 000000.png to 000200.png.
* The program uses OpenCV to read the images, detect SIFT features, and then match the features between consecutive images.
* The matches are filtered using the ratio test (because FLANN returns the 2 best matches) and RANSAC to find inliers.
* The matches are then drawn on the images and saved as a video file named "tracking_result.mp4".
*/
int main() {
    std::cout << "OpenCV image test started!" << std::endl;

    const std::string basePath = "200_images/";
    const int totalFrames = 201;
    const cv::Size imageSize(760, 480);
    const cv::Size frameSize(1520, 480);

    cv::VideoWriter cloudWriter("pointcloud_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, cv::Size(600, 600));
    if (!cloudWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }

    // Load the first image
    cv::Mat prevImg = loadImage(0, basePath);
    if (prevImg.empty()) return -1;
    cv::resize(prevImg, prevImg, imageSize);
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Mat prevDescriptors;
    detectAndCompute(prevImg, prevKeypoints, prevDescriptors);

    // Loop over all frames
    for (int i = 1; i < totalFrames; ++i) {
        std::cout << "Processing frame: " << i << std::endl;
        cv::Mat currImg = loadImage(i, basePath);
        if (currImg.empty()) continue;
        cv::resize(currImg, currImg, imageSize);

        std::vector<cv::KeyPoint> currKeypoints;
        cv::Mat currDescriptors;
        detectAndCompute(currImg, currKeypoints, currDescriptors);
        if (currDescriptors.empty()) continue;

        auto matches = matchFeatures(prevDescriptors, currDescriptors);
        auto inlierMatches = filterMatchesRANSAC(prevKeypoints, currKeypoints, matches);

        //extract points from the matches
        std::vector<cv::Point2f> prevPoints, currPoints;
        for (const auto& match : inlierMatches) {
            prevPoints.push_back(prevKeypoints[match.queryIdx].pt);
            currPoints.push_back(currKeypoints[match.trainIdx].pt);
        }
        
        // compute the fundamental matrix
        cv::Mat F = cv::findFundamentalMat(prevPoints, currPoints, cv::FM_RANSAC);

        // compute the essential matrix
        cv::Mat E = K.t() * F * K;

        //print F and E
        std::cout << "F: " << F << std::endl;
        std::cout << "E: " << E << std::endl;
        cv::Mat R, t;
        cv::recoverPose(E, prevPoints, currPoints, K, R, t);
        std::cout << "R: " << R << std::endl;
        std::cout << "t: " << t << std::endl;

        // projection matrix
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
        cv::Mat Rt(3, 4, CV_64F);
        R.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
        t.copyTo(Rt(cv::Rect(3, 0, 1, 3)));
        cv::Mat P2 = K * Rt;

       
        //normalize the points
        //std::vector<cv::Point2f> prevPointsNorm, currPointsNorm;
        //cv::undistortPoints(prevPoints, prevPointsNorm, K, cv::Mat());
        //cv::undistortPoints(currPoints, currPointsNorm, K, cv::Mat());

        // triangulate points
        cv::Mat points4D;     
        cv::triangulatePoints(P1, P2, prevPoints, currPoints, points4D);
        for (int i = 0; i < points4D.cols; ++i) {
            float w = points4D.at<float>(3, i);
            if (std::abs(w) > 1e-8) { // Avoid divide by zero
                points4D.at<float>(0, i) /= w;
                points4D.at<float>(1, i) /= w;
                points4D.at<float>(2, i) /= w;
                points4D.at<float>(3, i) = 1.0;
            }
        }        
        std::cout << "points4D: " << points4D << std::endl;
        
        //create point cloud
        cv::Mat cloud = cv::Mat::zeros(600, 600, CV_8UC3);

        for (int p = 0; p < points4D.cols; p++) {
            float x = points4D.at<float>(0, p);
            float y = points4D.at<float>(1, p);
            float z = points4D.at<float>(2, p);

            int u = static_cast<int>(x / 3 + 300);
            int v = static_cast<int>(z / 3 + 300);

            if (u >= 0 && u < 600 && v >= 0 && v < 600) {
                cv::circle(cloud, cv::Point(u, v), 1, cv::Scalar(0, 255, 0), -1);
            }
        }
        cloudWriter.write(cloud);
        cv::imshow("Point Cloud", cloud);
        cv::waitKey(1);


        prevImg = currImg.clone();
        prevKeypoints = currKeypoints;
        prevDescriptors = currDescriptors.clone();
    }

    cloudWriter.release();
    std::cout << "Video saved as pointcloud_result.mp4" << std::endl;
    return 0;
}