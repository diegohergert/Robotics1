#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <vector>


cv::Mat computeColVariance(const cv::Mat& mat) {
    cv::Mat mean, stddev;
    cv::Mat variance(1, mat.cols, CV_64F);

    for (int col = 0; col < mat.cols; ++col) {
        cv::Mat colMat = mat.col(col);
        cv::meanStdDev(colMat, mean, stddev);
        variance.at<double>(0, col) = stddev.at<double>(0, 0) * stddev.at<double>(0, 0);
    }

    return variance; 
}

void reduceDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2, cv::Mat &reducedDescriptors1, cv::Mat &reducedDescriptors2) {
    // Check if descriptors are empty
    if (descriptors1.empty() || descriptors2.empty()) {
        reducedDescriptors1 = descriptors1.clone();
        reducedDescriptors2 = descriptors2.clone();
        return;
    }

    // Compute the mean and standard deviation of the descriptors
    cv::Mat allDescriptors;
    cv::vconcat(descriptors1, descriptors2, allDescriptors);
    cv::Mat variance = computeColVariance(allDescriptors);

    std::vector<int> indices(variance.cols);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&variance](int a, int b) {
        return variance.at<double>(0, a) > variance.at<double>(0, b);
    });

    // total amount of dimensions to keep
    int dimensionsToKeep = std::min(127, (int)indices.size());
    
    std::vector<int> topIndices;
    for(int i = 0; i < dimensionsToKeep; i++) {
        topIndices.push_back(indices[i]);
    }

    reducedDescriptors1 = cv::Mat(descriptors1.rows, topIndices.size(), descriptors1.type());
    reducedDescriptors2 = cv::Mat(descriptors2.rows, topIndices.size(), descriptors2.type());
    for (size_t i = 0; i < topIndices.size(); i++) {
        int index = topIndices[i];
        reducedDescriptors1.col(i) = descriptors1.col(index);
        reducedDescriptors2.col(i) = descriptors2.col(index);
    }
}

int main() {
    std::cout << "OpenCV image test started!" << std::endl;

    std::string imagePath = "200_images/";
    const int frames = 200;
    const cv::Size imageSize(760, 480);
    const cv::Size frameSize(1520, 480);

    cv::VideoWriter videoWriter("tracking_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 60, frameSize);
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(1500);

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
        cv::Mat reducedPrevDescriptors, reducedCurrentDescriptors;
        reduceDescriptors(prevDescriptors, currentDescriptors, reducedPrevDescriptors, reducedCurrentDescriptors);

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

        // Uncomment these lines if you want to display the matches while processing
        // Display the matches
    cv::imshow("Matches", imgMatches);

    // Wait for a short delay (1 millisecond here) and check if the pause key ('p') was pressed.
    int key = cv::waitKey(1) & 0xFF;
    if (key == 'p') {  // If 'p' is pressed, pause indefinitely.
        std::cout << "Paused. Press any key to continue..." << std::endl;
        cv::waitKey(0);
    }


        prevImg = currentImg.clone();
        prevKeypoints = currentKeypoints;
        prevDescriptors = currentDescriptors.clone();
    }

    videoWriter.release();
    std::cout << "Video saved as tracking_result.mp4" << std::endl;
    return 0;
}