#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>


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

int main() {
    std::cout << "OpenCV image test started!" << std::endl;

    std::string imagePath1 = "200_images/000000.png";
    std::string imagePath2 = "200_images/000001.png";
    cv::Mat image1 = cv::imread(imagePath1);
    cv::Mat image2 = cv::imread(imagePath2);

    if (image1.empty() || image2.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        system("pause");
        return -1;
    }

    // Resizing images 
    cv::Mat resimage1, resimage2;
    cv::resize(image1, resimage1, cv::Size(760, 480), 0, 0);
    cv::resize(image2, resimage2, cv::Size(760, 480), 0, 0);

    // Convert to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(resimage1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resimage2, gray2, cv::COLOR_BGR2GRAY);

    // orb detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    //check if descriptors are empty
    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "No descriptors found in one or both images!" << std::endl;
        return -1;
    }
    
    // converting descriptors to 32 bit float
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);

    // find variance of descriptors
    cv::Mat allDescriptors;
    cv::vconcat(descriptors1, descriptors2, allDescriptors);
    cv::Mat variance = computeColVariance(allDescriptors);
    for (int i = 0; i < variance.cols; ++i) {
        std::cout << "Dimension " << i << ": " << variance.at<double>(0, i) << std::endl;
    }

    //FLANN uses kd-trees for matching with top 5 variance dimensions   (i am here)
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();

    


    std::cout << "Image1 loaded! Size: " << resimage1.cols << " x " << resimage1.rows << std::endl;
    std::cout << "Image1 loaded! Size: " << resimage2.cols << " x " << resimage2.rows << std::endl;

    cv::Mat combined;
    cv::hconcat(resimage1, resimage2, combined);
    cv::imshow("Previous (Left) + Current (Right)", combined);
    cv::waitKey(0);

    return 0;
}
