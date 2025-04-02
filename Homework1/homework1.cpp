#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main() {
    std::string imagePath = "Homework1/200_images/sample.jpg"; // Pick any test image
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Failed to load image at " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully! Size: " 
              << image.cols << "x" << image.rows << std::endl;

    cv::imshow("Test Image", image);
    cv::waitKey(0);
    return 0;
}
