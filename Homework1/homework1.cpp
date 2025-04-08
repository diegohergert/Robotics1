#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

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

    cv::Mat resimage1, resimage2;
    cv::resize(image1, resimage1, cv::Size(760, 480), 0, 0);
    cv::resize(image2, resimage2, cv::Size(760, 480), 0, 0);



    
    std::cout << "Image1 loaded! Size: " << resimage1.cols << " x " << resimage1.rows << std::endl;
    std::cout << "Image1 loaded! Size: " << resimage2.cols << " x " << resimage2.rows << std::endl;

    cv::Mat combined;
    cv::hconcat(resimage1, resimage2, combined);
    cv::imshow("Previous (Left) + Current (Right)", combined);
    cv::waitKey
    (0);

    return 0;
}
