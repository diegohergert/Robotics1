#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>

int main() {
    std::cout << "🚀 OpenCV image test started!" << std::endl;

    std::string imagePath = "200_images/000000.png";  // Make sure this image exists
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "❌ Failed to load image at " << imagePath << std::endl;
        system("pause");
        return -1;
    }

    std::cout << "✅ Image loaded! Size: " << image.cols << " x " << image.rows << std::endl;

    cv::imshow("Test Image", image);
    cv::waitKey(0);

    return 0;
}
