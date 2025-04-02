#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
int dummyCheck = (std::cout << "🔥 I AM RUNNING THIS BUILD 🔥\n", 0);
int main() {
    std::cout << "Program started!" << std::endl;
    std::string imagePath = "200_images/000000.jpg";
    std::cout << "Trying to load image: " << imagePath << std::endl;
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Failed to load image at " << imagePath << std::endl;
        system("pause");
        return -1;
    }

    std::cout << "Image loaded successfully! Size: "
              << image.cols << "x" << image.rows << std::endl;

    cv::imshow("Test Image", image);
    cv::waitKey(0);

    system("pause"); // ✅ pause to keep the console open
    return 0;
}
