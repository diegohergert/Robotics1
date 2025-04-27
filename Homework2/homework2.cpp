#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "Program started." << std::endl;
    // Set your image path here
    std::string imagePath = "C:/Users/diego/Downloads/dataset/sequences/00/image_2/000000.png";
    std::cout << "Trying to load image: " << imagePath << std::endl;
    // Load the image
    cv::Mat image = cv::imread(imagePath);

    // Check if the image is loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return -1;
    }

    // Create a window
    cv::namedWindow("Test Image", cv::WINDOW_NORMAL);

    // Show the image
    cv::imshow("Test Image", image);

    // Wait for a key press indefinitely
    cv::waitKey(0);

    return 0;
}