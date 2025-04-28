#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

// Create a SIFT detector
cv::Ptr<cv::SIFT> sift = cv::SIFT::create(5000);

// Camera intrinsic matrix
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
        std::cerr << "Error: Could not load image: " << ss.str() << std::endl;
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

// Filter 3D points by removing outliers based on distance from median
std::vector<cv::Point3f> filterPointCloud(const std::vector<cv::Point3f>& points, double threshold = 5.0) {
    if (points.size() < 4) return points;
    
    // Calculate median depth
    std::vector<float> depths;
    for (const auto& pt : points) {
        depths.push_back(pt.z);
    }
    std::sort(depths.begin(), depths.end());
    float medianDepth = depths[depths.size() / 2];
    
    // Filter points that are too far from median depth
    std::vector<cv::Point3f> filteredPoints;
    for (const auto& pt : points) {
        if (std::abs(pt.z - medianDepth) < threshold) {
            filteredPoints.push_back(pt);
        }
    }
    return filteredPoints;
}

void drawPointCloudView(
    cv::Mat& view,
    const std::vector<cv::Point3f>& points,
    const cv::Mat& globalR,
    const cv::Mat& globalT,
    float f = 300.0f,
    cv::Point2f center = cv::Point2f(300, 300))
{
    view = cv::Mat::zeros(600, 600, CV_8UC3);
    
    for (const auto& point : points) {
        // Transform to camera coordinates
        cv::Mat ptWorld = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
        cv::Mat ptCam = globalR.t() * (ptWorld - globalT);

        double X = ptCam.at<double>(0, 0);
        double Y = ptCam.at<double>(1, 0);
        double Z = ptCam.at<double>(2, 0);

        if (Z <= .1) continue; // Only points in front of camera

        float u = static_cast<float>(f * X / Z + center.x);
        float v = static_cast<float>(f * Y / Z + center.y);
        
        if (u >= 0 && u < view.cols && v >= 0 && v < view.rows) {
            int size = std::min(5, std::max(1, static_cast<int>(5.0f / Z)));
            int colorVal = std::min(255, std::max(0, static_cast<int>(128 + Y * 20)));
            cv::circle(view, cv::Point(u, v), size, cv::Scalar(0, colorVal, 255 - colorVal), -1);
        }
    }

    cv::putText(view, "Camera Perspective View", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}



void drawCameraPathOnlyView(
    cv::Mat& pathView, 
    const std::vector<cv::Point3f>& cameraPath,
    float scale = 5.0f,
    cv::Point2f offset = cv::Point2f(300, 300)) 
{
    pathView = cv::Mat::zeros(600, 600, CV_8UC3);

    if (cameraPath.size() > 1) {
        for (size_t i = 1; i < cameraPath.size(); ++i) {
            int x1 = static_cast<int>(cameraPath[i-1].x * scale + offset.x);
            int y1 = static_cast<int>(offset.y - cameraPath[i-1].z * scale);
            int x2 = static_cast<int>(cameraPath[i].x * scale + offset.x);
            int y2 = static_cast<int>(offset.y - cameraPath[i].z * scale);
            
            if (x1 >= 0 && x1 < pathView.cols && y1 >= 0 && y1 < pathView.rows &&
                x2 >= 0 && x2 < pathView.cols && y2 >= 0 && y2 < pathView.rows) {
                cv::line(pathView, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
            }
        }
        
        // Mark current camera position
        int currX = static_cast<int>(-(cameraPath.back().x * scale + offset.x));
        int currY = static_cast<int>(-(offset.y - cameraPath.back().z * scale));
        if (currX >= 0 && currX < pathView.cols && currY >= 0 && currY < pathView.rows) {
            cv::circle(pathView, cv::Point(currX, currY), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    
    cv::putText(pathView, "Camera Trajectory", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}


int main() {
    std::cout << "OpenCV image test started!" << std::endl;

    const std::string basePath = "200_images/";
    const int totalFrames = 201;
    const cv::Size imageSize(760, 480);

    cv::VideoWriter pathWriter("trajectory_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, cv::Size(600, 600));
    if (!pathWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }
    cv::VideoWriter cloudWriter("pointcloud_result.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30, cv::Size(600, 600));
    if (!cloudWriter.isOpened()) {
        std::cerr << "Error: Could not open the video writer." << std::endl;
        return -1;
    }

    // Initialize global camera pose
    cv::Mat globalR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat globalT = cv::Mat::zeros(3, 1, CV_64F);

    // Global point cloud and camera path storage
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point3f> cameraPath;
    
    // Add initial camera position to path
    cameraPath.push_back(cv::Point3f(0, 0, 0));

    // Load the first image
    cv::Mat prevImg = loadImage(0, basePath);
    if (prevImg.empty()) return -1;
    cv::resize(prevImg, prevImg, imageSize);
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Mat prevDescriptors;
    detectAndCompute(prevImg, prevKeypoints, prevDescriptors);

    // Create window for visualization
    cv::namedWindow("Point Cloud", cv::WINDOW_NORMAL);
    cv::resizeWindow("Point Cloud", 600, 600);

    // Loop over all frames
    for (int i = 1; i < totalFrames; ++i) {
        std::cout << "Processing frame: " << i << std::endl;
        cv::Mat currImg = loadImage(i, basePath);
        if (currImg.empty()) continue;
        cv::resize(currImg, currImg, imageSize);

        std::vector<cv::KeyPoint> currKeypoints;
        cv::Mat currDescriptors;
        detectAndCompute(currImg, currKeypoints, currDescriptors);
        if (currDescriptors.empty() || currKeypoints.size() < 8) {
            std::cout << "Not enough keypoints detected. Skipping frame." << std::endl;
            continue;
        }

        auto matches = matchFeatures(prevDescriptors, currDescriptors);
        if (matches.size() < 8) {
            std::cout << "Not enough matches found. Skipping frame." << std::endl;
            continue;
        }
        
        auto inlierMatches = filterMatchesRANSAC(prevKeypoints, currKeypoints, matches);
        if (inlierMatches.size() < 8) {
            std::cout << "Not enough inliers after RANSAC. Skipping frame." << std::endl;
            continue;
        }

        // Extract points from the matches
        std::vector<cv::Point2f> prevPoints, currPoints;
        for (const auto& match : inlierMatches) {
            prevPoints.push_back(prevKeypoints[match.queryIdx].pt);
            currPoints.push_back(currKeypoints[match.trainIdx].pt);
        }
        
        // Compute the essential matrix directly (skip the fundamental matrix step)
        cv::Mat E, R, t;
        E = cv::findEssentialMat(prevPoints, currPoints, K, cv::RANSAC);
        // Recover pose between views
        int inlierCount = cv::recoverPose(E, prevPoints, currPoints, K, R, t);
        
        

        // Scale translation (since SfM is up to scale)
        double scale = .1;
        t = t * scale;
        
        // Store previous global pose
        cv::Mat prevGlobalR = globalR.clone();
        cv::Mat prevGlobalT = globalT.clone();
        
        // Correct global pose update
        globalR = R * prevGlobalR;
        globalT = R * prevGlobalT + t;

        
        // Add current camera position to path (inverse of pose)
        cv::Mat currCamPos = globalR.t() * globalT;
        cameraPath.push_back(cv::Point3f(
            currCamPos.at<double>(0, 0),
            currCamPos.at<double>(1, 0),
            currCamPos.at<double>(2, 0)
        ));

        // Define projection matrices for triangulation
        // We use P1 as identity in camera coordinates
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
        
        // First camera is at identity pose (current view reference frame)
        cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
        I.copyTo(P1(cv::Rect(0, 0, 3, 3)));
        P1 = K * P1;
        
        // Second camera is at relative pose R,t
        R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
        t.copyTo(P2(cv::Rect(3, 0, 1, 3)));
        P2 = K * P2;

        // Triangulate points
        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, prevPoints, currPoints, points4D);
        
        // Convert homogeneous coordinates to 3D points and store in global point cloud
        std::vector<cv::Point3f> newPoints;
        for (int j = 0; j < points4D.cols; ++j) {
            float w = points4D.at<float>(3, j);
            if (std::abs(w) > 1e-8) { // Avoid divide by zero
                cv::Point3f point(
                    points4D.at<float>(0, j) / w,
                    points4D.at<float>(1, j) / w,
                    points4D.at<float>(2, j) / w
                );
                
                // Transform point to global coordinate frame
                cv::Mat pt3d = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
                cv::Mat pt3dGlobal = prevGlobalR.t() * (pt3d - prevGlobalT);
                
                cv::Point3f pointGlobal(
                    pt3dGlobal.at<double>(0, 0),
                    pt3dGlobal.at<double>(1, 0),
                    pt3dGlobal.at<double>(2, 0)
                );
                
                // Only add points with positive depth and within a reasonable range
                if (pointGlobal.z > 0 && pointGlobal.z < 30) {
                    newPoints.push_back(pointGlobal);
                }
            }
        }
        
        // Filter point cloud to remove outliers
        if (!newPoints.empty()) {
            newPoints = filterPointCloud(newPoints);
            
            // Add filtered points to global point cloud
            points3D.insert(points3D.end(), newPoints.begin(), newPoints.end());
            
            // Limit the number of points to avoid memory issues
            const size_t maxPoints = 100000;
            if (points3D.size() > maxPoints) {
                points3D.erase(points3D.begin(), points3D.begin() + (points3D.size() - maxPoints));
            }
        }

        // Draw point cloud and camera path
        cv::Mat cloud;
        drawPointCloudView(cloud, points3D, globalR, globalT);

        cv::Mat pathView;
        drawCameraPathOnlyView(pathView, cameraPath);
        
        // Add frame number
        cv::putText(cloud, "Frame: " + std::to_string(i) + " | Points: " + std::to_string(points3D.size()), 
                cv::Point(10, 570), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        cv::putText(pathView, "Frame: " + std::to_string(i),
                cv::Point(10, 570), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Display and save
        cv::imshow("Point Cloud", cloud);
        cloudWriter.write(cloud);
        pathWriter.write(pathView);


        cv::waitKey(1);

        // Update for next iteration
        prevImg = currImg.clone();
        prevKeypoints = currKeypoints;
        prevDescriptors = currDescriptors.clone();
    }

    cloudWriter.release();
    pathWriter.release();
    cv::destroyAllWindows();
    std::cout << "Point cloud video saved as pointcloud_result.mp4" << std::endl;
    return 0;
}