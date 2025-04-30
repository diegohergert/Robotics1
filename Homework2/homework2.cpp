#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

// Camera intrinsic matrix
cv::Mat K = (cv::Mat_<double>(3, 3) << 
    7.070493e02, 0,           6.040814e02, 
    0,           7.070493e02, 1.805066e02, 
    0,           0,           1);

// Function to plot the 3D point cloud and trajectory
cv::Mat plot3DTrajectory(const std::vector<cv::Point3f>& trajectory,
                         const std::vector<cv::Point3f>& globalCloud,
                         const cv::Size& plotSize)
{
    cv::Mat plot = cv::Mat::zeros(plotSize, CV_8UC3);
    float scale = .8f;
    cv::Point2f center(plotSize.width/2, plotSize.height/2 + 120);

    // draw point cloud
    for (const auto& pt : globalCloud) {
        
        cv::Point2f p2(
            center.x + pt.x * scale,
            center.y - pt.z * scale   // invert Y for display
        );
        if (p2.x>=0 && p2.x<plotSize.width && p2.y>=0 && p2.y<plotSize.height)
            cv::circle(plot, p2, 1, cv::Scalar(0,100,0), -1);
    }

    // draw trajectory
    if (trajectory.size() > 1) {
        for (size_t i = 1; i < trajectory.size(); ++i) {
            cv::Point2f p1(
                center.x + trajectory[i-1].x * scale,
                center.y - trajectory[i-1].z * scale
            );
            cv::Point2f p2(
                center.x + trajectory[i].x * scale,
                center.y - trajectory[i].z * scale
            );
            cv::line(plot, p1, p2, cv::Scalar(255,255,0), 2);
        }
        // current camera pos
        cv::Point2f cur(
            center.x + trajectory.back().x * scale,
            center.y - trajectory.back().z * scale
        );
        cv::circle(plot, cur, 5, cv::Scalar(255,255,0), -1);
    }

    return plot;
}

int main() {
    std::cout << "OpenCV 3D Cloud & Trajectory demo\n";

    const std::string imagePath = "200_images/";
    const int frames = 201;
    const cv::Size imageSize(1500, 400);

    cv::namedWindow("3D Cloud & Trajectory", cv::WINDOW_NORMAL);
    cv::resizeWindow("3D Cloud & Trajectory", 1500, 800);

    cv::VideoWriter writer(
        "tracking_result.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        12,
        cv::Size(imageSize.width, imageSize.height*2)
    );
    if (!writer.isOpened()) {
        std::cerr << "Could not open video writer\n";
        return -1;
    }

    // SIFT setup
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(3000, 3.1, .05, 12, 2);

    // read first frame
    std::stringstream ss;
    ss << imagePath << std::setw(6) << std::setfill('0') << 0 << ".png";
    cv::Mat prevImg = cv::imread(ss.str());
    if (prevImg.empty()) {
        std::cerr << "Error reading first image\n";
        return -1;
    }
    cv::resize(prevImg, prevImg, imageSize);

    std::vector<cv::KeyPoint> prevKp;
    cv::Mat prevDesc;
    sift->detectAndCompute(prevImg, cv::noArray(), prevKp, prevDesc);

    // initial trajectory & map
    std::vector<cv::Point3f> trajectory;
    trajectory.emplace_back(0,0,0);

    cv::Mat globalPose = cv::Mat::eye(4,4,CV_64F);
    std::vector<cv::Point3f> globalCloud;

    // initial display
    cv::Mat plot = plot3DTrajectory(trajectory, globalCloud, imageSize);
    cv::Mat combined;
    cv::vconcat(prevImg, plot, combined);
    cv::imshow("3D Cloud & Trajectory", combined);
    writer.write(combined);

    // process all frames
    for (int i = 1; i < frames; ++i) {
        std::stringstream ssCurr;
        ssCurr << imagePath << std::setw(6) << std::setfill('0') << i << ".png";
        cv::Mat currImg = cv::imread(ssCurr.str());
        if (currImg.empty()) {
            std::cerr << "Skipping frame " << i << "\n";
            continue;
        }
        cv::resize(currImg, currImg, imageSize);

        // detect features
        std::vector<cv::KeyPoint> currKp;
        cv::Mat currDesc;
        sift->detectAndCompute(currImg, cv::noArray(), currKp, currDesc);

        // match
        auto matcher = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(prevDesc, currDesc, knn, 2);

        std::vector<cv::DMatch> good;
        for (auto &v : knn) {
            if (v.size()>=2 && v[0].distance < 0.55*v[1].distance && v[0].distance < 150)
                good.push_back(v[0]);
        }

        // extract pts
        std::vector<cv::Point2f> pts1, pts2;
        for (auto &m : good) {
            pts1.push_back(prevKp[m.queryIdx].pt);
            pts2.push_back(currKp[m.trainIdx].pt);
        }

        // essential + R|t
        cv::Mat inMask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.99999, 1.0, inMask);
        cv::Mat R, t;
        cv::recoverPose(E, pts1, pts2, K, R, t, inMask);

        // build 4x4 relative-pose
        cv::Mat T = cv::Mat::eye(4,4,CV_64F);
        R.copyTo(T(cv::Rect(0,0,3,3)));
        t.copyTo(T(cv::Rect(3,0,1,3)));

        // update globalPose: chain inverse so it stays camera→world
        cv::Mat posePrev = globalPose.clone();
        globalPose = posePrev * T.inv();

        // update trajectory
        trajectory.emplace_back(
            float(globalPose.at<double>(0,3)),
            float(globalPose.at<double>(1,3)),
            float(globalPose.at<double>(2,3))
        );

        // triangulate in prev frame coords
        cv::Mat P1 = K * cv::Mat::eye(3,4,CV_64F);
        cv::Mat P2 = K * T(cv::Rect(0,0,4,3));
        cv::Mat pts4D;
        cv::triangulatePoints(P1, P2, pts1, pts2, pts4D);

        std::vector<cv::Point3f> tmpPts;
        cv::convertPointsFromHomogeneous(pts4D.t(), tmpPts);

        // map each into world via posePrev
        for (auto &p : tmpPts) {
            cv::Mat ph = (cv::Mat_<double>(4,1) << p.x, p.y, p.z, 1.0);
            cv::Mat pw = posePrev * ph;
            double w = pw.at<double>(3,0);
            if (w > 1e-6) {
                globalCloud.emplace_back(
                    float(pw.at<double>(0,0)/w),
                    float(pw.at<double>(1,0)/w),
                    float(pw.at<double>(2,0)/w)
                );
            }
        }

        // draw inliers
        for (int k = 0; k < (int)good.size(); ++k) {
            if (inMask.at<uchar>(k))
                cv::circle(currImg, currKp[good[k].trainIdx].pt, 4, cv::Scalar(0,255,0), -1);
        }

        // redraw plot
        plot = plot3DTrajectory(trajectory, globalCloud, imageSize);
        cv::vconcat(currImg, plot, combined);
        cv::imshow("3D Cloud & Trajectory", combined);
        cv::waitKey(1);
        writer.write(combined);

        // swap for next iteration
        prevImg = currImg.clone();
        prevKp = currKp;
        prevDesc = currDesc.clone();

        std::cout << "Processed frame " << i << "\n";
    }

    writer.release();
    std::cout << "Done! Video saved as tracking_result.mp4\n";
    return 0;
}
