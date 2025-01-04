#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <algorithm>

class CloudFromImages
{
public:
    void calibrateZhang(std::vector<cv::Mat> chessboards, cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& distCoefs);

    void saveCameraParameters(std::string fileAddress, cv::Mat K, cv::Mat R, cv::Mat t, cv::Mat dist);
    
    void downloadCameraParameters(std::string fileAddress, cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& dist);

    std::vector<cv::Mat> undistortAll(std::vector<cv::Mat> images, cv::Mat K, cv::Mat distCoefs);

    std::vector<cv::Mat> imagesFromFolder(std::string path) {
        std::vector<std::string> imageNames;
        cv::glob(path, imageNames, false);
        std::vector<cv::Mat> images;
        for (int i = 0; i < imageNames.size(); i++) {
            cv::Mat img = cv::imread(imageNames.at(i));
            images.push_back(img);
        }
        return images;
    }


    void detectAndMatchFeatures(cv::Mat img1, cv::Mat img2,
                                std::vector<cv::Point2f>& ps1, std::vector<cv::Point2f>& ps2, cv::Mat& matchImage,
                                int numberOfFeatures = 0) {
        //Automatic feature detection on the image pair with the calibrated camera(OpenCV implementation) 5 points.
        std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> ps;

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(numberOfFeatures);
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat dsc1, dsc2;

        cv::BFMatcher matcher(cv::NORM_L2, true);
        std::vector<cv::DMatch> matches;

        sift->detectAndCompute(img1, cv::noArray(), keypoints1, dsc1);
        sift->detectAndCompute(img2, cv::noArray(), keypoints2, dsc2);

        matcher.match(dsc1, dsc2, matches);

        std::vector<cv::DMatch> matches2;
        // distance between match points
        /*float d = 0, min_d = 1e10, max_d = 0;
        for (cv::DMatch match : matches) {
            d = match.distance;
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
        }*/
        // simple filtering
        for (cv::DMatch match : matches) {
            if (matches2.size() < 8) {
                cv::Point2f p1 = cv::Point2f((int)keypoints1[match.queryIdx].pt.x, (int)keypoints1[match.queryIdx].pt.y);
                cv::Point2f p2 = cv::Point2f((int)keypoints2[match.trainIdx].pt.x, (int)keypoints2[match.trainIdx].pt.y);
                if (std::count(ps1.begin(), ps1.end(), p1) != 0
                    || std::count(ps2.begin(), ps2.end(), p2)) continue;
                matches2.push_back(match);
                ps1.push_back(p1);
                ps2.push_back(p2);
            }
            /*
            else if (match.distance <= 2 * min_d) {
                matches2.push_back(match);
                cv::Point2f p1 = cv::Point2f((int)keypoints1[match.queryIdx].pt.x, (int)keypoints1[match.queryIdx].pt.y);
                cv::Point2f p2 = cv::Point2f((int)keypoints2[match.trainIdx].pt.x, (int)keypoints2[match.trainIdx].pt.y);
                ps1.push_back(p1);
                ps2.push_back(p2);
            }*/
            else {
                int maxIndex = 0;
                for (int j = 0; j < matches2.size(); j++) 
                    if (matches2[j].distance > matches2[maxIndex].distance) 
                        maxIndex = j;
                if (match.distance < matches2[maxIndex].distance) {
                    cv::Point2f p1 = cv::Point2f((int)keypoints1[match.queryIdx].pt.x, (int)keypoints1[match.queryIdx].pt.y);
                    cv::Point2f p2 = cv::Point2f((int)keypoints2[match.trainIdx].pt.x, (int)keypoints2[match.trainIdx].pt.y);
                    if (std::count(ps1.begin(), ps1.end(), p1) != 0
                        || std::count(ps2.begin(), ps2.end(), p2)) continue;
                    matches2[maxIndex] = match;
                    ps1[maxIndex] = p1;
                    ps2[maxIndex] = p2;
                }
            }
        }
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches2, matchImage);
    }
    
    cv::Mat cameraToCameraRotation(cv::Mat r1, cv::Mat r2);
    cv::Mat cameraToCameraTranslation(cv::Mat r1, cv::Mat r2, cv::Mat t1, cv::Mat t2);

    void rectifyPair(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, 
                    cv::Mat Kleft, cv::Mat distCoefsLeft, cv::Mat Kright, cv::Mat distCoefsRight,
                    cv::Mat& img1Rect, cv::Mat& img2Rect, double& baseline) {
        //Rectify the image pair (OpenCV implementation; cv::stereoRectify() + cv::initUndistortRectifyMap() 
        //                        or cv::stereoRectifyUncalibrated() + transformation with homography) 
        // EDIT: you can get the camera baseline here) 8 points.
        cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
        cv::Mat H1, H2;
        cv::Mat E = Kright.t() * F * Kleft;
        cv::Mat R, T, mask;
        cv::recoverPose(E, points1, points2, Kleft, R, T, mask); 
        cv::Mat Rt = R * T;
        baseline = sqrt(Rt.at<double>(0, 0) * Rt.at<double>(0, 0) +
                        Rt.at<double>(1, 0) * Rt.at<double>(1, 0) +
                        Rt.at<double>(2, 0) * Rt.at<double>(2, 0));
        /*cv::Mat R1, R2, P1, P2, Q;
        cv::stereoRectify(Kleft, distCoefsLeft, Kright, distCoefsRight, img1.size(), R, T, R1, R2, P1, P2, Q);
        cv::Mat m1X, m1Y, m2X, m2Y;
        cv::initUndistortRectifyMap(Kleft, distCoefsLeft, R1, P1, img1.size(), CV_32F, m1X, m1Y);
        cv::initUndistortRectifyMap(Kright, distCoefsRight, R2, P2, img2.size(), CV_32F, m2X, m2Y);
        cv::remap(img1, img1Rect, m1X, m1Y, cv::INTER_LINEAR);
        cv::remap(img2, img2Rect, m2X, m2Y, cv::INTER_LINEAR);*/
        /**/cv::stereoRectifyUncalibrated(points1, points2, F, cv::Size(img1.cols, img1.rows), H1, H2);
        warpPerspective(img1, img1Rect, H1, img1.size());
        warpPerspective(img2, img2Rect, H2, img2.size());
    }
    cv::Mat calculateDisparityMap(cv::Mat img1, cv::Mat img2) {
        // Calculate disparity map on the rectified images (OpenCV or other library;
        // it doesn't need to be "very high quality",
        // however try your best OR calculate the disparity of the detected feature points on the rectified image) 7 points.
        cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
        int blockSize = 3;
        //cv::Ptr<cv::StereoSGBM> stereoSGBM = cv::StereoSGBM::create(0, 128, 5,
                                                                    //8 * img1.channels() * blockSize * blockSize, 32*img1.channels() * blockSize * blockSize, 
                                                                    //1, 63, 10, 100, 32, cv::StereoSGBM::MODE_HH);
        //cv::Ptr<cv::StereoSGBM> stereoSGBM = cv::StereoSGBM::create(0, 16, blockSize, // minDisparity, numDisparities, blockSize
        //                                                            8 * img1.channels() * blockSize * blockSize,  // P1
        //                                                            32 * img1.channels() * blockSize * blockSize, // P2
        //                                                            16, 16, 11, 200, 4, // disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange
        //                                                            cv::StereoSGBM::MODE_HH); // mode
        
        cv::Ptr<cv::StereoSGBM> stereoSGBM = cv::StereoSGBM::create(-192, 256, blockSize, // minDisparity, numDisparities, blockSize
                                                                    8 * img1.channels() * blockSize * blockSize,  // P1
                                                                    32 * img1.channels() * blockSize * blockSize, // P2
                                                                    32, 63, 1, 256, 64, // disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange
                                                                    cv::StereoSGBM::MODE_HH); // mode
        cv::Mat disparityMap;
        stereoSGBM->compute(img1, img2, disparityMap);
        return disparityMap;
    }

    void triangulatePoints(cv::Mat img1, cv::Mat disparityMap, cv::Mat K, double baselineLength,
                            std::vector<cv::Point3f>& points3D, std::vector<cv::Vec3b>& colors) {
        // From the disparity map triangulate the points (case of Standard Stereo; your own implementation is needed here!) 20 points.
        // we use those of the left camera because it's our main camera 
        // and the disparity map is calculated with the left as main
        baselineLength = 0.073f;
        double f = K.at<double>(0, 0);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);
        cv::Mat depth(img1.rows, img1.cols, CV_64F);
        for (int y = 0; y < disparityMap.rows; y++) {
            for (int x = 0; x < disparityMap.cols; x++) {
                // I assume that x - cx is u_1
                float d = disparityMap.at<short>(y, x) / 16.f;
                if (d > 20) {
                    cv::Vec3b color = img1.at<cv::Vec3b>(y, x);
                    if (color[0] != 0 || color[1] != 0 || color[2] != 0) {
                        cv::Point3f point(-0.5f * ((x - cx) - d + (x - cx)),
                            (y - cy),
                            -f);
                        point *= baselineLength / d;
                        points3D.push_back(point);
                        colors.push_back(color);
                    }
                }
            }
        }
    }

    void saveToPLY(std::string fileAddress, std::vector<cv::Point3f> points, std::vector<cv::Vec3b> colors) {
        // Colorize the point cloud you get after triangulation using the images and save it to PLY format. 5 points.
        std::stringstream strstr;
        strstr << "ply\nformat ascii 1.0\nelement vertex ";
        strstr << points.size();
        strstr << "\nproperty float x\nproperty float y\nproperty float z\n";
        strstr << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n ";
        for (int i = 0; i < points.size(); i++) {
            strstr << points[i].x << " " << points[i].y << " " << points[i].z << " ";
            //strstr << 255 << " " << 0 << " " << 0 << "\n";
            strstr << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << "\n";
        }
        std::ofstream outFile;
        outFile.open(fileAddress);
        outFile << strstr.rdbuf();
        outFile.close();
    }
private:
    cv::Mat compute_intrinsics(std::vector<cv::Mat> H_list);
    std::vector<cv::Point2f> get_corners(cv::Mat image, cv::Size checkerboard_size, bool fastCheck = false) {
        std::vector<cv::Point2f> corners;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        bool ret = fastCheck ? cv::findChessboardCorners(image, checkerboard_size, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH |
            cv::CALIB_CB_FAST_CHECK |
            cv::CALIB_CB_NORMALIZE_IMAGE) :
            cv::findChessboardCorners(image, checkerboard_size, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH |
                cv::CALIB_CB_NORMALIZE_IMAGE);
        if (ret) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.0001));
        }

        return corners;
    }
};

