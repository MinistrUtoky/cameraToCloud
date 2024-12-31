#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>

class CameraCalibration
{
public:
    cv::Mat compute_intrinsics(std::vector<cv::Mat> H_list) {
        cv::Mat V(H_list.size()*2, 6, CV_64F);
        for (int i = 0; i < H_list.size(); i++) {
            cv::Mat H = H_list.at(i);
            V.at<double>(2 * i, 0) = H.at<double>(0, 0) * H.at<double>(0, 1);
            V.at<double>(2 * i, 1) = H.at<double>(0, 0) * H.at<double>(1, 1) + H.at<double>(1, 0) * H.at<double>(0, 1);
            V.at<double>(2 * i, 2) = H.at<double>(1, 0) * H.at<double>(1, 1);
            V.at<double>(2 * i, 3) = H.at<double>(2, 0) * H.at<double>(0, 1) + H.at<double>(0, 0) * H.at<double>(2, 1);
            V.at<double>(2 * i, 4) = H.at<double>(2, 0) * H.at<double>(1, 1) + H.at<double>(1, 0) * H.at<double>(2, 1);
            V.at<double>(2 * i, 5) = H.at<double>(2, 0) * H.at<double>(2, 1);

            V.at<double>(2 * i+1, 0) = H.at<double>(0, 0) * H.at<double>(0, 0) - H.at<double>(0, 1) * H.at<double>(0, 1);
            V.at<double>(2 * i+1, 1) = 2 * (H.at<double>(0, 0) * H.at<double>(1, 0) - H.at<double>(0, 1) * H.at<double>(1, 1));
            V.at<double>(2 * i+1, 2) = H.at<double>(1, 0) * H.at<double>(1, 0) - H.at<double>(1, 1) * H.at<double>(1, 1);
            V.at<double>(2 * i+1, 3) = 2 * (H.at<double>(2, 0) * H.at<double>(0, 0) - H.at<double>(2, 1) * H.at<double>(0, 1));
            V.at<double>(2 * i+1, 4) = 2 * (H.at<double>(2, 0) * H.at<double>(1, 0) - H.at<double>(2, 1) * H.at<double>(1, 1));
            V.at<double>(2 * i+1, 5) = H.at<double>(2, 0) * H.at<double>(2, 0) - H.at<double>(2, 1) * H.at<double>(2, 1);
        }

        cv::Mat b;
        cv::SVD::solveZ(V, b);
        double v0 = (b.at<double>(1, 0) * b.at<double>(3, 0) - b.at<double>(0, 0) * b.at<double>(4, 0)) / (b.at<double>(0, 0) * b.at<double>(2, 0) - b.at<double>(1, 0) * b.at<double>(1, 0));
        double lambda = b.at<double>(5, 0) - (b.at<double>(3, 0) * b.at<double>(3, 0) + v0 * (b.at<double>(1, 0) * b.at<double>(3, 0) - b.at<double>(0, 0) * b.at<double>(4, 0))) / b.at<double>(0, 0);
        double alpha = std::sqrt(lambda / b.at<double>(0, 0));
        double beta = std::sqrt(lambda * b.at<double>(0, 0) / (b.at<double>(0, 0) * b.at<double>(2, 0) - b.at<double>(1, 0) * b.at<double>(1, 0)));
        double gamma = -b.at<double>(1, 0) * alpha * alpha * beta / lambda;
        double u0 = gamma * v0 / beta - b.at<double>(3, 0) * alpha * alpha / lambda;

        cv::Mat K = (cv::Mat_<double>(3, 3) <<
            alpha, gamma, u0,
            0, beta, v0,
            0, 0, 1);
        return K;
    }

    std::vector<cv::Point2f> get_corners(cv::Mat image, cv::Size checkerboard_size, bool fastCheck=false) {
        std::vector<cv::Point2f> corners;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        bool ret = fastCheck? cv::findChessboardCorners(image, checkerboard_size, corners,
                                                        cv::CALIB_CB_ADAPTIVE_THRESH |
                                                        cv::CALIB_CB_FAST_CHECK | 
                                                        cv::CALIB_CB_NORMALIZE_IMAGE) :
                               cv::findChessboardCorners(image, checkerboard_size, corners,
                                                        cv::CALIB_CB_ADAPTIVE_THRESH |
                                                        //cv::CALIB_CB_FAST_CHECK | 
                                                        cv::CALIB_CB_NORMALIZE_IMAGE);
        if (ret) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        }

        return corners;
    }

    void get_XU(cv::Mat image, std::vector<cv::Point3f>& X, std::vector<cv::Point2f>& U) {
        float h = image.rows, w = image.cols;
        int second_half_shift = int(w * 17 / 32);
        cv::Mat left = image(cv::Range::all(), cv::Range(0, second_half_shift)), 
                right = image(cv::Range::all(), cv::Range(second_half_shift, w));


        cv::Size checkerboardSize(3, 3);
        std::vector<cv::Point2f> U1, U2;
        std::pair<std::vector<cv::Point2f>, cv::Size> U1s, U2s;
        for (int i = 3; i < 11; i++)
            for (int j = 3; j < 11; j++) {
                checkerboardSize = cv::Size(i, j);
                std::vector<cv::Point2f> U1 = get_corners(left, checkerboardSize);
                if (U1.size() > 0) {
                    std::vector<cv::Point2f> U2 = get_corners(right, checkerboardSize);
                    if (U2.size() > 0) {
                        U1s.first = U1; U1s.second = checkerboardSize;
                        U2s.first = U2; U2s.second = checkerboardSize;

                    }
                }
            }
        U1 = U1s.first;
        U2 = U2s.first;
        for (cv::Point2f& point : U2)
            point.x += second_half_shift;


        float cell_size = 0.1;
        int grid_width1 = U1s.second.width, grid_height1 = U1s.second.height;
        int grid_width2 = U2s.second.width, grid_height2 = U2s.second.height;

        std::vector<cv::Point3f> X1, X2;
        for (int y = 0; y < grid_height1; ++y)
            for (int x = 0; x < grid_width1; ++x)
                X1.emplace_back((grid_width1 - x) * cell_size,
                                (grid_height1 - y) * cell_size,
                                0.0);

        for (int y = 0; y < grid_height2; ++y)
            for (int x = 0; x < grid_width2; ++x)
                X2.emplace_back(0.0,
                                (grid_height2 - y) * cell_size,
                                (x + 1) * cell_size);

        X = X1; X.insert(X.end(), X2.begin(), X2.end());
        U = U1; U.insert(U.end(), U2.begin(), U2.end());
        for (cv::Point3f& point : X)
            point.y = -point.y;
    }
};

