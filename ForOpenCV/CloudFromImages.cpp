#include "CloudFromImages.h"


void CloudFromImages::calibrateZhang(std::vector<cv::Mat> chessboards, cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& distCoefs) {
    /*Camera calibration and undistortion of images (OpenCV built-in functions using chessboard) 15 points.*/
    cv::Size checkerboardSize(6, 6);
    double cellSize = 0.1;

    std::vector<cv::Point3f> worldPoints;
    for (int i = 0; i < checkerboardSize.height; ++i)
        for (int j = 0; j < checkerboardSize.width; ++j)
            worldPoints.emplace_back(j * cellSize, i * cellSize, 0.0);

    //std::vector<cv::Mat> H_list;
    std::vector<std::vector<cv::Point2f>> allCorners;
    std::vector<std::vector<cv::Point3f>> allWorldPoints;
    for (int i = 0; i < chessboards.size(); i++) {
        std::vector<cv::Point2f> corners = get_corners(chessboards[i], checkerboardSize, true);
        /*
        cv::Mat displayImage; image.copyTo(displayImage);
        cv::drawChessboardCorners(displayImage, checkerboardSize, corners, true);
        cv::resize(displayImage, displayImage, cv::Size(960, 600));
        cv::imshow("corners", displayImage);
        cv::waitKey(0);
        */
        std::cout << (i + 1) << "/" << chessboards.size() << std::endl;
        if (corners.empty()) {
            std::cerr << "Failed to detect corners in image #" << i << std::endl;
            continue;
        }
        allCorners.push_back(corners);
        allWorldPoints.push_back(worldPoints);
        //cv::Mat H = cv::findHomography(worldPoints, corners);
        //H_list.push_back(H);
    }
    /*
    K = compute_intrinsics(H_list);
    cv::Mat H0 = H_list[0];
    cv::Mat K_inv = K.inv();
    cv::Mat r1r2t = K_inv * H0;
    cv::Mat r1 = r1r2t.col(0);
    cv::Mat r2 = r1r2t.col(1);
    t = r1r2t.col(2);
    cv::Mat r3 = r1.cross(r2);
    cv::hconcat(r1, r2, R);
    cv::hconcat(R, r3, R);
    R = R.t();

    std::cout << "K:\n" << K << std::endl;
    std::cout << "R1:\n" << R << std::endl;
    std::cout << "t1:\n" << t << std::endl;
    */
    double error = cv::calibrateCamera(allWorldPoints, allCorners, cv::Size(chessboards[0].cols, chessboards[0].rows),
        K, distCoefs, R, t);
    std::cout << "K:\n" << K << std::endl;
    std::cout << "R1:\n" << R.row(0) << std::endl;
    std::cout << "t1:\n" << t.row(0) << std::endl;
    std::cout << "dst:\n" << distCoefs << std::endl;
    std::cout << "error:\n" << error << std::endl;
}

void CloudFromImages::saveCameraParameters(std::string fileAddress, cv::Mat K, cv::Mat R, cv::Mat t, cv::Mat dist) {
    std::stringstream strstr;
    for (int i = 0; i < K.rows; i++) {
        strstr << K.at<double>(i, 0) << " " << K.at<double>(i, 1) << " " << K.at<double>(i, 2) << "\n";
    }

    strstr << R.rows << "\n";
    for (int i = 0; i < R.rows; i++)
        strstr << R.at<double>(i, 0) << " " << R.at<double>(i, 1) << " " << R.at<double>(i, 2) << "\n";

    strstr << t.rows << "\n";
    for (int i = 0; i < t.rows; i++)
        strstr << t.at<double>(i, 0) << " " << t.at<double>(i, 1) << " " << t.at<double>(i, 2) << "\n";

    strstr << "\n";
    strstr << dist.at<double>(0, 0) << " " << dist.at<double>(0, 1) << " " << dist.at<double>(0, 2) << " "
        << dist.at<double>(0, 3) << " " << dist.at<double>(0, 4) << "\n";
    strstr << "\n";
    std::ofstream outFile;
    outFile.open(fileAddress);
    outFile << strstr.rdbuf();
}

void CloudFromImages::downloadCameraParameters(std::string fileAddress, cv::Mat& K, cv::Mat& R, cv::Mat& t, cv::Mat& dist)
{
    std::ifstream file;
    file.open(fileAddress);
    std::string ss;
    K = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            file >> K.at<double>(i, j);

    int rsize; file >> rsize;
    R = cv::Mat(rsize, 3, CV_64F);
    for (int i = 0; i < rsize; i++)
        for (int j = 0; j < 3; j++)
            file >> R.at<double>(i, j);

    int tsize; file >> tsize;
    t = cv::Mat(tsize, 3, CV_64F);
    for (int i = 0; i < tsize; i++)
        for (int j = 0; j < 3; j++)
            file >> t.at<double>(i, j);

    dist = cv::Mat(1, 5, CV_64F);
    for (int i = 0; i < 5; i++)
        file >> dist.at<double>(0, i);
    file.close();
}

std::vector<cv::Mat> CloudFromImages::undistortAll(std::vector<cv::Mat> images, cv::Mat K, cv::Mat distCoefs) {
    std::vector<cv::Mat> undistortedImages;
    for (int i = 0; i < images.size(); i++) {
        cv::Mat undistortedImage; cv::Mat outK;
        cv::undistort(images[i], undistortedImage, K, distCoefs);
        undistortedImages.push_back(undistortedImage);
    }
    return undistortedImages;
}

cv::Mat CloudFromImages::cameraToCameraRotation(cv::Mat r1, cv::Mat r2) {
    cv::Mat R1, R2;
    cv::Rodrigues(r1, R1);
    cv::Rodrigues(r2, R2);
    return R1.t() * R2;
}
cv::Mat CloudFromImages::cameraToCameraTranslation(cv::Mat r1, cv::Mat r2, cv::Mat t1, cv::Mat t2) {
    cv::Mat R1, R2;
    cv::Rodrigues(r1, R1);
    cv::Rodrigues(r2, R2);
    t1 = t1.reshape(1, 3);
    t2 = t2.reshape(1, 3);
    return R1.t() * t2 - R1.t() * t1;
}

cv::Mat CloudFromImages::compute_intrinsics(std::vector<cv::Mat> H_list) {
    cv::Mat A(H_list.size() * 2, 6, CV_64F);
    for (int i = 0; i < H_list.size(); i++) {
        cv::Mat H = H_list.at(i);
        A.at<double>(2 * i, 0) = H.at<double>(0, 0) * H.at<double>(0, 1);
        A.at<double>(2 * i, 1) = H.at<double>(0, 0) * H.at<double>(1, 1) + H.at<double>(1, 0) * H.at<double>(0, 1);
        A.at<double>(2 * i, 2) = H.at<double>(1, 0) * H.at<double>(1, 1);
        A.at<double>(2 * i, 3) = H.at<double>(2, 0) * H.at<double>(0, 1) + H.at<double>(0, 0) * H.at<double>(2, 1);
        A.at<double>(2 * i, 4) = H.at<double>(2, 0) * H.at<double>(1, 1) + H.at<double>(1, 0) * H.at<double>(2, 1);
        A.at<double>(2 * i, 5) = H.at<double>(2, 0) * H.at<double>(2, 1);

        A.at<double>(2 * i + 1, 0) = H.at<double>(0, 0) * H.at<double>(0, 0) - H.at<double>(0, 1) * H.at<double>(0, 1);
        A.at<double>(2 * i + 1, 1) = 2 * (H.at<double>(0, 0) * H.at<double>(1, 0) - H.at<double>(0, 1) * H.at<double>(1, 1));
        A.at<double>(2 * i + 1, 2) = H.at<double>(1, 0) * H.at<double>(1, 0) - H.at<double>(1, 1) * H.at<double>(1, 1);
        A.at<double>(2 * i + 1, 3) = 2 * (H.at<double>(2, 0) * H.at<double>(0, 0) - H.at<double>(2, 1) * H.at<double>(0, 1));
        A.at<double>(2 * i + 1, 4) = 2 * (H.at<double>(2, 0) * H.at<double>(1, 0) - H.at<double>(2, 1) * H.at<double>(1, 1));
        A.at<double>(2 * i + 1, 5) = H.at<double>(2, 0) * H.at<double>(2, 0) - H.at<double>(2, 1) * H.at<double>(2, 1);
    }

    cv::Mat b;
    cv::SVD::solveZ(A, b);
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
