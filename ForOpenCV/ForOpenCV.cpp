#include "CameraCalibration.h"

void ZhangIt() {
    CameraCalibration cc;
    cv::Size checkerboardSize(6, 6); 
    double cellSize = 0.1;

    std::vector<cv::Point3f> worldPoints;
    for (int i = 0; i < checkerboardSize.height; ++i)
        for (int j = 0; j < checkerboardSize.width; ++j)
            worldPoints.emplace_back(j * cellSize, i * cellSize, 0.0);

    std::vector<cv::Mat> H_list;
    for (int i = 1; i <= 22; ++i) {
        //std::string imagePath = "T:/files/calib_images/imgs/rightcamera/Im_R_" + std::to_string(i) + ".png";
        std::string imagePath = "T:/files/calib_images/imgs/cameracamera/" + std::to_string(i) + ".jpg";
        cv::Mat image = cv::imread(imagePath);
        //cv::imshow("bruh", image);
        //cv::waitKey(0);
        std::vector<cv::Point2f> corners;

        corners = cc.get_corners(image, checkerboardSize, true);
        if (corners.empty()) {
            std::cerr << "Failed to detect corners in image: " << imagePath << std::endl;
            continue;
        }
        /*
        for (int a = 6; a < 10; a++) {
            for (int b = 6; b < 8; b++) {
                checkerboardSize = cv::Size(a, b);
                corners = cc.get_corners(image, checkerboardSize);
                std::cout << a << " " << b << std::endl;
                if (corners.empty()) {
                    std::cerr << "Failed to detect corners in image: " << imagePath << std::endl;
                    continue;
                }
            }
        }*/

        cv::Mat H = cv::findHomography(worldPoints, corners);
        H_list.push_back(H);
    }
    std::cout << H_list.size() << std::endl;
    
    cv::Mat K = cc.compute_intrinsics(H_list);
    cv::Mat H0 = H_list[0];
    cv::Mat K_inv = K.inv();

    cv::Mat r1r2t = K_inv * H0;
    cv::Mat r1 = r1r2t.col(0);
    cv::Mat r2 = r1r2t.col(1);
    cv::Mat t = r1r2t.col(2);
    cv::Mat r3 = r1.cross(r2);
    cv::Mat R; 
    cv::hconcat(r1, r2, R);
    cv::hconcat(R, r3, R);
    R = R.t();

    std::cout << "K:\n" << K << std::endl;
    std::cout << "R:\n" << R << std::endl;
    std::cout << "t:\n" << t << std::endl;
}

void Objection() {
    std::string imagePath = "T:/files/calib_images/imgs/Img2.jpg";
    cv::Mat image = cv::imread(imagePath);
    CameraCalibration cc;
    std::vector<cv::Point3f> X; std::vector<cv::Point2f> U;
    cc.get_XU(image, X, U);

    /*
    cv::Mat displayImage;
    image.copyTo(displayImage);
    for (int i = 0; i < U.size(); i++) {
        cv::circle(displayImage, U.at(i), 10, cv::Scalar(0, 0, 255), -1);
    }
    cv::resize(displayImage, displayImage, cv::Size(390, 520));
    cv::imshow("Point show", displayImage);
    cv::waitKey(0);*/

    int num_points = X.size();
    cv::Mat A = cv::Mat::zeros(num_points * 2, 12, CV_32F);

    float x, y, z, u, v;
    for (int idx = 0; idx < num_points; idx++) {
        x = X[idx].x; y = X[idx].y; z = X[idx].z;
        u = U[idx].x; v = U[idx].y;
            
        A.at<float>(2 * idx, 0) = x; A.at<float>(2 * idx, 1) = y;  
        A.at<float>(2 * idx, 2) = z;  A.at<float>(2 * idx, 3) = 1;
        A.at<float>(2 * idx, 8) = -u * x; A.at<float>(2 * idx, 9) = -u * y; 
        A.at<float>(2 * idx, 10) = -u * z; A.at<float>(2 * idx, 11) = -u;

        A.at<float>(2 * idx + 1, 4) = x; A.at<float>(2 * idx + 1, 5) = y; 
        A.at<float>(2 * idx + 1, 6) = z; A.at<float>(2 * idx + 1, 7) = 1;
        A.at<float>(2 * idx + 1, 8) = -v * x; A.at<float>(2 * idx + 1, 9) = -v * y; 
        A.at<float>(2 * idx + 1, 10) = -v * z; A.at<float>(2 * idx + 1, 11) = -v;
    }
    cv::Mat eigenvalues, eigenvectors;
    eigen(A.t() * A, eigenvalues, eigenvectors);
    cv::Mat P = eigenvectors.row(eigenvectors.rows - 1).reshape(1, 3);
    P = P * 1 / P.at<float>(2, 3);

    cv::Mat X_homogeneous(X.size(), 4, CV_32F);
    for (size_t i = 0; i < X.size(); ++i) {
        X_homogeneous.at<float>(i, 0) = X[i].x;
        X_homogeneous.at<float>(i, 1) = X[i].y;
        X_homogeneous.at<float>(i, 2) = X[i].z;
        X_homogeneous.at<float>(i, 3) = 1.0;
    }

    P.convertTo(P, CV_32F);
    X_homogeneous.convertTo(X_homogeneous, CV_32F);
    cv::Mat U_projected_homogeneous = P*X_homogeneous.t();
    U_projected_homogeneous = U_projected_homogeneous.t();

    float error = 0.0, u_proj, v_proj;
    for (int i = 0; i < U_projected_homogeneous.rows; ++i) {
        u_proj = U_projected_homogeneous.at<float>(i, 0) / U_projected_homogeneous.at<float>(i, 2);
        v_proj = U_projected_homogeneous.at<float>(i, 1) / U_projected_homogeneous.at<float>(i, 2);
        u = U[i].x;
        v = U[i].y;
        error += sqrt(pow(u - u_proj, 2) + pow(v - v_proj, 2));
    }
    error /= U.size();

    std::cout << "Reprojection error: " << error << std::endl;

    cv::Mat K, R, t;
    cv::Mat M = P(cv::Range(0, 3), cv::Range(0, 3));
    RQDecomp3x3(M, K, R);
    K *= -1; R *= -1; K.col(0) *= -1; R.row(0) *= -1;
    t = K.inv() * P.col(3);
    cv::Mat P_reconstructed;
    hconcat(K * R, K * t, P_reconstructed);

    std::cout << "K:\n" << K << std::endl;
    std::cout << "R:\n" << R << std::endl;
    std::cout << "t:\n" << t << std::endl;
}

int main()
{
    Objection();
    ZhangIt();
    return 0;
}
