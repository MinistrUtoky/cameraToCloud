#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "cloudFromImages.h"


int main(int argc, char** argv)
{
    CloudFromImages cfi;
    cv::Mat K, R, t, dist,
            K2, R2, t2, dist2;
    /*
    std::vector<cv::Mat> chessboards;
    std::vector<cv::Mat> chessboards2;
    chessboards = cfi.imagesFromFolder("T:\\task3\\footage\\Dev0_some_chessboard\\*.jpg");
    chessboards2 = cfi.imagesFromFolder("T:\\task3\\footage\\Dev1_some_chessboard\\*.jpg");
    cfi.calibrateZhang(chessboards, K, R, t, dist);
    cfi.calibrateZhang(chessboards2, K2, R2, t2, dist2);
    cfi.saveCameraParameters("T:\\task3\\KRt.txt", K, R, t, dist); 
    cfi.saveCameraParameters("T:\\task3\\KRt_Dev1.txt", K2, R2, t2, dist2); 
    */
    cfi.downloadCameraParameters("T:\\task3\\KRt.txt", K, R, t, dist);
    cfi.downloadCameraParameters("T:\\task3\\KRt_Dev1.txt", K2, R2, t2, dist2);
    /*
    std::vector<cv::Mat> images = cfi.imagesFromFolder("T:\\task3\\footage\\Dev0_some_footage\\*.jpg");
    std::vector<cv::Mat> images2 = cfi.imagesFromFolder("T:\\task3\\footage\\Dev1_some_footage\\*.jpg");

    images = cfi.undistortAll(images, K, dist);
    images2 = cfi.undistortAll(images2, K2, dist2);
    int minSize = images.size() > images2.size() ? images2.size() : images.size();
    for (int i = 0; i < minSize; i += 1) {
        cv::imwrite("T:\\task3\\undistortsDiv0\\" + std::to_string(i) + ".jpg", images[i]);
        cv::imwrite("T:\\task3\\undistortsDiv1\\" + std::to_string(i) + ".jpg", images2[i]);
    }
    return 0;*/
    // DIV 1 is left DIV 0 is right !!!
    std::vector<cv::Mat> images = cfi.imagesFromFolder("T:\\task3\\undistortsDiv0\\*.jpg"),
                         images2 = cfi.imagesFromFolder("T:\\task3\\undistortsDiv1\\*.jpg");
    int minSize = images.size() > images2.size() ? images2.size() : images.size();
    std::vector<cv::Mat> imagesRectifiedLeft = cfi.imagesFromFolder("T:\\task3\\rectifiedLeft\\*.jpg"),
                         imagesRectifiedRight = cfi.imagesFromFolder("T:\\task3\\rectifiedRight\\*.jpg");

    //std::vector<cv::Point3f> pointsTotal;
    double baselineLength = 0.1f;
    for (int i = 0; i < minSize; i+=1){ 
        /*
        cv::Mat matchImage; 
        cv::Mat leftImage = images2[i], rightImage = images[i];
        std::vector<cv::Point2f> ps1, ps2;
        cfi.detectAndMatchFeatures(leftImage, rightImage, ps1, ps2, matchImage);
        cv::imwrite("T:\\task3\\matched\\match" + std::to_string(i) + ".jpg", matchImage);

        cv::Mat img1Rect, img2Rect; 
        cfi.rectifyPair(leftImage, rightImage, ps1, ps2, K2, dist2, K, dist, img1Rect, img2Rect, baselineLength);
        cv::imwrite("T:\\task3\\rectifiedLeft\\" + std::to_string(i) + ".jpg", img1Rect);
        cv::imwrite("T:\\task3\\rectifiedRight\\" + std::to_string(i) + ".jpg", img2Rect);*/
        // we output left to img1Rect and right to img2Rect hence the switch
        // i just decided to use two cameras later on because i didnt like the results
        std::cout << i << std::endl;
        cv::Mat img1Rect = imagesRectifiedLeft[i];
        cv::Mat img2Rect = imagesRectifiedRight[i];
        cv::Mat disparityMap = cfi.calculateDisparityMap(img2Rect, img1Rect);
        short mn = *std::min_element(disparityMap.begin<short>(), disparityMap.end<short>());
        short mx = *std::max_element(disparityMap.begin<short>(), disparityMap.end<short>());
        cv::Mat disparityMapPositive;
        cv::normalize(disparityMap, disparityMapPositive, 0, mx - mn, cv::NORM_MINMAX, CV_16U);
        cv::Mat disparityMapNormalized;
        cv::normalize(disparityMap, disparityMapNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("T:\\task3\\disparities\\disparityMap" + std::to_string(i) + ".jpg", disparityMapNormalized);
        cv::imwrite("T:\\task3\\disparityLeftMatch\\disparityBaseLeft" + std::to_string(i) + ".jpg", img1Rect);
        cv::imwrite("T:\\task3\\disparityRightMatch\\disparityBaseRight" + std::to_string(i) + ".jpg", img2Rect);


        std::vector<cv::Point3f> points; 
        std::vector<cv::Vec3b> colors;
        cfi.triangulatePoints(img1Rect, disparityMapPositive, K2, baselineLength, points, colors);
        //pointsTotal.insert(pointsTotal.end(), points.begin(), points.end());
        cfi.saveToPLY("T:\\task3\\Scene3D" + std::to_string(i) + std::to_string(i+1) + ".ply", points, colors);
        /**/
    }
    return 0;
}
