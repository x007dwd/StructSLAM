/***
 * 本程序测试在EUROC数据集上双目特征提取部分算法
 */

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <common/include/ygz/Feature.h>

#include "ygz/Frame.h"
#include "ygz/Settings.h"
#include "ygz/ORBExtractor.h"
#include "ygz/EurocReader.h"

using namespace std;
using namespace ygz;


// 路径
string leftFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam0/data";
string rightFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam1/data";
string timeFolder = "./examples/EuRoC_TimeStamps/MH01.txt";
string configFile = "./examples/EuRoC.yaml";
string imuFolder = "/home/xiang/dataset/euroc/MH_01_easy/imu0/data.csv";

// string leftFolder = "/Users/gaoxiang/dataset/euroc/MH_01_easy/cam0/data";
// string rightFolder = "/Users/gaoxiang/dataset/euroc/MH_01_easy/cam1/data";
// string imuFolder = "/Users/gaoxiang/dataset/euroc/MH_01_easy/imu0/data.csv";
// string timeFolder = "./examples/EuRoC_TimeStamps/MH01.txt";
// string configFile = "./examples/EuRoC.yaml";

int main(int argc, char **argv) {
    LOG(INFO) << "test feature extraction step in EuRoC" << endl;

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    VecIMU vimus;

    LoadImages(leftFolder, rightFolder, timeFolder, vstrImageLeft, vstrImageRight, vTimeStamp);
    LoadImus(imuFolder, vimus);

    if (vstrImageLeft.empty() || vstrImageRight.empty()) {
        cerr << "ERROR: No images in provided path." << endl;
        return 1;
    }

    if (vstrImageLeft.size() != vstrImageRight.size()) {
        cerr << "ERROR: Different number of left and right images." << endl;
        return 1;
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(configFile, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
        D_r.empty() ||
        rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l,
                                M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r,
                                M2r);

    const int nImages = vstrImageLeft.size();

    // Create camera object
    setting::initSettings();
    float fx = fsSettings["Camera.fx"];
    float fy = fsSettings["Camera.fy"];
    float cx = fsSettings["Camera.cx"];
    float cy = fsSettings["Camera.cy"];

    shared_ptr<CameraParam> camera = make_shared<CameraParam>(fx, fy, cx, cy);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    size_t imuIndex = 0;

    // ORBExtractor extractor(ORBExtractor::ORB_SLAM2);
    // ORBExtractor extractor(ORBExtractor::FAST_MULTI_LEVEL);
    ORBExtractor extractor( ORBExtractor::FAST_SINGLE_LEVEL );

    for (int ni = 0; ni < nImages; ni++) {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        if (imRight.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageRight[ni]) << endl;
            return 1;
        }

        cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        VecIMU vimu;

        double tframe = vTimeStamp[ni];
        while (1) {
            const ygz::IMUData &imudata = vimus[imuIndex];
            if (imudata.mfTimeStamp >= tframe)
                break;
            vimu.push_back(imudata);
            imuIndex++;
        }

        shared_ptr<Frame> frame = make_shared<Frame>(imLeftRect, imRightRect, tframe, camera, vimu);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // do some operation
        LOG(INFO) << "detecting features" << endl;
        extractor.Detect(frame, true);
        extractor.Detect(frame, false);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "time of feature extraction: " << timeCost << endl;

        // show the result
        LOG(INFO) << "showing features, size = " << frame->mFeaturesLeft.size() << endl;
        Mat img_left;
        cv::cvtColor(imLeftRect, img_left, cv::COLOR_GRAY2BGR);
        for (shared_ptr<Feature> feature: frame->mFeaturesLeft) {

            Vector3f color;
            feature->mScore /= 100.;
            if (feature->mScore < 0.5) {
                color = Vector3f(0, 2 * feature->mScore, 1) * 255;
            } else if (feature->mScore < 1) {
                color = Vector3f(0, 1, 1 - 2 * (feature->mScore - 0.5)) * 255;
            } else {
                color = Vector3f(0, 1, 0) * 255;
            }

            cv::circle(img_left, cv::Point2f(feature->mPixel[0], feature->mPixel[1]), 1,
                       cv::Scalar(color[0], color[1], color[2]), 2);
        }
        cv::imshow("Feature Left", img_left);

        Mat img_right;
        cv::cvtColor(imRightRect, img_right, cv::COLOR_GRAY2BGR);
        for (shared_ptr<Feature> feature: frame->mFeaturesRight) {
            feature->mScore /= 100.;
            Vector3f color;
            if (feature->mScore < 0.5) {
                color = Vector3f(0, 2 * feature->mScore, 1) * 255;
            } else if (feature->mScore < 1) {
                color = Vector3f(0, 1, 1 - 2 * (feature->mScore - 0.5)) * 255;
            } else {
                color = Vector3f(0, 1, 0) * 255;
            }

            cv::circle(img_right, cv::Point2f(feature->mPixel[0], feature->mPixel[1]), 1,
                       cv::Scalar(color[0], color[1], color[2]), 2);
        }
        cv::imshow("Feature Right", img_right);
        cv::waitKey(1);
    }

    setting::destroySettings();
    return 0;
}

