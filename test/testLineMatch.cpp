//
// Created by bobin on 17-6-26.
//

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <common/include/ygz/Feature.h>
#include "ygz/Viewer.h"
#include "ygz/Frame.h"
#include "ygz/Settings.h"
#include "ygz/SparePoint.h"
#include "ygz/DSOCoarseTracker.h"
#include "ygz/Frame.h"
#include "ygz/EurocReader.h"
#include "ygz/PixelSelector.h"
#include "ygz/BackendSlidingWindowG2O.h"
#include "ygz/LineFeature.h"
#include "ygz/LineFeature.h"
// kitti data set
string leftFolder = "/home/bobin/data/euroc/MH_01/cam0/data/";
string rightFolder = "/home/bobin/data/euroc/MH_01/cam1/data/";
string timeFolder = "../examples/EuRoC_TimeStamps/MH01.txt";
string configFile = "../examples/EuRoC.yaml";
string imuFolder = "/home/bobin/data/euroc/MH_01/imu0/data.csv";

using namespace ygz;
shared_ptr<CoarseTracker> pTracker = nullptr;

int main(int argc, char **argv) {
    pTracker = shared_ptr<CoarseTracker>(new CoarseTracker);
    pTracker->TestStereoMatch();
    return 0;
}


int CoarseTracker::TestStereoMatch() {
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
    float bf = fsSettings["Camera.bf"];

    shared_ptr<CameraParam> camera(new CameraParam(fx, fy, cx, cy, bf));

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    size_t imuIndex = 0;


    camera->MakeK();
    this->mpCam = camera;
    PixelSelector mPixelSelector(setting::imageWidth, setting::imageHeight);
    Mat floatMap(setting::imageHeight, setting::imageWidth, CV_32F);
    float *pFloatMap = (float *) (floatMap.data);

    vector<SparePoint> vecSparePoints;

    srand(time(nullptr));

    LineExtract extractor;

    float densities[] = {0.03, 0.05, 0.15, 0.5, 1};

    for (int ni = 0; ni < nImages; ni++) {
//        LOG(INFO) << "Loading " << ni << " image" << endl;
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

        shared_ptr<Frame> frame(new Frame(imLeftRect, imRightRect, tframe, camera, vimu));

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // do some operation
        // 左右眼各提一次
        LOG(INFO) << "Detecting points in frame " << frame->mnId << endl;

        Mat img_left, img_right;
        cv::cvtColor(imLeftRect, img_left, cv::COLOR_GRAY2BGR);
        int valid = 0;
        // 这里要分一下层，才能把所有的idepth计算完整。


        Mat out;
        Mat imLine;
        imLeftRect.convertTo(imLine,CV_64F);
        cvtColor(imLeftRect, out, CV_GRAY2BGR);
        vector<LineFeature> feats;
        extractor.DetectLine(imLine, feats);
        extractor.drawLine(out,out,feats);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        LOG(INFO) << "lsd cost time: " << timeCost << endl;



        cv::imshow("test", out);
        cv::waitKey(1);

        LOG(INFO) << "point with valid depth: " << valid << endl;
        cv::imshow("Feature and distance", img_left);
        cv::waitKey(1);
    }

    setting::destroySettings();
    return 0;
}

void CoarseTracker::TestTracker() {

    //LOG(INFO) << "test " << endl;

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    VecIMU vimus;

    LoadImages(leftFolder, rightFolder, timeFolder, vstrImageLeft, vstrImageRight, vTimeStamp);
    LoadImus(imuFolder, vimus);

    if (vstrImageLeft.empty() || vstrImageRight.empty()) {
        cerr << "ERROR: No images in provided path." << endl;
        return;
    }

    if (vstrImageLeft.size() != vstrImageRight.size()) {
        cerr << "ERROR: Different number of left and right images." << endl;
        return;
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(configFile, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "ERROR: Wrong path to settings" << endl;
        return;
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
        return;
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

    srand(time(nullptr));
    // create a backend
    shared_ptr<BackendSlidingWindowG2O> backend(new BackendSlidingWindowG2O(pTracker));
    mpBackEnd = backend;

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    size_t imuIndex = 0;

    Viewer viewer(true);
//    mbVisionOnlyMode = true;

    camera->MakeK();
    this->mpCam = camera;

    float densities[] = {0.03, 0.05, 0.15, 0.5, 1};


    for (int ni = 0; ni < nImages; ni++) {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return;
        }

        if (imRight.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageRight[ni]) << endl;
            return;
        }
        assert(imLeft.channels() == 1);
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

        SE3d Twb = this->InsertStereo(imLeftRect, imRightRect, tframe, vimu);
        LOG(INFO) << "current Twb = \n" << Twb.matrix() << endl;

        if (mpCurrentFrame->IsKeyFrame()) {
            viewer.AddFrame(mpCurrentFrame, true);
        } else {
            viewer.SetCurrentFrame(mpCurrentFrame);
        }

        if (mState == eTrackingState::OK) {
            mState = eTrackingState::NOT_INITIALIZED; //
        }
    }

}