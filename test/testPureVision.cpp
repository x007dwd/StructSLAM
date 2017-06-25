#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/Frame.h"
#include "ygz/Feature.h"
#include "ygz/Settings.h"
#include "ygz/ORBExtractor.h"
#include "ygz/ORBMatcher.h"
#include "ygz/EurocReader.h"
#include "ygz/Tracker.h"
#include "ygz/BackendSlidingWindowG2O.h"
#include "ygz/Viewer.h"
#include "ygz/Memory.h"

using namespace std;
using namespace ygz;

/***
 * 本程序测试在EUROC数据集上双目纯视觉的Tracking
 */

// 路径
double discardtime = 0; // discard some frames in the beginning. for V101~7s
/*
string leftFolder = "/media/jp/JingpangPassport/3dataset/EuRoC-VIO/zipfiles/MH_01_easy/mav0/cam0/data";
string rightFolder = "/media/jp/JingpangPassport/3dataset/EuRoC-VIO/zipfiles/MH_01_easy/mav0/cam1/data";
string timeFolder = "/home/jp/opensourcecode/ygz-stereo-inertial/examples/EuRoC_TimeStamps/MH01.txt";
string imuFolder = "/media/jp/JingpangPassport/3dataset/EuRoC-VIO/zipfiles/MH_01_easy/mav0/imu0/data.csv";
string configFile = "/home/jp/opensourcecode/ygz-stereo-inertial/examples/EuRoC.yaml";
string vocFile = "/home/jp/opensourcecode/ygz-stereo-inertial/Vocabulary/ORBvoc.bin";
 */

string leftFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam0/data";
string rightFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam1/data";
string imuFolder = "/home/xiang/dataset/euroc/MH_01_easy/imu0/data.csv";
string timeFolder = "/home/xiang/code/ygz-stereo-inertial/examples/EuRoC_TimeStamps/MH01.txt";
string configFile = "/home/xiang/code/ygz-stereo-inertial/examples/EuRoC.yaml";
string vocFile = "/home/xiang/code/ygz-stereo-inertial/Vocabulary/ORBvoc.bin";

int start_index = 1000 ; // 起始图像的索引

shared_ptr<Tracker> pTracker = nullptr;

int main(int argc, char **argv) {
    pTracker = make_shared<Tracker>();
    pTracker->TestPureVision();
    return 0;
}

void Tracker::TestPureVision() {

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    VecIMU vimus;

    LoadImages(leftFolder, rightFolder, timeFolder, vstrImageLeft, vstrImageRight, vTimeStamp);
    LoadImus(imuFolder, vimus);

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

    // Create vocabulary
    shared_ptr<ORBVocabulary> pVocabulary = make_shared<ORBVocabulary>();
    pVocabulary->loadFromBinaryFile(vocFile);
    Frame::pORBvocabulary = pVocabulary;

    const int nImages = vstrImageLeft.size();

    // Create camera object
    setting::initSettings();
    float fx = fsSettings["Camera.fx"];
    float fy = fsSettings["Camera.fy"];
    float cx = fsSettings["Camera.cx"];
    float cy = fsSettings["Camera.cy"];
    float bf = fsSettings["Camera.bf"];

    shared_ptr<CameraParam> camera = make_shared<CameraParam>(fx, fy, cx, cy, bf);
    this->mpCam = camera;

    srand(time(nullptr));
    // create a backend
    shared_ptr<BackendSlidingWindowG2O> backend = make_shared<BackendSlidingWindowG2O>(pTracker);
    mpBackEnd = backend;

    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    size_t imuIndex = 0;

    // set up visualization
    Viewer viewer(true);
    mbVisionOnlyMode = true;

    double tstart = vTimeStamp[0];
    for (int ni = start_index; ni < nImages; ni++) {

        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);

        cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        VecIMU vimu;

        double tframe = vTimeStamp[ni];

        if (tframe - tstart < discardtime)
            continue;

        while (1) {
            const ygz::IMUData &imudata = vimus[imuIndex];
            if (imudata.mfTimeStamp >= tframe)
                break;
            vimu.push_back(imudata);
            imuIndex++;
        }

        this->InsertStereo(imLeftRect, imRightRect, tframe, vimu);

        if (mpCurrentFrame->IsKeyFrame()) {
            viewer.AddFrame(mpCurrentFrame, true);
        } else {
            viewer.SetCurrentFrame(mpCurrentFrame);
        }

        if (mState == eTrackingState::OK) {
            mState = eTrackingState::NOT_INITIALIZED; //
        }
    }

    setting::destroySettings();
    viewer.Close();
}
