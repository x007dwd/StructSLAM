#include "ygz/Frame.h"

/***
 * 本程序测试在EUROC数据集上前后帧的匹配算法
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
#include "ygz/ORBMatcher.h"
#include "ygz/EurocReader.h"

using namespace std;
using namespace ygz;

string leftFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam0/data";
string rightFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam1/data";
string timeFolder = "./examples/EuRoC_TimeStamps/MH01.txt";
string configFile = "./examples/EuRoC.yaml";
string imuFolder = "/home/xiang/dataset/euroc/MH_01_easy/imu0/data.csv";

int main(int argc, char **argv) {
    cout << "test feature extraction step in EuRoC" << endl;

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

    // load the vocabulary
    cout << "Loading vocabulary at ./Vocabulary/ORBvoc.bin" << endl;
    shared_ptr<ORBVocabulary> pVocabulary = make_shared<ORBVocabulary>();
    pVocabulary->loadFromBinaryFile("./Vocabulary/ORBvoc.bin");
    if (pVocabulary->getDepthLevels() == 0) {
        cerr << "Cannot load vocabulary" << endl;
        return 1;
    }
    LOG(INFO)<<"Load done."<<endl;
    Frame::pORBvocabulary = pVocabulary;


    // Main loop
    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    size_t imuIndex = 0;

    ORBExtractor extractor(ORBExtractor::ORB_SLAM2);
    ORBMatcher matcher(0.6, false);

    shared_ptr<Frame> last_frame = nullptr;
    shared_ptr<Frame> curr_frame = nullptr;

    srand(time(nullptr));

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

        curr_frame = make_shared<Frame>(imLeftRect, imRightRect, tframe, camera, vimu);

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // do some operation
        extractor.Detect(curr_frame, true);
        curr_frame->ComputeBoW();

        vector<Match> matches;
        if (last_frame) {
            // 与上一个帧进行特征匹配
            cout << "last id = " << last_frame->mnId << ", curr id = " << curr_frame->mnId << endl;

            matcher.SearchBruteForce(last_frame, curr_frame, matches);
            // matcher.SearchByBoW( last_frame, curr_frame, matches );

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            cout << "time of feature matching: " << timeCost << endl;

            // select good ones
            int min_dist =
                    std::min_element(matches.begin(), matches.end(),
                                     [](const Match &m1, const Match &m2) { return m1.dist < m2.dist; })->dist;
            cout << "min dist = " << min_dist << endl;
            int th = std::max(int(3.0 * min_dist), 30);

            Mat img_last, img_curr;
            cv::cvtColor(last_frame->mImLeft, img_last, CV_GRAY2BGR);
            cv::cvtColor(curr_frame->mImLeft, img_curr, CV_GRAY2BGR);

            int nGood = 0;
            for (Match &m: matches) {
                if (m.dist < th) {
                    // a good one
                    nGood++;
                    int r = rand() % 255;
                    int g = rand() % 255;
                    int b = rand() % 255;
                    shared_ptr<Feature> f1 = last_frame->mFeaturesLeft[m.index1];
                    shared_ptr<Feature> f2 = curr_frame->mFeaturesLeft[m.index2];

                    cv::circle(img_last, cv::Point2f(f1->mPixel[0], f1->mPixel[1]), 5, cv::Scalar(b, g, r), 2);
                    cv::circle(img_curr, cv::Point2f(f2->mPixel[0], f2->mPixel[1]), 5, cv::Scalar(b, g, r), 2);
                }
            }
            cout << "good matches: " << nGood << endl;
            cv::imshow("last frame", img_last);
            cv::imshow("curr frame", img_curr);

            cv::waitKey(1);
        }

        last_frame = curr_frame;
    }

    setting::destroySettings();

    return 0;
}

