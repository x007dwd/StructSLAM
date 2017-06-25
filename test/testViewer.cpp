/**
 * 本程序测试Viewer的功能
 * 请注意Euroc前几帧似乎没有真实pose
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <iomanip>

#include "ygz/Frame.h"
#include "ygz/Settings.h"
#include "ygz/ORBExtractor.h"
#include "ygz/ORBMatcher.h"
#include "ygz/EurocReader.h"
#include "ygz/Viewer.h"
#include "ygz/Memory.h"


string leftFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam0/data";
string rightFolder = "/home/xiang/dataset/euroc/MH_01_easy/cam1/data";
string timeFolder = "./examples/EuRoC_TimeStamps/MH01.txt";
string imuFolder = "/home/xiang/dataset/euroc/MH_01_easy/imu0/data.csv";
string groundTruthFile = "/home/xiang/dataset/euroc/MH_01_easy/state_groundtruth_estimate0/data.csv";
string configFile = "./examples/EuRoC.yaml";

using namespace std;
using namespace ygz;

int main(int argc, char **argv) {
    setting::initSettings();
    LOG(INFO)<<"TBC = \n"<<setting::TBC.matrix()<<endl;
    Viewer viewer(false);
    viewer.TestViewer();
    setting::destroySettings();
}

// 往viewer中添加新图像的线程
void SetNewImage(Viewer *viewer);

void Viewer::TestViewer() {

    std::thread th(SetNewImage, this);
    this->RunAndSpin();
    th.join();
}

void SetNewImage(Viewer *viewer) {

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimeStamp;
    VecIMU vimus;
    TrajectoryType gtTraj;   // ground truth trajectory

    LoadImages(leftFolder, rightFolder, timeFolder, vstrImageLeft, vstrImageRight, vTimeStamp);
    LoadImus(imuFolder, vimus);
    LoadGroundTruthTraj(groundTruthFile, gtTraj);

    assert(!vstrImageLeft.empty());
    assert(!vstrImageRight.empty());
    assert(!vimus.empty());
    assert(!gtTraj.empty());
    
    LOG(INFO) << "Trajectory poses: " << gtTraj.size() << endl;

    // read camera parameters
    cv::FileStorage fsSettings(configFile, cv::FileStorage::READ);
    assert(fsSettings.isOpened());

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
        LOG(ERROR) << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
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
    float bf = fsSettings["Camera.bf"];

    shared_ptr<CameraParam> camera = make_shared<CameraParam>(fx, fy, cx, cy, bf);

    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    size_t imuIndex = 0;

    auto gtPoseIter = gtTraj.begin();
    SE3d Tiw;   // T-world-initial

    for (int ni = 0; ni < nImages; ni++) {

        if (viewer->IsRunning() == false)
            break;

        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);

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
        LOG(INFO) << "Show frame " << frame->mnId << endl;

        ORBExtractor extractor(ORBExtractor::ORB_SLAM2);
        extractor.Detect(frame, true);
        extractor.Detect(frame, false);

        ORBMatcher matcher;
        matcher.ComputeStereoMatches(frame);

        // find its ground truth pose
        for (auto iter = gtPoseIter; iter != gtTraj.end(); iter++) {
            if (iter->first > tframe) {
                gtPoseIter = iter;
                break;
            }
        }

        if (ni == 0)
            Tiw = gtPoseIter->second.inverse();

        LOG(INFO) << std::setprecision(18) << "Current frame time: " << tframe << ", pose time " << gtPoseIter->first
                  << endl;
        LOG(INFO) << "Frame " << frame->mnId << " pose is set to \n" << gtPoseIter->second.matrix() << endl;
        frame->SetPose(Tiw * gtPoseIter->second);   // 设置相对于第一个帧的Pose

        viewer->SetCurrentFrame( frame );

        usleep(3000);

    }

    //memory::CleanAllMemory();

    camera = nullptr;
    setting::destroySettings();
}
