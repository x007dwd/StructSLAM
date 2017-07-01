//
// Created by bobin on 17-6-29.
//

#include <iostream>

#include "ygz/Frame.h"
#include "ygz/Feature.h"
#include "ygz/MapPoint.h"
#include "ygz/BackendSlidingWindowG2O.h"
#include "ygz/Viewer.h"
#include "g2o/types/slam3d/edge_se3_lineendpts.h"

using namespace ygz;

/**
 * 后端BA函数的测试，使用仿真数据
 * 结论：Pose基本能估计到原值，而地图点的世界坐标，由于采用了逆深度，和pose估计值有关，所以和世界坐标有一定差距
 */


//void add_line_featues(vector<Vector3d> line, vector<Vector2d>obs, ){
//
//}


int main(int argc, char **argv) {
    shared_ptr<BackendSlidingWindowG2O> backend = make_shared<BackendSlidingWindowG2O>(nullptr);
    backend->testLocalBA();
    backend = nullptr;
    return 0;
}

void BackendSlidingWindowG2O::testLocalBA() {
    // 测试函数实现
    setting::initSettings();

    // fake extrinsic, align camera and body frame
    setting::TBC = Sophus::SE3d();

    double fakeb = 0.1;  // fake baseline
    int fakef = 400;     // fake focal length in pixel
    int numPoints = 10;
    int numLines = 10;
    int numFrames = 10;
    double sigmaObs = 1.0;  // 观测误差的高斯分布
    double sigmaInvD = /*1.414**/0.5*sigmaObs/(fakef*fakeb);  // 逆深度的高斯分布
    vector<shared_ptr<MapPoint>  > allPoints;
    vector<shared_ptr<Frame>  > allFrames;

    VecVector3d realPosWorld;
    VecSE3d realPose;

    // 相机
    shared_ptr<CameraParam> cam = make_shared<CameraParam>(fakef, fakef, 320, 240, fakef*fakeb);

    // 随机生成地图点
//    LOG(INFO) << "Generating data..." << endl;
//    for (int i = 0; i < numPoints; i++) {
//        double x = rand() % 10 - 5;
//        double y = rand() % 10 - 5;
//        double z = rand() % 10 + 10; // 让地图点都在 z>0 的地方
//        realPosWorld.push_back(Vector3d(x, y, z));
//        shared_ptr<MapPoint>  mp = make_shared<MapPoint>();
//        mp->mWorldPos = Vector3d(x, y, z);
//        allPoints.push_back(mp);
//    }
    LOG(INFO) << "Generating data in line .." << endl;

    for (int j = 0; j < numLines; ++j) {
        Vector3d ori(5, 1, 0);

        for (int i = 0; i < numPoints; i++) {
            double x = i - 5;
            double y = 1 * x / 5;
            double z = j ; // 让地图点都在 z>0 的地方
            realPosWorld.push_back(Vector3d(x, y, z));
            shared_ptr<MapPoint> mp = make_shared<MapPoint>();
            mp->mWorldPos = Vector3d(x, y, z);
            allPoints.push_back(mp);
        }
    }

    // 随机生成一些kf
    for (int i = 0; i < numFrames; i++) {
        shared_ptr<Frame> f = make_shared<Frame>();

        // 注意这个是Twb
        SE3d pose(SO3d(0, 0, 0), Vector3d(0.5*(i - 5), 0, 0));    // 沿X轴匀速运动

        f->SetPose(pose);
        f->mpCam = cam;
        f->SetThisAsKeyFrame();

        f->SetPoseGT(pose);

        allFrames.push_back(f);
        realPose.push_back(pose);
    }

    // 随机数发生器
    cv::RNG rng;
    rng.gaussian(1);

    LOG(INFO) << "Generating observations ... " << endl;
    // 生成一些 reference 和 observation，默认1~10号点由kf1看到，11~20号点由kf2看到
    for (int i = 0; i < numPoints * numLines; i++) {
        shared_ptr<MapPoint>  mp = allPoints[i];
        shared_ptr<Frame>  kf = allFrames[i / 10];    // reference
        // generate observation
        Vector3d ptCam = kf->mRcw * mp->mWorldPos + kf->mtcw;
        if (ptCam[2] < 0)
            continue;
        // 添加一个观测
        mp->mpRefKF = kf;
        shared_ptr<Feature> feat = make_shared<Feature>();
        feat->mpPoint = mp;

        // 首次观测是准的
        feat->mPixel = Vector2f(
                cam->fx * ptCam[0] / ptCam[2] + cam->cx,
                cam->fy * ptCam[1] / ptCam[2] + cam->cy
        );
        LOG(INFO) << feat->mPixel[0] << '\t' << feat->mPixel[1] << endl;
        feat->mfInvDepth = 1.0 / ptCam[2] /*+ 0.1*rng.gaussian(sigmaInvD)*/;
        feat->mfInvDepth = feat->mfInvDepth < 0.0001 ? 0.0001 : feat->mfInvDepth;   // avoid negative inv depth
        kf->mFeaturesLeft.push_back(feat);  // 这个值应该也需要加噪声
        mp->AddObservation(kf, kf->mFeaturesLeft.size() - 1);

        // 添加在其他帧中的观测
        for (shared_ptr<Frame>  frame: allFrames) {
            if (frame == kf)
                continue;

            // generate observation
            Vector3d ptCam = frame->mRcw * mp->mWorldPos + frame->mtcw;
            if (ptCam[2] < 0)
                continue;

            // 添加一个观测
            shared_ptr<Feature> feat = make_shared<Feature>();
            feat->mpPoint = mp;
            feat->mPixel = Vector2f(
                    cam->fx * ptCam[0] / ptCam[2] + cam->cx + rng.gaussian(sigmaObs),
                    cam->fy * ptCam[1] / ptCam[2] + cam->cy + rng.gaussian(sigmaObs)
            );
            feat->mfInvDepth = 1.0 / ptCam[2];
            frame->mFeaturesLeft.push_back(feat);
            mp->AddObservation(frame, frame->mFeaturesLeft.size() - 1);
        }
    }

    // 将这些KF和Point放入后端进行BA
    for (shared_ptr<Frame>  f: allFrames) {
        mpKFs.push_back(f);
    }
    for (shared_ptr<MapPoint>  mp: allPoints) {
        mpPoints.insert(mp);
    }

    LOG(INFO) << "Solving BA ... " << endl;
    LocalBAWithoutIMU(true);

    // 对比估计值和真实值
    LOG(INFO) << "Results: " << endl;
    for (int i = 0; i < 10; i++) {
        LOG(INFO) << "Real pose = \n" << realPose[i].matrix() << endl;
        LOG(INFO) << "Estimated pose = \n" << SE3d(allFrames[i]->mRwb, allFrames[i]->mTwb).matrix() << endl;
    }
    LOG(INFO) << endl;
    for (int i = 0; i < 100; i++) {
        LOG(INFO) << "Real map point pos: " << realPosWorld[i].transpose() << endl;
        LOG(INFO) << "Estimated map point pos: " << allPoints[i]->mWorldPos.transpose() << endl;
    }

    Viewer viewer(false);
    viewer.ShowConnection(false);
    viewer.ShowCurrentImg(false);   // 没有image可以show

    for (shared_ptr<Frame>  frame: allFrames) {
        viewer.AddFrame(frame);
    }

    viewer.RunAndSpin();
    LOG(INFO) << "Clean the data" << endl;

    // clean the data
    for (shared_ptr<Frame>  f: allFrames) {
        f = nullptr;
    }

    for (shared_ptr<MapPoint>  mp: allPoints) {
        mp = nullptr;
    }

    setting::destroySettings();

}
