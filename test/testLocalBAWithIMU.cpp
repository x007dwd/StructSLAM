#include <iostream>

#include "ygz/Frame.h"
#include "ygz/Feature.h"
#include "ygz/MapPoint.h"
#include "ygz/BackendSlidingWindowG2O.h"
#include "ygz/Tracker.h"
#include "ygz/Viewer.h"

using namespace ygz;

int main(int argc, char **argv) {
    shared_ptr<BackendSlidingWindowG2O> backend = make_shared<BackendSlidingWindowG2O>(nullptr);
    backend->testLocalBAIMU();
    backend = nullptr;
    return 0;
}

void BackendSlidingWindowG2O::testLocalBAIMU() {

    // 测试带IMU的Local BA是否正确
    setting::initSettings();

    setting::TBC = Sophus::SE3d();

    const double DURATION = 10.0;  // 10 seconds motion
    const double IMU_RATE = 200.0;  // 1 kHz
    const double DT = 1.0 / IMU_RATE;  // time increments

    // set the imu parameters
    // 陀螺和加计的其他参数保持与setting中一致
    double g = 9.81;        // 重力值
    Vector3d gWorld(0, 0, g);

    // let's generate a really stupid motion: constant translation
    // 用匀速运动生成IMU数据（加速度和角速度都为零）
    cv::RNG rng;    // 随机数生成器
    VecVector3d vBiasG;
    VecVector3d vBiasA;
    Vector3d bgCurrent = Vector3d(0, 0, 0); // 初始bias不妨为零
    Vector3d baCurrent = Vector3d(0, 0, 0); // 初始bias不妨为零

    // 沿x轴1m/s的匀速运动
    Vector3d speed(1, 0, 0);
    VecIMU imuData;

    double gyrBiasRw = sqrt(setting::gyrBiasRw2) * DT;
    double accBiasRw = sqrt(setting::accBiasRw2) * DT;
    double gyrMeasRw = sqrt(setting::gyrMeasError2 * DT);
    double accMeasRw = sqrt(setting::accMeasError2 * DT);

    for (size_t i = 0; i <= +DURATION * IMU_RATE; i++) {
        double time = DT * i;
        bgCurrent += Vector3d(rng.gaussian(gyrBiasRw), rng.gaussian(gyrBiasRw), rng.gaussian(gyrBiasRw));
        baCurrent += Vector3d(rng.gaussian(accBiasRw), rng.gaussian(accBiasRw), rng.gaussian(accBiasRw));

        Vector3d gyr = Vector3d::Zero() + bgCurrent +
                       Vector3d(rng.gaussian(gyrMeasRw), rng.gaussian(gyrMeasRw), rng.gaussian(gyrMeasRw));
        Vector3d acc = -gWorld + baCurrent +
                       Vector3d(rng.gaussian(accMeasRw), rng.gaussian(accMeasRw), rng.gaussian(accMeasRw));

        vBiasG.push_back(bgCurrent);
        vBiasA.push_back(baCurrent);

        imuData.push_back(IMUData(gyr, acc, time));
        LOG(INFO) << "bg: "<<bgCurrent.transpose()<<", ba: "<<baCurrent.transpose()<<endl;
    }

    // 生成地图点和关键帧
    shared_ptr<CameraParam> cam = make_shared<CameraParam>(400, 400, 320, 240, 40);

    // 需要用到Tracker里的g
    mpTracker = make_shared<Tracker>();
    mpTracker->SetBackEnd(shared_ptr<BackendInterface>(this));
    mpTracker->SetGravity(gWorld);

    int numPoints = 100;
    int numFrames = 10;

    vector<shared_ptr<MapPoint>> allPoints;
    vector<shared_ptr<Frame> > allFrames;

    VecVector3d realPosWorld;
    VecSE3d realPose;

    double sigmaObs = 1.0;  // 观测误差的高斯分布
    double sigmaInvD = 1/40.0*0.1;  // 逆深度的高斯分布

    LOG(INFO) << "Generating points ..." << endl;
    for (int i = 0; i < numPoints; i++) {
        double x = rand() % 10 - 5;
        double y = rand() % 10 - 5;
        double z = rand() % 10 + 10; // 让地图点都在 z>0 的地方
        realPosWorld.push_back(Vector3d(x, y, z));
        shared_ptr<MapPoint>  mp = make_shared<MapPoint>();
        mp->mWorldPos = Vector3d(x, y, z);
        allPoints.push_back(mp);
    }

    // 每秒一个关键帧，共10个
    LOG(INFO) << "Generating keyframes ..." << endl;
    for (int i = 0, imuIndex = 0; i < numFrames; i++) {
        double time = i;
        LOG(INFO)<<"Creating frame "<<i<<endl;
        shared_ptr<Frame> f = make_shared<Frame>();

        LOG(INFO)<<"set pose "<<i<<endl;
        // 注意这个是Twb
        SE3d pose(SO3d(0, 0, 0), Vector3d(-5, 0, 0) + speed * time);
        f->SetPose(pose);
        f->mpCam = cam;
        f->SetThisAsKeyFrame();
        f->SetSpeed(speed);
        f->SetBiasA(Vector3d(0, 0, 0));
        f->SetBiasG(Vector3d(0, 0, 0));
        f->mTimeStamp = time;

        // insert the imu data
        LOG(INFO)<<"insert imu data "<<i<<endl;
        for (size_t j = imuIndex; j < imuData.size(); j++) {
            IMUData &data = imuData[j];
            if (data.mfTimeStamp < time) {
                f->mvIMUDataSinceLastFrame.push_back(data);
            } else {
                imuIndex = j;
                break;
            }
        }

        if (i == 0) {
            f->mpReferenceKF.reset();
        } else {
            f->mpReferenceKF = weak_ptr<Frame>(allFrames[i - 1]);
            LOG(INFO)<<"compute imu pre int "<<i<<endl;
            f->ComputeIMUPreInt();
        }

        LOG(INFO)<<"push to frames "<<i<<endl;
        allFrames.push_back(f);
        LOG(INFO)<<"push to pose "<<i<<endl;
        realPose.push_back(pose);
    }

    // Generate observations
    // 所有点一开始都从kf1生成，然后被其他kf观测到
    LOG(INFO)<<"Generating observations"<<endl;
    for (size_t i = 0; i < numPoints; i++) {

        shared_ptr<MapPoint> mp = allPoints[i];
        shared_ptr<Frame> kf = allFrames[i/10];

        // generate observation
        Vector3d ptCam = kf->mRcw * mp->mWorldPos + kf->mtcw;
        if (ptCam[2] < 0)
            continue;

        mp->mpRefKF = kf;
        shared_ptr<Feature> feat = make_shared<Feature>();
        feat->mpPoint = mp;

        // 首次观测是准的
        feat->mPixel = Vector2f(
                cam->fx * ptCam[0] / ptCam[2] + cam->cx,
                cam->fy * ptCam[1] / ptCam[2] + cam->cy
        );

        feat->mfInvDepth = 1.0 / ptCam[2] /*+ rng.gaussian(sigmaInvD)*/;
        feat->mfInvDepth = feat->mfInvDepth < 0.0001 ? 0.0001 : feat->mfInvDepth;   // avoid negative inv depth
        kf->mFeaturesLeft.push_back(feat);  // 这个值应该也需要加噪声
        mp->AddObservation(kf, kf->mFeaturesLeft.size() - 1);

        // 添加在其他帧中的观测
        for (shared_ptr<Frame> frame: allFrames) {
            if (frame == kf)
                continue;

            // generate observation
            ptCam = frame->mRcw * mp->mWorldPos + frame->mtcw;
            if (ptCam[2] < 0)
                continue;

            // 添加一个观测
            feat = make_shared<Feature>();
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

    for (shared_ptr<Frame> f: allFrames) {
        mpKFs.push_back(f);
    }

    for (shared_ptr<MapPoint>  mp: allPoints) {
        mpPoints.insert(mp);
    }

    LOG(INFO)<<"Call local ba with imu "<<endl;
    LocalBAWithIMU(true);

    // 对比估计值和真实值
    LOG(INFO) << "Results: " << endl;
    for (int i = 0; i < 10; i++) {
        LOG(INFO) << "Real pose = \n" << realPose[i].matrix() << endl;
        LOG(INFO) << "Estimated pose = \n" << SE3d(allFrames[i]->mRwb, allFrames[i]->mTwb).matrix() << endl;
        LOG(INFO) << "Estimate bias = " << allFrames[i]->BiasG().transpose() << ", "
                  << allFrames[i]->BiasA().transpose() << endl;
    }

    LOG(INFO) << endl;
    for (int i = 0; i < 100; i++) {
        LOG(INFO) << "Real map point pos: " << realPosWorld[i].transpose() << endl;
        LOG(INFO) << "Estimated map point pos: " << allPoints[i]->mWorldPos.transpose() << endl;
    }

    Viewer viewer(false);
    viewer.ShowConnection(true);
    viewer.ShowCurrentImg(false);   // 没有image可以show

    for (shared_ptr<Frame> frame: allFrames) {
        viewer.AddFrame(frame);
    }

    viewer.RunAndSpin();
    LOG(INFO) << "Clean the data" << endl;

    // clean the data
    for (shared_ptr<Frame> f: allFrames) {
        f = nullptr;
    }

    for (shared_ptr<MapPoint>  mp: allPoints) {
        mp = nullptr;
    }

    mpTracker = nullptr;

    setting::destroySettings();
    cam = nullptr;
}
