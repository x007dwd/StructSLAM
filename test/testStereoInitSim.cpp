#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <common/include/ygz/Feature.h>

#include "ygz/G2OTypes.h"
#include "ygz/Frame.h"
#include "ygz/Settings.h"
#include "ygz/ORBExtractor.h"
#include "ygz/ORBMatcher.h"
#include "ygz/EurocReader.h"
#include "ygz/Tracker.h"
#include "ygz/BackendSlidingWindowG2O.h"
#include "ygz/Viewer.h"
#include "ygz/Memory.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>


using namespace std;
using namespace ygz;

string configFile = "/home/jp/opensourcecode/ygz-stereo-inertial/examples/EuRoC.yaml";
/***
 * 本程序测试在仿真数据上IMU初始化
 */

class ImagePRt {
public:
    ImagePRt(double t, const Matrix3d &R_, const Vector3d &P_) :
        mTimestamp(t), R(R_), P(P_) {}

    double mTimestamp;
    Matrix3d R;
    Vector3d P;
};

void LoadImusSim(VecIMU &vImus, vector<ImagePRt> &vImagePRt);
void testIMUInit(const vector<shared_ptr<Frame>> &vpKFs);
Vector3d IMUInitEstBg(const std::vector<shared_ptr<Frame>> &vpKFs);

int main(int argc, char **argv) {
    Tracker tracker;
    tracker.TestStereoInit();
    return 0;
}

void Tracker::TestStereoInit() {

    vector<ImagePRt> vImagePRt;
    VecIMU vimus;

    LoadImusSim(vimus, vImagePRt);
    int nImages = vImagePRt.size();


    LOG(INFO) << vimus.size() << " imu data" << endl;
    LOG(INFO) << vImagePRt.size() << " image P/R/timestamp" << endl;

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

    // Main loop
    size_t imuIndex = 0;

    vector<shared_ptr<Frame>> vpKFs;
    mpLastKeyFrame = nullptr;

    for (int ni = 0; ni < nImages; ni++) {

        VecIMU vimu;
        const ImagePRt &tmpImagePRt = vImagePRt[ni];
        double tframe = tmpImagePRt.mTimestamp;
        while (1) {
            const ygz::IMUData &imudata = vimus[imuIndex];
            if (imudata.mfTimeStamp >= tframe)
                break;
            vimu.push_back(imudata);
            imuIndex++;
        }

        mpCurrentFrame = make_shared<Frame>();
        mpCurrentFrame->mTimeStamp = tframe;
        mpCurrentFrame->mvIMUDataSinceLastFrame = vimu;
        mpCurrentFrame->SetPose(Sophus::SE3d(tmpImagePRt.R,tmpImagePRt.P));
        mpCurrentFrame->SetThisAsKeyFrame();
        mpCurrentFrame->mpReferenceKF = mpLastKeyFrame;

        vpKFs.push_back(mpCurrentFrame);

        mpLastKeyFrame = mpCurrentFrame;

        if (mState == eTrackingState::OK) {
            LOG(INFO) << "Stereo is successfully initialized." << endl;
            break;
        }
    }

    LOG(INFO) << "vpKFs.size = " << vpKFs.size() << endl;

    testIMUInit(vpKFs);

    setting::destroySettings();

}

void testIMUInit(const vector<shared_ptr<Frame>  > &vpKFs) {

    int N = vpKFs.size();

    Vector3d v3zero = Vector3d::Zero();
    for (shared_ptr<Frame>  pKF: vpKFs) {
        pKF->SetBiasG(v3zero);
        pKF->SetBiasA(v3zero);
    }
    for (auto pkf : vpKFs) {
        if (pkf->mnId == 0)
            continue;
        pkf->ComputeIMUPreInt();
    }

    // Step1. gyroscope bias estimation
    //        update bg and re-compute pre-integration
    // 第一步，估计陀螺偏置
    Vector3d bgest = IMUInitEstBg(vpKFs);
    // 重新计算预积分器
    for (shared_ptr<Frame>  pKF: vpKFs) {
        pKF->SetBiasG(bgest);
    }
    for (auto pkf : vpKFs) {
        if (pkf->mnId == 0)
            continue;
        pkf->ComputeIMUPreInt();
    }

    // Step2. accelerometer bias and gravity estimation (gv = Rvw*gw)
    // Solve C*x=D for x=[gw; ba] (3+3)x1 vector
    MatrixXd C(3 * (N - 2), 6);
    C.setZero();

    VectorXd D(3 * (N - 2));
    D.setZero();

    Matrix3d I3 = Matrix3d::Identity();
    for (int i = 0; i < N - 2; i++) {

        // 三个帧才能建立加速度约束
        shared_ptr<Frame>  pKF1 = vpKFs[i];
        shared_ptr<Frame>  pKF2 = vpKFs[i + 1];
        shared_ptr<Frame>  pKF3 = vpKFs[i + 2];

        // Poses
        Matrix3d R1 = pKF1->mRwb.matrix();
        Matrix3d R2 = pKF2->mRwb.matrix();
        Vector3d p1 = pKF1->mTwb;
        Vector3d p2 = pKF2->mTwb;
        Vector3d p3 = pKF3->mTwb;

        // Delta time between frames
        double dt12 = pKF2->mIMUPreInt.getDeltaTime();
        double dt23 = pKF3->mIMUPreInt.getDeltaTime();
        // Pre-integrated measurements
        Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();
        Vector3d dv12 = pKF2->mIMUPreInt.getDeltaV();
        Vector3d dp23 = pKF3->mIMUPreInt.getDeltaP();

        Matrix3d Jpba12 = pKF2->mIMUPreInt.getJPBiasa();
        Matrix3d Jvba12 = pKF2->mIMUPreInt.getJVBiasa();
        Matrix3d Jpba23 = pKF3->mIMUPreInt.getJPBiasa();

        // 谜之计算
        Matrix3d lambda = 0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * I3;
        Matrix3d phi = R2 * Jpba23 * dt12 - R1 * Jpba12 * dt23 + R1 * Jvba12 * dt12 * dt23;
        Vector3d gamma = p3 * dt12 + p1 * dt23 + R1 * dp12 * dt23 - p2 * (dt12 + dt23)
                         - R2 * dp23 * dt12 - R1 * dv12 * dt12 * dt23;

        C.block<3, 3>(3 * i, 0) = lambda;
        C.block<3, 3>(3 * i, 3) = phi;
        D.segment<3>(3 * i) = gamma;
    }
    // Use svd to compute C*x=D, x=[gw; ba] 6x1 vector
    JacobiSVD<MatrixXd> svd2(C, ComputeThinU | ComputeThinV);
    VectorXd y = svd2.solve(D);

    Vector3d gwest = y.head(3);
    Vector3d baest = y.tail(3);


    LOG(INFO) << "Estimated gravity: " << gwest.transpose() << ", |gw| = " << gwest.norm() << endl;
    LOG(INFO) << "Estimated acc bias: " << baest.transpose() << endl;
    LOG(INFO) << "Estimated gyr bias: " << bgest.transpose() << endl;
}

void LoadImusSim(VecIMU &vImus, vector<ImagePRt> &vImagePRt) {
    string simimupath = "/home/jp/opensourcecode/ygz-stereo-inertial/examples/simimu/";
    ifstream fImus(simimupath + "imu.txt");
    ifstream fImunoise(simimupath + "imunoise.txt");

    ifstream fP(simimupath + "pos.txt");
    ifstream fR(simimupath + "rot.txt");

    vImus.reserve(3000);
    vImagePRt.reserve(3000);

    int imucnt = -1;
    while (!fImus.eof()) {

        // imu sensor data
        double timu;
        string s;
        getline(fImus, s);
        string sn;
        getline(fImunoise, sn);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double tmpd;
            int cnt = 0;
            double data[7];    // time, wx,wy,wz, ax,ay,az
            while (ss >> tmpd) {
                data[cnt] = tmpd;
                cnt++;
                if (cnt == 7)
                    break;
                if (ss.peek() == ',' || ss.peek() == ' ')
                    ss.ignore();
            }

            stringstream ssn;
            ssn << sn;
            double datan[7];
            cnt = 0;
            while (ssn >> tmpd) {
                datan[cnt] = tmpd;
                cnt++;
                if (cnt == 7)
                    break;
                if (ssn.peek() == ',' || ssn.peek() == ' ')
                    ssn.ignore();
            }

//            // imu data with no noise, tested. Result good
//            ygz::IMUData imudata(data[1], data[2], data[3],
//                                   data[4], data[5], data[6], data[0]);
//            // Result:
//            I0601 23:59:09.775097  6981 testStereoInitSim.cpp:242] Estimated gravity: -0.00992688 -0.0129066 00009.7965, |gw| = 9.79651
//            I0601 23:59:09.775177  6981 testStereoInitSim.cpp:243] Estimated acc bias: -0.00732317 0-0.0213014 -0.00344583
//            I0601 23:59:09.775209  6981 testStereoInitSim.cpp:244] Estimated gyr bias: 00.000356564 00.000349475 -7.86878e-05

//            // add noise to imu data, tested. Result good
//            ygz::IMUData imudata(data[1] + datan[1], data[2] + datan[2], data[3] + datan[3],
//                                 data[4] + datan[4], data[5] + datan[5], data[6] + datan[6],
//                                 data[0]);
//            // Result:
//            I0601 23:54:14.749037  6737 testStereoInitSim.cpp:242] Estimated gravity: -0.00854108 -0.0123049 0009.79648, |gw| = 9.79649
//            I0601 23:54:14.749126  6737 testStereoInitSim.cpp:243] Estimated acc bias: -0.00568111 0-0.0211953 -0.00413282
//            I0601 23:54:14.749156  6737 testStereoInitSim.cpp:244] Estimated gyr bias: 00.000409856 00.000410918 -0.000133576

            // add noise and bias to imu data, tested. Result good
            ygz::IMUData imudata(data[1] + datan[1] + 0.01, data[2] + datan[2]+ 0.02, data[3] + datan[3] - 0.01,
                                 data[4] + datan[4] + 0.1, data[5] + datan[5] + 0.2, data[6] + datan[6] - 0.1,
                                 data[0]);
//            // Result:
//            I0601 23:57:38.773819  6881 testStereoInitSim.cpp:242] Estimated gravity: -0.00857802 -0.0122971 0009.79649, |gw| = 9.7965
//            I0601 23:57:38.773898  6881 testStereoInitSim.cpp:243] Estimated acc bias: 0.0943534 00.178765 -0.104127
//            I0601 23:57:38.773927  6881 testStereoInitSim.cpp:244] Estimated gyr bias: 00.0104295 00.0204358 -0.0101496



            // record imu data
            vImus.push_back(imudata);
            timu = imudata.mfTimeStamp;
        }
        else {
            LOG(ERROR) << "s.empty!" <<endl;
        }


        // position
        Vector3d tmpp;
        string sp;
        getline(fP, sp);
        if (!sp.empty()) {

            stringstream ss;
            ss << sp;
            double tmpd;
            int cnt = 0;
            double data[4];    // time, wx,wy,wz, ax,ay,az
            while (ss >> tmpd) {
                data[cnt] = tmpd;
                cnt++;
                if (cnt == 4)
                    break;
                if (ss.peek() == ',' || ss.peek() == ' ')
                    ss.ignore();
            }

            tmpp = Vector3d(data[1], data[2], data[3]);
        }

        // rotation
        Sophus::SO3d tmpRso3;
        string sr;
        getline(fR, sr);
        if (!sr.empty()) {

            stringstream ss;
            ss << sr;
            double tmpd;
            int cnt = 0;
            double data[10];    // time, wx,wy,wz, ax,ay,az
            while (ss >> tmpd) {
                data[cnt] = tmpd;
                cnt++;
                if (cnt == 10)
                    break;
                if (ss.peek() == ',' || ss.peek() == ' ')
                    ss.ignore();
            }

            Matrix3d tmpR;
            tmpR << data[1], data[4], data[7],
                    data[2], data[5], data[8],
                    data[3], data[6], data[9];
            tmpRso3 = Sophus::SO3d (tmpR);
        }

        // record image P/R/t
        if(imucnt < 0) {
            imucnt = 0;
            vImagePRt.push_back(ImagePRt(timu, tmpRso3.matrix(), tmpp));
        }
        else {
            imucnt++;
        }

        if(imucnt >= 100) {
            imucnt = 0;
            vImagePRt.push_back(ImagePRt(timu, tmpRso3.matrix(), tmpp));
        }

    }

    fImus.close();
    fImunoise.close();
    fP.close();
    fR.close();
}

// Input: KeyFrame rotation Rwb
Vector3d IMUInitEstBg(const std::vector<shared_ptr<Frame>  > &vpKFs) {

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Add vertex of gyro bias, to optimizer graph
    ygz::VertexGyrBias *vBiasg = new ygz::VertexGyrBias();
    vBiasg->setEstimate(Eigen::Vector3d::Zero());
    vBiasg->setId(0);
    optimizer.addVertex(vBiasg);

    // Add unary edges for gyro bias vertex
    for (shared_ptr<Frame>  pKF : vpKFs) {
        // Ignore the first KF
        if (pKF == vpKFs.front())
            continue;

        assert(pKF->mbIsKeyFrame);

        shared_ptr<Frame>  pPrevKF = pKF->mpReferenceKF.lock();

        const IMUPreIntegration &imupreint = pKF->GetIMUPreInt();
        EdgeGyrBias *eBiasg = new EdgeGyrBias();
        eBiasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

        // measurement is not used in EdgeGyrBias
        eBiasg->dRbij = imupreint.getDeltaR();
        eBiasg->J_dR_bg = imupreint.getJRBiasg();
        eBiasg->Rwbi = pPrevKF->mRwb.matrix();
        eBiasg->Rwbj = pKF->mRwb.matrix();
        eBiasg->setInformation(imupreint.getCovPVPhi().bottomRightCorner(3, 3).inverse());
        optimizer.addEdge(eBiasg);

    }

    // It's actualy a linear estimator, so 1 iteration is enough.
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    // update bias G
    VertexGyrBias *vBgEst = static_cast<VertexGyrBias *>(optimizer.vertex(0));

    return vBgEst->estimate();
}

