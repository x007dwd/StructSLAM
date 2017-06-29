#include "ygz/Feature.h"
#include "ygz/Tracker.h"
#include "ygz/ORBMatcher.h"
#include "ygz/ORBExtractor.h"
#include "ygz/MapPoint.h"
#include "ygz/BackendInterface.h"
#include "ygz/IMUPreIntegration.h"
#include "ygz/LKFlow.h"

// g2o related
#include "ygz/G2OTypes.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ygz {

    Tracker::Tracker(const string &settingFile) {

        cv::FileStorage fSettings(settingFile, cv::FileStorage::READ);
        if (fSettings.isOpened() == false) {
            cerr << "Setting file not found." << endl;
            return;
        }

        // create camera object
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        float bf = fSettings["Camera.bf"];

        mpCam = make_shared<CameraParam>(fx, fy, cx, cy, bf);
        mpExtractor = make_shared<ORBExtractor>(ORBExtractor::ORB_SLAM2);

        mState = NO_IMAGES_YET;
    }

    Tracker::Tracker() {
        mState = NO_IMAGES_YET;
    }

    Tracker::~Tracker() {
    }

    SE3d Tracker::InsertStereo(
            const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp,
            const VecIMU &vimu) {

        mpCurrentFrame = shared_ptr<Frame>(new Frame(imRectLeft, imRectRight, timestamp, mpCam, vimu));
        mpCurrentFrame->ComputeImagePyramid();

        mvIMUSinceLastKF.insert(mvIMUSinceLastKF.end(), vimu.begin(), vimu.end());   // 追加imu数据

        // Extract and compute stereo matching in current frame
        ORBExtractor extractor(ORBExtractor::OPENCV_GFTT);
        extractor.Detect(mpCurrentFrame, true);
        extractor.Detect(mpCurrentFrame, false);

        ORBMatcher matcher;
        matcher.ComputeStereoMatches(mpCurrentFrame);

        mpCurrentFrame->AssignFeaturesToGrid();
        mpCurrentFrame->ComputeBoW();

        LOG(INFO) << "\n\n********* Tracking frame " << mpCurrentFrame->mnId << " **********" << endl;

        Track();
        LOG(INFO) << "Tracker returns. \n\n" << endl;
        return mpCurrentFrame->GetPose();
    }

    // ------------------------------------------
    // 主要Track函数
    void Tracker::Track() {
        // 这里有比较麻烦的流程

        int TrackInliersCnt = 0;

        if (mState == NO_IMAGES_YET) {
            // 尝试通过视觉双目构建初始地图
            if (mpCurrentFrame->mFeaturesLeft.size() < setting::minStereoInitFeatures) {
                LOG(INFO) << "Feature is not enough: " << mpCurrentFrame->mFeaturesLeft.size() << endl;
                return;
            }

            // TODO 检查这个图像能否初始化双目
            if (StereoInitialization() == false) {
                LOG(INFO) << "Stereo init failed." << endl;
                return;
            }

            // 向后端添加关键帧
            InsertKeyFrame();

            mpLastFrame = mpCurrentFrame;
            mpLastKeyFrame = mpCurrentFrame;

            // 置状态为等待初始化状态
            mState = NOT_INITIALIZED;

            return;

        } else if (mState == NOT_INITIALIZED) {

            // IMU未初始化时，位姿预测不靠谱，先用纯视觉追踪
            bool bOK = false;
            mpCurrentFrame->SetPose(mpLastFrame->GetPose());  // same with last, or use speed model

            bOK = TrackLastFrame(false);
            if (bOK == false) {
                LOG(INFO) << "Track last frame failed." << endl;
                bOK = TrackReferenceKF(bOK);
            }

            if (bOK) {
                bOK = TrackLocalMap(TrackInliersCnt);
            }

            if (bOK == false) {
                // 纯视觉追踪失败了，reset整个系统
                // 我们并不希望出现这个结果
                Reset();
                return;
            }

            if (NeedNewKeyFrame(TrackInliersCnt))
                InsertKeyFrame();

            // 尝试带着IMU初始化
            if (mbVisionOnlyMode == false) {
                if (IMUInitialization() == true) {
                    // 初始化成功，置状态为OK
                    mState = OK;
                }
            }

            mpLastFrame = mpCurrentFrame;
            return;

        } else {
            // state == OK or WEAK
            bool bOK = false;
            if (mState == OK) {
                // 正常追踪
                // 用imu预测位姿，再尝试用预测的位姿进行特征匹配
                PredictCurrentPose();

                // 与上一个帧比较
                bOK = TrackLastFrame();
                if (bOK)
                    TrackLastFrame(false);    // 位姿可能给的不好，尝试用视觉信息进行追踪

                // 如果还不行，可能是lastFrame不行，与上一个关键帧对比
                if (bOK == false)
                    bOK = TrackReferenceKF(false);

            } else {
                // 视觉追踪不好
                PredictCurrentPose();   // 这步可能不可靠

                bOK = TrackLastFrame(false);
                if (bOK == false)
                    bOK = TrackReferenceKF(false);
            }

            if (bOK)
                bOK = TrackLocalMap(TrackInliersCnt);

            // 判定状态
            if (bOK) {
                mState = OK;
            } else {
                mState = WEAK;
            }

            // 处理关键帧
            if (NeedNewKeyFrame(TrackInliersCnt))
                InsertKeyFrame();

            if (bOK) {
                mpLastFrame = mpCurrentFrame;
            }

            return;
        }
    }

    void Tracker::PredictCurrentPose() {

        // step 1. 计算从上一个关键帧到当前帧的IMU积分
        // TODO 如果假设了关键帧bias不动，这里似乎没必要重新算起？
        IMUPreIntegration imuPreInt;

        Vector3d bg = mpLastKeyFrame->BiasG();
        Vector3d ba = mpLastKeyFrame->BiasA();

        // remember to consider the gap between the last KF and the first IMU
        const IMUData &imu = mvIMUSinceLastKF.front();
        double dt = std::max(0.0, imu.mfTimeStamp - mpLastKeyFrame->mTimeStamp);
        imuPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);

        // integrate each imu
        for (size_t i = 0; i < mvIMUSinceLastKF.size(); i++) {
            const IMUData &imu = mvIMUSinceLastKF[i];
            double nextt = 0;

            if (i == (mvIMUSinceLastKF.size() - 1))
                nextt = mpCurrentFrame->mTimeStamp;    // last IMU, next is this KeyFrame
            else
                nextt = mvIMUSinceLastKF[i + 1].mfTimeStamp;  // regular condition, next is imu data

            // delta time
            double dt = std::max(0.0, nextt - imu.mfTimeStamp);
            // update pre-integrator
            imuPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }

        // step 2. 用预积分估算当前帧位姿
        mpCurrentFrame->UpdatePoseFromPreintegration(imuPreInt, mgWorld);
    }

    bool Tracker::TrackLastFrame(bool usePoseInfomation) {

        // 参考帧的信息可能已经被更新，用它更新上一帧的pose
        UpdateLastFrame();

        // LOG(INFO) << "Tracking " << mpCurrentFrame->mnId << " with last: " << mpLastFrame->mnId << endl;

        // 用上一帧的数据跟踪当前帧
        ORBMatcher matcher(0.9, true);
        int nmatches = 0;
        if (usePoseInfomation) {

            nmatches = matcher.SearchByProjection(mpCurrentFrame, mpLastFrame, 7);
            if (nmatches < setting::minTrackLastFrameFeatures)    // 匹配不够，尝试扩大范围
                nmatches = matcher.SearchByProjection(mpCurrentFrame, mpLastFrame, 15);

            if (nmatches < setting::minTrackLastFrameFeatures) {
                // 依然不够
                //LOG(INFO)<<"Match with last frame: "<<nmatches<<", return false"<<endl;
                return false;
            }

            OptimizeCurrentPose();

        } else {

            vector<Match> matches;
            matcher.SearchByBoW(mpLastFrame, mpCurrentFrame, matches, true);
            nmatches = matches.size();

            // set the matched points
            for (Match &m: matches) {
                shared_ptr<Feature> feat1 = mpLastFrame->mFeaturesLeft[m.index1];
                shared_ptr<Feature> feat2 = mpCurrentFrame->mFeaturesLeft[m.index2];
                feat2->mpPoint = feat1->mpPoint;
            }

            if (matches.size() < setting::minTrackLastFrameFeatures) {
                //LOG(INFO)<<"Match with last frame: "<<nmatches<<", return false"<<endl;
                return false;
            }

            nmatches = OptimizeCurrentPose();
        }

        // remove the outliers
        for (shared_ptr<Feature> &feature: mpCurrentFrame->mFeaturesLeft) {
            if (feature->mbOutlier) {
                feature->mpPoint = nullptr;
                feature->mbOutlier = false;
            }
        }

        //LOG(INFO) << "Track last frame inliers: " << nmatches << endl;

        if (nmatches >= setting::minTrackLastFrameFeatures) {
            return true;
        } else {
            //LOG(INFO) << "Inlier is small: " << nmatches << endl;
            return false;
        }
    }

    bool Tracker::TrackReferenceKF(bool usePoseInfomation) {

        // 与参考关键帧进行比较
        ORBMatcher matcher(0.7, false);

        vector<Match> matches;
        matcher.SearchByBoW(mpLastKeyFrame, mpCurrentFrame, matches);
        int nmatches = matches.size();

        if (nmatches < setting::minTrackRefKFFeatures) {
            return false;
        }

        // 关联地图点
        for (Match &m: matches) {
            auto f1 = mpLastKeyFrame->mFeaturesLeft[m.index1];
            auto f2 = mpCurrentFrame->mFeaturesLeft[m.index2];

            if (f1->mpPoint)
                f2->mpPoint = f1->mpPoint;
        }

        OptimizeCurrentPose();

        // Discard outliers
        int nMatchesMap = 0;

        for (auto feature: mpCurrentFrame->mFeaturesLeft) {
            if (!feature->mpPoint) {
                feature->mbOutlier = false;
                continue;
            } else if (feature->mpPoint && feature->mbOutlier) {
                feature->mpPoint = nullptr;
                feature->mbOutlier = false;
            } else if (feature->mpPoint->Observations() > 0) {
                nMatchesMap++;
            }
        }

        return nMatchesMap >= setting::minTrackRefKFFeatures;
    }

    bool Tracker::TrackLocalMap(int &inliers) {
        // Step 1. 从Local Map中投影到当前帧
        set<shared_ptr<MapPoint>> localmap = mpBackEnd->GetLocalMap();
        for (auto mp: localmap)
            mp->mbTrackInView = false;
        for (auto feat: mpCurrentFrame->mFeaturesLeft)
            if (feat->mpPoint && feat->mbOutlier == false)
                feat->mpPoint->mbTrackInView = true;

        // 筛一下视野内的点
        set<shared_ptr<MapPoint> > mpsInView;
        for (auto mp: localmap) {
            if (mpCurrentFrame->isInFrustum(mp, 0.5)) {
                mpsInView.insert(mp);
            }
        }

        //LOG(INFO) << "Mappoints in View: " << mpsInView.size() << endl;

        ORBMatcher matcher(0.8);
        int th = 1; // 窗口大小倍率
        int nMatches = matcher.SearchByProjection(mpCurrentFrame, mpsInView, th);

        // Optimize Pose
        int optinliers;
        if (mState == OK)
            optinliers = OptimizeCurrentPoseWithIMU();
        else
            optinliers = OptimizeCurrentPoseWithoutIMU();

        /*int */inliers = 0;

        // Update MapPoints Statistics
        for (shared_ptr<Feature> feat : mpCurrentFrame->mFeaturesLeft) {
            if (feat->mpPoint) {
                if (!feat->mbOutlier) {
                    feat->mpPoint->IncreaseFound();
                    if (feat->mpPoint->Observations() > 0)   // only count the mappoints with at least 1 observations
                        inliers++;
                } else {
                    // 删掉对此地图点的引用
                    feat->mpPoint = nullptr;
                }
            }
        }

        if (inliers != optinliers)
            LOG(ERROR) << "Track local map inliers vs optinliers: " << inliers << "/" << optinliers << endl;

        // Decide if the tracking was succesful
        if (inliers < setting::minTrackLocalMapInliers)
            return false;
        else
            return true;

    }

    bool Tracker::NeedNewKeyFrame(const int &trackinliers) {

        //LOG(INFO)<<"Testing if need new keyframe"<<endl;

        int nKFs = mpBackEnd->GetAllKF().size();

        // matches in reference KeyFrame
        int nMinObs = 2;
        if (nKFs <= 2)
            nMinObs = nKFs;
        int nRefMatches = mpLastKeyFrame->TrackedMapPoints(nMinObs);

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;
        for (int i = 0; i < mpCurrentFrame->mFeaturesLeft.size(); i++) {
            shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[i];
            if (feat->mfInvDepth > setting::minNewMapPointInvD && feat->mfInvDepth < setting::maxNewMapPointInvD) {
                if (feat->mpPoint != nullptr && !feat->mbOutlier)
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
        bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);


        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        //Condition 1c: tracking is weak
        const bool c1 = trackinliers < nRefMatches * 0.25 || bNeedToInsertClose;

        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = ((trackinliers < nRefMatches * thRefRatio || bNeedToInsertClose) && trackinliers > 15);

        double timegap = setting::keyframeTimeGapTracking;
        // if (mState != OK && !mbVisionOnlyMode)
        if ( mState != OK || mbVisionOnlyMode )
            timegap = setting::keyframeTimeGapInit;
        bool cTimeGap = (mpCurrentFrame->mTimeStamp - mpLastKeyFrame->mTimeStamp) >= timegap;

        bool isBackendBusy = mpBackEnd->IsBusy();

        //LOG(INFO) << "nTrackedClose vs nNonTrackedClose: " << nTrackedClose << " vs " << nNonTrackedClose << endl;
        //LOG(INFO) << "track inliers vs nRefMatches: " << trackinliers << " vs " << nRefMatches << endl;
        //LOG(INFO) << "bNeedToInsertClose / c1 / c2 / cTimeGap / !isBackendBusy: "
        //          << bNeedToInsertClose << "/" << c1 << "/" << c2 << "/" << cTimeGap << "/" << !isBackendBusy << endl;

        if ((c1 || c2 || cTimeGap) && !isBackendBusy)
            return true;

        if (mState == NOT_INITIALIZED) {

        } else if (mState == OK) {
            // 正常选择
        } else if (mState == WEAK) {
            // 视觉弱时，也尽量多选一些关键帧
        }

        return false;
    }

    void Tracker::UpdateLastFrame() {

    }

    void Tracker::InsertKeyFrame() {
        //LOG(INFO)<<"Call insert key frame"<<endl;
        mpCurrentFrame->SetThisAsKeyFrame();

        // 处理IMU数据，将Tracker累积的IMU传递至current
        mpCurrentFrame->mvIMUDataSinceLastFrame = mvIMUSinceLastKF;

        // 清空现有的IMU
        mvIMUSinceLastKF.clear();

        mpCurrentFrame->mpReferenceKF = mpLastKeyFrame;
        if (mpLastKeyFrame)
            mpCurrentFrame->ComputeIMUPreIntSinceLastFrame(mpLastKeyFrame, mpCurrentFrame->mIMUPreInt);

        mpLastKeyFrame = mpCurrentFrame;

        // 将当前帧作为新的关键帧插入到后端，这一步会创建一些地图点
        mpBackEnd->InsertKeyFrame(mpCurrentFrame);

        //LOG(INFO) << "Insert keyframe done." << endl;
    }

    bool Tracker::StereoInitialization() {
        // 双目初始化
        // 将双目匹配的点拉出来即可
        int cntValidFeat = 0;
        for (auto feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mfInvDepth > 0)
                cntValidFeat++;
        }
        if (cntValidFeat < setting::minValidInitFeatures) {
            //LOG(INFO) << "Valid feature is not enough: " << cntValidFeat << endl;
            return false;
        }

        // 创建新地图点
        for (size_t i = 0; i < mpCurrentFrame->mFeaturesLeft.size(); i++) {
            auto feat = mpCurrentFrame->mFeaturesLeft[i];
            if (feat->mfInvDepth > 0) {
                shared_ptr<MapPoint> mp = make_shared<MapPoint>(mpCurrentFrame, i);   // 创建新地图点
                feat->mpPoint = mp;
            }
        }

        // set first frame's pose as Twb = I, then Rwb=Rwc*Rcb, twb=Rwc*tcb+twb
        // mpCurrentFrame->SetPose(Sophus::SE3d());

        return true;
    }

    // Input: KeyFrame rotation Rwb
    Vector3d Tracker::IMUInitEstBg(const std::deque<shared_ptr<Frame>> &vpKFs) {

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
        shared_ptr<Frame> pPrevKF0 = vpKFs.front();
        for (auto pKF : vpKFs) {
            // Ignore the first KF
            if (pKF == vpKFs.front())
                continue;

            assert(pKF->mbIsKeyFrame);

            shared_ptr<Frame> pPrevKF = pKF->mpReferenceKF.lock();

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

            pPrevKF0 = pKF;
        }

        // It's actualy a linear estimator, so 1 iteration is enough.
        optimizer.initializeOptimization();
        optimizer.optimize(1);

        // update bias G
        VertexGyrBias *vBgEst = static_cast<VertexGyrBias *>(optimizer.vertex(0));

        return vBgEst->estimate();
    }

    bool Tracker::IMUInitialization() {
        // IMU 初始化 需要由后端提供关键帧（不需要地图）
        //LOG(INFO) << "Try IMU init" << endl;

        if (!mpCurrentFrame->IsKeyFrame())
            return false;

        std::deque<shared_ptr<Frame>> vpKFs = mpBackEnd->GetAllKF();
        int N = vpKFs.size();
        if (N < setting::minInitKFs)    // 初始化需要若干个关键帧
            return false;

        // Note 1.
        // Input : N (N>=4) KeyFrame/Frame poses of stereo vslam
        //         Assume IMU bias are identical for N KeyFrame/Frame
        // Compute :
        //          bg: gyroscope bias
        //          ba: accelerometer bias
        //          gv: gravity in vslam frame
        //          Vwbi: velocity of N KeyFrame/Frame
        // (Absolute scale is available for stereo vslam)

        // Note 2.
        // Convention by wangjing:
        // W: world coordinate frame (z-axis aligned with gravity, gW=[0;0;~9.8])
        // B: body coordinate frame (IMU)
        // V: camera frame of first camera (vslam's coordinate frame)
        // TWB/T : 6dof pose of frame, TWB = [RWB, PWB; 0, 1], XW = RWB*XW + PWB
        // RWB/R : 3dof rotation of frame's body(IMU)
        // PWB/P : 3dof translation of frame's body(IMU) in world
        // XW/XB : 3dof point coordinate in world/body

        // Step0. get all keyframes in map
        //        reset v/bg/ba to 0
        //        re-compute pre-integration

        Vector3d v3zero = Vector3d::Zero();
        for (auto pKF: vpKFs) {
            pKF->SetBiasG(v3zero);
            pKF->SetBiasA(v3zero);
        }
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
        }

        // Step1. gyroscope bias estimation
        //        update bg and re-compute pre-integration
        // 第一步，估计陀螺偏置
        Vector3d bgest = IMUInitEstBg(vpKFs);
        // 重新计算预积分器
        for (auto pKF: vpKFs) {
            pKF->SetBiasG(bgest);
        }
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
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
            shared_ptr<Frame> pKF1 = vpKFs[i];
            shared_ptr<Frame> pKF2 = vpKFs[i + 1];
            shared_ptr<Frame> pKF3 = vpKFs[i + 2];

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


        // Step2.2
        Vector3d gpre = y.head(3);
        Vector3d g0 = gpre / gpre.norm() * setting::gravity;
        Matrix3d g0hat = Sophus::SO3d::hat(g0);

        MatrixXd A(3 * (N - 2), 5);
        A.setZero();
        VectorXd B(3 * (N - 2));
        B.setZero();
        for (int i = 0; i < N - 2; i++) {

            // 三个帧才能建立加速度约束
            shared_ptr<Frame> pKF1 = vpKFs[i];
            shared_ptr<Frame> pKF2 = vpKFs[i + 1];
            shared_ptr<Frame> pKF3 = vpKFs[i + 2];

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

            Matrix3d newlambda = -lambda * g0hat;
            Vector3d newgamma = gamma - lambda * g0;

            A.block<3, 2>(3 * i, 0) = newlambda.block<3, 2>(0, 0);
            A.block<3, 3>(3 * i, 2) = phi;
            B.segment<3>(3 * i) = newgamma;
        }
        // Use svd to compute A*x=B, x=[thetax, thetay; ba] 6x1 vector
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        VectorXd y2 = svd.solve(B);

        Vector3d baest0 = y.tail(3);
        Vector3d baest = y2.tail(3);

        Vector2d thetaxy = y2.head(2);
        Vector3d theta(thetaxy[0], thetaxy[1], 0);
        Vector3d gvest = Sophus::SO3d::exp(theta) * g0;


        // update ba and re-compute pre-integration
        for (auto pkf : vpKFs) {
            pkf->SetBiasA(baest);
        }
        for (int i = 1; i < N; i++) {
            vpKFs[i]->ComputeIMUPreInt();
        }

        // Step3. velocity estimation
        for (int i = 0; i < N; i++) {
            auto pKF = vpKFs[i];
            if (i != N - 1) {
                // not last KeyFrame, R1*dp12 = p2 - p1 -v1*dt12 - 0.5*gw*dt12*dt12
                //  ==>> v1 = 1/dt12 * (p2 - p1 - 0.5*gw*dt12*dt12 - R1*dp12)

                auto pKF2 = vpKFs[i + 1];
                const Vector3d p2 = pKF2->mTwb;
                const Vector3d p1 = pKF->mTwb;
                const Matrix3d R1 = pKF->mRwb.matrix();
                const double dt12 = pKF2->mIMUPreInt.getDeltaTime();
                const Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();

                Vector3d v1 = (p2 - p1 - 0.5 * gvest * dt12 * dt12 - R1 * dp12) / dt12;
                pKF->SetSpeed(v1);
            } else {
                // last KeyFrame, R0*dv01 = v1 - v0 - gw*dt01 ==>> v1 = v0 + gw*dt01 + R0*dv01
                auto pKF0 = vpKFs[i - 1];
                const Matrix3d R0 = pKF0->mRwb.matrix();
                const Vector3d v0 = pKF0->mSpeedAndBias.segment<3>(0);
                const double dt01 = pKF->mIMUPreInt.getDeltaTime();
                const Vector3d dv01 = pKF->mIMUPreInt.getDeltaV();

                Vector3d v1 = v0 + gvest * dt01 + R0 * dv01;
                pKF->SetSpeed(v1);
            }
        }

        static Vector3d baestpre(0, 0, 0);
        double gprenorm = gpre.norm();
        double baestdif = (baest0 - baest).norm();

        LOG(INFO) << "Estimated gravity before: " << gpre.transpose() << ", |gw| = " << gpre.norm() << endl;
        LOG(INFO) << "Estimated gravity after: " << gvest.transpose() << ", |gw| = " << gvest.norm() << endl;
        LOG(INFO) << "Estimated acc bias before: " << baest0.transpose() << endl;
        LOG(INFO) << "Estimated acc bias after: " << baest.transpose() << endl;
        LOG(INFO) << "Estimated gyr bias: " << bgest.transpose() << endl;
        LOG(INFO) << "gprenorm: " << gprenorm << ", baestdif: " << baestdif << endl;
        LOG(INFO) << "baest change: " << (baest - baestpre).norm() << endl;

        // TODO 判断这个是否正确，若正确请更新整张地图
        // TODO: how to choose success condition ??????
        static int goodcnt = 0;
        bool initflag = false;
        if (gprenorm > 9.7 && gprenorm < 9.9 && baestdif < 0.2 &&
            baest.norm() < 1 && (baest - baestpre).norm() < 0.2) {
            goodcnt++;
            if (goodcnt >= 2)
                initflag = true;
        } else {
            goodcnt = 0;
        }
        baestpre = baest;


        // align 'world frame' to gravity vector, making mgWorld = [0,0,9.8]
        if (initflag) {
            // compute Rvw
            Vector3d gw1(0, 0, 1);
            Vector3d gv1 = gvest / gvest.norm();
            Vector3d gw1xgv1 = gw1.cross(gv1);
            Vector3d vhat = gw1xgv1 / gw1xgv1.norm();
            double theta = std::atan2(gw1xgv1.norm(), gw1.dot(gv1));
            Matrix3d Rvw = Sophus::SO3d::exp(vhat * theta).matrix();
            Matrix3d Rwv = Rvw.transpose();
            Sophus::SE3d Twv(Rwv, Vector3d::Zero());
            // 设置重力
            Vector3d gw = Rwv * gvest;
            mgWorld = gw;
            LOG(INFO) << "mgWorld: " << mgWorld.transpose() << endl;

            // rotate pose/rotation/velocity to align with 'world' frame
            for (int i = 0; i < N; i++) {
                auto pKF = vpKFs[i];
                Sophus::SE3d Tvb = pKF->GetPose();
                Vector3d Vvb = pKF->Speed();
                // set pose/speed/biasg/biasa
                pKF->SetPose(Twv * Tvb);
                pKF->SetSpeed(Rwv * Vvb);
                pKF->SetBiasG(bgest);
                pKF->SetBiasA(baest);
            }
            // re-compute pre-integration for KeyFrame (except for the first KeyFrame)
            for (int i = 1; i < N; i++) {
                vpKFs[i]->ComputeIMUPreInt();
            }

            // MapPoints
            auto vsMPs = mpBackEnd->GetLocalMap();
            for (auto mp : vsMPs) {
                Vector3d Pv = mp->GetWorldPos();
                Vector3d Pw = Rwv * Pv;
                mp->SetWorldPos(Pw);
            }
        }


        return initflag;
    }

    int Tracker::OptimizeCurrentPose() {

        assert(mState == OK || mState == WEAK || mState == NOT_INITIALIZED);

        if (mState == OK) {
            // 正常追踪时，可以带有IMU约束
            return OptimizeCurrentPoseWithIMU();
        } else {
            // WEAK 或 初始化时，不要带IMU
            return OptimizeCurrentPoseWithoutIMU();
        }
    }

    // 带IMU的优化
    // 考虑上一个关键帧到当前帧的优化关系
    int Tracker::OptimizeCurrentPoseWithIMU() {

        assert(mpCurrentFrame != nullptr);
        assert(mpLastKeyFrame != nullptr);
        assert(mvIMUSinceLastKF.size() != 0);

        IMUPreIntegration imupreint = GetIMUFromLastKF();

        // Extrinsics
        SE3d Tcb = setting::TBC.inverse();

        // setup g2o
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set current frame vertex PVR/Bias
        // 当前帧需要 P+R+V
        // P+R
        VertexPR *vPR = new VertexPR();
        vPR->setEstimate(mpCurrentFrame->PR());
        vPR->setId(0);
        optimizer.addVertex(vPR);

        // Speed
        VertexSpeed *vSpeed = new VertexSpeed();
        vSpeed->setEstimate(mpCurrentFrame->Speed());
        vSpeed->setId(1);
        optimizer.addVertex(vSpeed);

        // Bg, Ba
        VertexGyrBias *vBiasG = new VertexGyrBias();
        vBiasG->setEstimate(mpCurrentFrame->BiasG());
        vBiasG->setId(2);
        optimizer.addVertex(vBiasG);

        VertexAcceBias *vBiasA = new VertexAcceBias();
        vBiasA->setEstimate(mpCurrentFrame->BiasA());
        vBiasA->setId(3);
        optimizer.addVertex(vBiasA);


        // 来自上一个关键帧的东西
        // 上一个关键帧需要 P+R+V+Bg+Ba，但不参与优化，全部 fix // TODO 加Prior是不是更优雅？
        // P+R
        VertexPR *vPRL = new VertexPR();
        vPRL->setEstimate(mpLastKeyFrame->PR());
        vPRL->setId(4);
        vPRL->setFixed(true);
        optimizer.addVertex(vPRL);

        // Speed
        VertexSpeed *vSpeedL = new VertexSpeed();
        vSpeedL->setEstimate(mpLastKeyFrame->Speed());
        vSpeedL->setId(5);
        vSpeedL->setFixed(true);
        optimizer.addVertex(vSpeedL);

        // Bg
        VertexGyrBias *vBiasGL = new VertexGyrBias();
        vBiasGL->setEstimate(mpLastKeyFrame->BiasG());
        vBiasGL->setId(6);
        vBiasGL->setFixed(true);
        optimizer.addVertex(vBiasGL);

        VertexAcceBias *vBiasAL = new VertexAcceBias();
        vBiasAL->setEstimate(mpLastKeyFrame->BiasA());
        vBiasAL->setId(7);
        vBiasAL->setFixed(true);
        optimizer.addVertex(vBiasAL);

        // Edges
        // Set PVR edge between LastKF-Frame
        // 顺序见EdgePRV定义
        EdgePRV *ePRV = new EdgePRV(mgWorld);
        ePRV->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPRL));
        ePRV->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPR));
        ePRV->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSpeedL));
        ePRV->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSpeed));
        ePRV->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasGL));
        ePRV->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasAL));
        ePRV->setMeasurement(imupreint);
        // information matrix
        Matrix9d InvCovPVR = imupreint.getCovPVPhi().inverse();
        ePRV->setInformation(InvCovPVR);
        // robust kernel
        const float thHuberNavStatePVR = sqrt(21.666);
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        ePRV->setRobustKernel(rk);
        rk->setDelta(thHuberNavStatePVR);
        optimizer.addEdge(ePRV);

        // bias 的随机游走，用两条边来约束
        EdgeBiasG *eBG = new EdgeBiasG();
        eBG->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasGL));
        eBG->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasG));
        eBG->setMeasurement(Vector3d::Zero());   // bias受白噪声影响
        Matrix3d infoBG = Matrix3d::Identity() * setting::gyrBiasRw2;
        eBG->setInformation(infoBG / imupreint.getDeltaTime());
        float thHuberNavStateBias = sqrt(16.812);
        g2o::RobustKernelHuber *rkb = new g2o::RobustKernelHuber;
        eBG->setRobustKernel(rkb);
        rkb->setDelta(thHuberNavStateBias);
        optimizer.addEdge(eBG);

        EdgeBiasA *eBA = new EdgeBiasA();
        eBA->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasAL));
        eBA->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vBiasA));
        eBA->setMeasurement(Vector3d::Zero());   // bias受白噪声影响
        Matrix3d infoBA = Matrix3d::Identity() * setting::accBiasRw2;
        eBA->setInformation(infoBA / imupreint.getDeltaTime());
        g2o::RobustKernelHuber *rkba = new g2o::RobustKernelHuber;
        eBA->setRobustKernel(rkba);
        rkba->setDelta(thHuberNavStateBias);
        optimizer.addEdge(eBA);

        // Set MapPoint vertices
        const int N = mpCurrentFrame->mFeaturesLeft.size();

        vector<EdgeProjectPoseOnly *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        const float delta = sqrt(5.991);

        int nInitialCorrespondences = 0; // 最初拥有的合法点
        for (size_t i = 0; i < N; i++) {
            shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[i];
            shared_ptr<MapPoint> pMP = feat->mpPoint;

            if (pMP) {
                // 添加投影量
                nInitialCorrespondences++;
                feat->mbOutlier = false;

                EdgeProjectPoseOnly *eProj = new EdgeProjectPoseOnly(mpCam.get(), pMP->GetWorldPos());
                eProj->setVertex(0, vPR);
                eProj->setInformation(Matrix2d::Identity() * setting::invLevelSigma2[feat->mLevel]);
                eProj->setMeasurement(feat->mPixel.cast<double>());
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                eProj->setRobustKernel(rk);
                rk->setDelta(delta);
                optimizer.addEdge(eProj);

                vpEdgesMono.push_back(eProj);
                vnIndexEdgeMono.push_back(i);
            }
        }

        // 处理优化，保留ORB2的四遍优化策略
        const float chi2th[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            // Reset estimate for vertex
            vPR->setEstimate(mpCurrentFrame->PR());
            vBiasG->setEstimate(mpCurrentFrame->BiasG());
            vBiasA->setEstimate(mpCurrentFrame->BiasA());

            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            // 统计 outlier
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                EdgeProjectPoseOnly *e = vpEdgesMono[i];
                const size_t idx = vnIndexEdgeMono[i];
                shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[idx];
                if (feat->mbOutlier == true) {
                    // 对于ouliter，在当前估计下重新计算一下误差
                    // 当然inlier在优化过程中就已经计算误差了
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2th[it]) {
                    feat->mbOutlier = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    feat->mbOutlier = false;
                    e->setLevel(0);
                }

                if (it == 2) {
                    // 后两次迭代不要使用robust
                    e->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        int inliers = nInitialCorrespondences - nBad;
        if (inliers < setting::minPoseOptimizationInliers) {
            LOG(WARNING) << "inliers is small, pose may by unreliable: " << inliers << endl;
        }

        mpCurrentFrame->SetPose(SE3d(vPR->R(), vPR->t()));
        mpCurrentFrame->SetSpeedBias(vSpeed->estimate(), vBiasG->estimate(), vBiasA->estimate());

        // Compute marginalized Hessian H and B, H*x=B, H/B can be used as prior for next optimization in PoseOptimization
        if (setting::trackerComputeMarginals) {
            // TODO: 计算current的边缘化
            // 然而。。。G2O由于没法给所有量加prior。。所以边缘化了似乎也用处不大。。。
        }

        // 为了显示效果，我们再计算一下所有观测点的深度，这样viewer里看起来比较均匀
        for (shared_ptr<Feature> feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mpPoint && feat->mbOutlier == false && feat->mfInvDepth < 0) {
                Vector3d pw = feat->mpPoint->mWorldPos;
                feat->mfInvDepth = 1.0 / (mpCurrentFrame->mRcw * pw + mpCurrentFrame->mtcw)[2];
            }
        }

        return inliers;
    }

    // 不带IMU的优化
    int Tracker::OptimizeCurrentPoseWithoutIMU() {

        // 不带IMU的清爽很多，只要优化当前帧的PR即可
        assert(mpCurrentFrame != nullptr);
        LOG(INFO) << "Optimizing frame " << mpCurrentFrame->mnId << endl;

        // setup g2o
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        // optimizer.setVerbose(true);

        // 当前帧的P+R
        VertexPR *vPR = new VertexPR();
        vPR->setEstimate(mpCurrentFrame->PR());
        vPR->setId(0);
        optimizer.addVertex(vPR);

        // 加入投影点
        // Set MapPoint vertices
        const int N = mpCurrentFrame->mFeaturesLeft.size();

        vector<EdgeProjectPoseOnly *> vpEdgesProj;
        vector<size_t> vnIndexEdges;
        vpEdgesProj.reserve(N);
        vnIndexEdges.reserve(N);

        const float delta = sqrt(5.991);

        int nInitialCorrespondences = 0; // 最初拥有的合法点
        for (size_t i = 0; i < N; i++) {
            shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[i];
            shared_ptr<MapPoint> pMP = feat->mpPoint;

            if (pMP) {
                // The points with only one observation are useless.
                if ( pMP->isBad() || pMP->Observations() < 1)
                    continue;

                // 添加投影量
                nInitialCorrespondences++;
                feat->mbOutlier = false;

                EdgeProjectPoseOnly *eProj = new EdgeProjectPoseOnly(mpCam.get(), pMP->GetWorldPos());

                eProj->setVertex(0, vPR);
                eProj->setInformation(Matrix2d::Identity() * setting::invLevelSigma2[feat->mLevel]);
                eProj->setMeasurement(feat->mPixel.cast<double>());

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                eProj->setRobustKernel(rk);
                rk->setDelta(delta);
                optimizer.addEdge(eProj);
                vpEdgesProj.push_back(eProj);
                vnIndexEdges.push_back(i);

            }
        }

        // 处理优化，保留ORB2的四遍优化策略
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {
            // Reset estimate for vertex
            vPR->setEstimate(mpCurrentFrame->PR());

            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            // 统计 outlier
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesProj.size(); i < iend; i++) {
                EdgeProjectPoseOnly *e = vpEdgesProj[i];
                const size_t idx = vnIndexEdges[i];
                shared_ptr<Feature> feat = mpCurrentFrame->mFeaturesLeft[idx];
                if (feat->mbOutlier == true) {
                    // 对于ouliter，在当前估计下重新计算一下误差
                    // 当然inlier在优化过程中就已经计算误差了
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it]) {
                    feat->mbOutlier = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    feat->mbOutlier = false;
                    e->setLevel(0);
                }

                if (it == 2) {
                    // 后两次迭代不要使用robust
                    e->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        int inliers = nInitialCorrespondences - nBad;
        if (inliers < setting::minPoseOptimizationInliers) {
            LOG(WARNING) << "inliers is small, pose may by unreliable: " << inliers << endl;
        } else {
            // recover current pose
            mpCurrentFrame->SetPose(SE3d(vPR->R(), vPR->t()));
        }

        // Compute marginalized Hessian H and B, H*x=B, H/B can be used as prior for next optimization in PoseOptimization
        if (setting::trackerComputeMarginals) {
            // TODO: 计算current的边缘化
            // 然而。。。G2O由于没法给所有量加prior。。所以边缘化了似乎也用处不大。。。
        }

        // 为了显示效果，我们再计算一下所有观测点的深度，这样viewer里看起来比较均匀
        for (shared_ptr<Feature> feat: mpCurrentFrame->mFeaturesLeft) {
            if (feat->mpPoint && feat->mbOutlier == false && feat->mfInvDepth < 0) {
                Vector3d pw = feat->mpPoint->mWorldPos;
                feat->mfInvDepth = 1.0 / (mpCurrentFrame->mRcw * pw + mpCurrentFrame->mtcw)[2];
            }
        }
        return inliers;

    }


    // 计算从上一个关键帧来的IMU预积分
    // 这个和Frame里的那个有啥区别么。。。。说好的代码重用呢
    IMUPreIntegration Tracker::GetIMUFromLastKF() {
        assert(mpLastKeyFrame != nullptr);
        assert(mvIMUSinceLastKF.size() != 0);

        // Reset pre-integrator first
        IMUPreIntegration IMUPreInt;

        Vector3d bg = mpLastKeyFrame->BiasG();
        Vector3d ba = mpLastKeyFrame->BiasA();

        const IMUData &imu = mvIMUSinceLastKF.front();
        double dt = imu.mfTimeStamp - mpLastKeyFrame->mTimeStamp;
        IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);

        // integrate each imu
        for (size_t i = 0; i < mvIMUSinceLastKF.size(); i++) {
            const IMUData &imu = mvIMUSinceLastKF[i];
            double nextt;
            if (i == mvIMUSinceLastKF.size() - 1)
                nextt = mpCurrentFrame->mTimeStamp;         // last IMU, next is this KeyFrame
            else
                nextt = mvIMUSinceLastKF[i + 1].mfTimeStamp;  // regular condition, next is imu data

            // delta time
            double dt = nextt - imu.mfTimeStamp;
            // update pre-integrator
            IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);

        }

        return IMUPreInt;
    }

    void Tracker::Reset() {
        // TODO 重置整个Tracker
    }
}
