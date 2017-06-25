#include "ygz/DSOCoarseTracker.h"
#include "ygz/Frame.h"
#include "ygz/SparePoint.h"
#include "ygz/MapPoint.h"
#include "ygz/PixelSelector.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "ygz/ORBExtractor.h"
#include "ygz/ORBMatcher.h"
#include "ygz/LKFlow.h"
#include "ygz/Feature.h"
#include "ygz/BackendSlidingWindowG2O.h"

namespace ygz {


// allocate an aligned array, put the pointer into rawPtrVec, and then return
// the aligned pointer
    template<int b, typename T>
    T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
        const int padT = 1 + ((1 << b) / sizeof(T));
        T *ptr = new T[size + padT];
        rawPtrVec.push_back(ptr);
        T *alignedPtr = (T *) ((((uintptr_t) (ptr + padT)) >> b) << b);
        return alignedPtr;
    }

    CoarseTracker::CoarseTracker() {

        mState = NO_IMAGES_YET;
    }

    CoarseTracker::~CoarseTracker() {
        for (auto p : ptrToDelete) {
            delete[] p;
        }
        ptrToDelete.clear();
    }

    SE3d CoarseTracker::InsertStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp,
                                     const VecIMU &vimu) {
        // let make a histogram equalization
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        Mat imgLeftAfter, imgRightAfter;
        clahe->apply(imRectLeft, imgLeftAfter);
        clahe->apply(imRectRight, imgRightAfter);

        mpCurrentFrame = shared_ptr<Frame>(new Frame(imgLeftAfter, imgRightAfter, timestamp, mpCam, vimu));
        mvIMUSinceLastKF.insert(mvIMUSinceLastKF.end(), vimu.begin(), vimu.end());   // 追加imu数据

        // don't need to extract feature in each frame
        LOG(INFO) << "\n\n********* Tracking frame " << mpCurrentFrame->mnId << " **********" << endl;

        Track();
        LOG(INFO) << "Tracker returns. \n\n" << endl;
        return mpCurrentFrame->GetPose();
    }

    void CoarseTracker::Track() {

        int TrackInliersCnt = 0;

        if (mState == NO_IMAGES_YET) {

            // 尝试通过视觉双目构建初始地图
            // first we build the pyramid and compute the features
            ORBExtractor extractor(ORBExtractor::OPENCV_GFTT);
            extractor.Detect(mpCurrentFrame, true, false);    // extract the keypoint in left eye
            ORBMatcher matcher;
            matcher.ComputeStereoMatches(mpCurrentFrame, ORBMatcher::OPTIFLOW_CV);

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

    bool CoarseTracker::TrackLastFrame(bool usePoseInfo) {

        // Track the points in last frame and create new features in current
        UpdateLastFrame();

        mpCurrentFrame->makeGradient(mpCurrentFrame->mPyramidLeft, mpCurrentFrame->mGradxPyramidLeft,
                     mpCurrentFrame->mGradyPyramidLeft,
                     mpCurrentFrame->mAbsSquaredGradLeft);

        mpCurrentFrame->makeGradient(mpCurrentFrame->mPyramidRight, mpCurrentFrame->mGradxPyramidRight,
                     mpCurrentFrame->mGradyPyramidRight,
                     mpCurrentFrame->mAbsSquaredGradRight);
        cv::Mat SelectMapLeft, SelectMapRight;

        int numMapPoints = 0;
        SelectMapLeft.create(mpCam->mh[0], mpCam->mw[0], CV_32F);
        SelectMapRight.create(mpCam->mh[0], mpCam->mw[0], CV_32F);

        float *pSelectMapLeft = (float *) (SelectMapLeft.data);
        float *pSelectMapRight = (float *) (SelectMapRight.data);
        pPixelSelector->makeMaps(mpCurrentFrame, pSelectMapLeft, setting::desiredImmatureDensity);
        pPixelSelector->makeMaps(mpCurrentFrame, pSelectMapRight, setting::desiredImmatureDensity);

        vector<vector<shared_ptr<Feature>>> allKeypoints;
        allKeypoints.resize(setting::numPyramid);
        for (int yi = 0; yi < mpCam->mh[0]; ++yi) {
            for (int xi = 0; xi < mpCam->mw[0]; ++xi) {
                if (std::fabs(SelectMapLeft.at<float>(yi, xi)) > 1e-5) {
                    if (xi < setting::boarder || yi < setting::boarder ||
                        xi > setting::imageWidth - setting::boarder || yi > setting::imageHeight - setting::boarder)
                        continue;

                    // compute the grid location
                    vector<size_t> indecies = mpCurrentFrame->GetFeaturesInArea(xi, yi, 20);
                    if (indecies.empty()) {
                        shared_ptr<Feature> feat(new Feature);
                        feat->mPixel = Vector2f(xi, yi);
                        feat->mLevel = 0;
                        allKeypoints[0].push_back(feat);
                    }
                }
            }
        }

        for (size_t level = 0; level < setting::numPyramid; level++) {
            vector<shared_ptr<Feature>> &features = allKeypoints[level];
            if (features.empty())
                continue;
            for (auto f: features) {
                f->mPixel *= setting::scaleFactors[level];

            }
        }
        LOG(INFO) << "DSO tracked points: " << numMapPoints << ", last frame features: "
                  << mpLastFrame->mFeaturesLeft.size() << endl;

        mpCurrentFrame->makeCoarseDepthL0(mpCurrentFrame);
        TrackNewCoarse();

        OptimizeCurrentPose();

        return false;
    }

    bool CoarseTracker::TrackLocalMap(int &inliers) {

        LOG(INFO) << "Tracking local map" << endl;

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

        ORBMatcher matcher;
        int cntMatches = matcher.SearchByDirectProjection(mpCurrentFrame, mpsInView);
        LOG(INFO) << "Track local map matches: " << cntMatches << ", current features: "
                  << mpCurrentFrame->mFeaturesLeft.size() << endl;

        // Optimize Pose
        int optinliers;
        if (mState == OK)
            optinliers = OptimizeCurrentPoseWithIMU();
        else
            optinliers = OptimizeCurrentPoseWithoutIMU();

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

        LOG(INFO) << "Track Local map inliers: " << inliers << ", total features: "
                  << mpCurrentFrame->mFeaturesLeft.size() << endl;

        // Decide if the tracking was succesful
        if (inliers < setting::minTrackLocalMapInliers)
            return false;
        else
            return true;
    }

    bool CoarseTracker::TrackReferenceKF(bool usePoseInfomation) {
        // TODO implement it
        return false;
    }

    void CoarseTracker::InsertKeyFrame() {

        // detect new features in current frame
        if (mpLastKeyFrame != nullptr) {
            mpCurrentFrame->AssignFeaturesToGrid();

            ORBExtractor extractor(ORBExtractor::OPENCV_GFTT);
            LOG(INFO) << "Features before " << mpCurrentFrame->mFeaturesLeft.size() << endl;
            extractor.Detect(mpCurrentFrame, true, false);    // extract new keypoints in left eye
            LOG(INFO) << "Features after " << mpCurrentFrame->mFeaturesLeft.size() << endl;
            ORBMatcher matcher;
            matcher.ComputeStereoMatches(mpCurrentFrame, ORBMatcher::OPTIFLOW_CV);
        }

        // call tracker's keyframe function
        Tracker::InsertKeyFrame();
    }

    void CoarseTracker::Reset() {

        LOG(INFO) << "Tracker is reseted" << endl;
        mpCurrentFrame->SetPose(mpLastFrame->GetPose());
        LOG(INFO) << "Current pose = \n" << mpCurrentFrame->GetPose().matrix() << endl;
        mpBackEnd->Reset();

        // test if we can just recover from stereo
        ORBExtractor extractor(ORBExtractor::OPENCV_GFTT);
        extractor.Detect(mpCurrentFrame, true, false);    // extract the keypoint in left eye
        ORBMatcher matcher;
        matcher.ComputeStereoMatches(mpCurrentFrame, ORBMatcher::OPTIFLOW_CV);

        if (StereoInitialization() == false) {
            LOG(INFO) << "Stereo init failed." << endl;
            // 置状态为等待初始化状态
            mState = NOT_INITIALIZED;
            return;
        } else {
            LOG(INFO) << "Stereo init succeed." << endl;
            // set the current as a new kf and track it
            InsertKeyFrame();

            mpLastFrame = mpCurrentFrame;
            mpLastKeyFrame = mpCurrentFrame;
        }
        return;
    }

    bool CoarseTracker::TrackNewestCoarse(shared_ptr<Frame> newFrame,
                                          SE3d &lastToNew_out, int coarsestLvl,
                                          const Vector5d &minResForAbort) {

        assert(coarsestLvl < setting::numPyramid && coarsestLvl > 0);

        mLastResiduals.setConstant(NAN);
        mLastFlowIndicators.setConstant(1000);

        mpCurrentFrame = newFrame;

        // 每层金字塔的迭代次数
        int maxIterations[] = {10, 20, 50, 50, 50};
        float lambdaExtrapolationLimit = 0.001; // L-M lambda limitation

        // 把上一次track的结果作为初始值
        SE3d refToNew_current = lastToNew_out;

        bool haveRepeated = false;
        // 从最粗糙的金字塔开始，直到最精细的
        for (int lvl = coarsestLvl; lvl >= 0; lvl--) {

            Matrix6d H; // 6 DoF pose
            Vector6d b;

            float levelCutoffRepeat = 1;

            // 先算一遍残差，调整Huber阈值
            Vector6d resOld = CalcRes(lvl, refToNew_current,
                                      setting::coarseCutoffTH * levelCutoffRepeat);
            while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
                // 60%以上饱和，提高饱和的阈值，直到饱和数量减小
                levelCutoffRepeat *= 2;
                resOld = CalcRes(lvl, refToNew_current,
                                 setting::coarseCutoffTH * levelCutoffRepeat);

                if (!setting::debugout_runquiet) {
                    printf("INCREASING cutoff to %f (ratio is %f)!\n",
                           setting::coarseCutoffTH * levelCutoffRepeat, resOld[5]);
                }
            }

            // Compute H and b
            CalcGS(lvl, H, b, refToNew_current);

            float lambda = 0.01;

            // L-M iteration
            for (int iteration = 0; iteration < maxIterations[lvl]; iteration++) {
                Matrix6d Hl = H;
                for (int i = 0; i < 6; i++) {
                    Hl(i, i) *= (1 + lambda);
                }
                Vector6d inc = Hl.ldlt().solve(-b);


                // 原始代码有三种模式分别是固定AB，固定A、固定B，这里只使用固定AB

                inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());

                float extrapFac = 1;
                if (lambda < lambdaExtrapolationLimit) {
                    extrapFac =
                            sqrt(sqrt(lambdaExtrapolationLimit / lambda)); // lambda 不能太小
                }

                inc *= extrapFac;

                Vector6d incScaled = inc;
                incScaled.segment<3>(0) *= setting::SCALE_XI_ROT;
                incScaled.segment<3>(3) *= setting::SCALE_XI_TRANS;

                if (!std::isfinite(incScaled.sum())) {
                    incScaled.setZero();
                }

                // left multiply the pose and add to a,b
                SE3d refToNew_new =
                        SE3d::exp((Vector6d) (incScaled.head<6>())) * refToNew_current;


                // calculate new residual after this update step
                Vector6d resNew = CalcRes(lvl, refToNew_new, setting::coarseCutoffTH * levelCutoffRepeat);

                // decide whether to accept this step
                // res[0]/res[1] is the average energy
                bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);


                if (accept) {
                    // 减小lambda
                    CalcGS(lvl, H, b, refToNew_new);
                    resOld = resNew;

                    refToNew_current = refToNew_new;
                    lambda *= 0.5;
                } else {
                    // 加大lambda,增强正定性
                    lambda *= 4;
                    if (lambda < lambdaExtrapolationLimit) {
                        lambda = lambdaExtrapolationLimit;
                    }
                }

                // 终止条件：增量较小
                if (!(inc.norm() > 1e-3)) {
                    if (setting::debugPrint) {
                        printf("inc too small, break!\n");
                    }
                    break;
                }
            } // end of L-M iteration

            // set last residual for that level, as well as flow indicators.
            mLastResiduals[lvl] = sqrtf((float) (resOld[0] / resOld[1]));
            mLastFlowIndicators = resOld.segment<3>(2);
            if (mLastResiduals[lvl] > 1.5 * minResForAbort[lvl]) {
                return false;
            }

            if (levelCutoffRepeat > 1 && !haveRepeated) {
                lvl++;
                haveRepeated = true;
                printf("REPEAT LEVEL!\n");
            }
        } // end of for: pyramid level

        // set!
        lastToNew_out = refToNew_current;

        // always return true
        return true;
    }





    void CoarseTracker::CalcGS(int lvl, Matrix6d &H_out, ygz::Vector6d &b_out, const Sophus::SE3d &refToNew) {
        acc.initialize();

        float fxl = mpCam->mfx[lvl];
        float fyl = mpCam->mfy[lvl];

        int n = res_pt_buf.size();
        for (int i = 0; i < n; i++) {
            float dx = res_pt_buf[i].dx * fxl;
            float dy = res_pt_buf[i].dy * fyl;
            float u = res_pt_buf[i].u;
            float v = res_pt_buf[i].v;
            float id = res_pt_buf[i].idepth;

            acc.updateSingle(id * dx, id * dy, -id * (u * dx + v * dy),
                             -(u * v * dx + dy * (1 + v * v)),  // 这里似乎少了 id^2 ？
                             u * v * dy + dx * (1 + u * u), // 这里似乎少了 id^2 ？
                             u * dy - v * dx, // 这里似乎少了 id ？
                             -1
            );
        }

        acc.finish();
        H_out = acc.H.topLeftCorner<6, 6>().cast<double>() * (1.0f / n);
        b_out = acc.H.topRightCorner<6, 1>().cast<double>() * (1.0f / n);
    }

    Vector6d CoarseTracker::CalcRes(int lvl, const SE3d &refToNew, float cutoffTH) {

        float E = 0;
        int numTermsInE = 0;
        int numTermsInWarped = 0;
        int numSaturated = 0;

        int wl = mpCam->mw[lvl];
        int hl = mpCam->mh[lvl];

        float fxl = mpCam->mfx[lvl];
        float fyl = mpCam->mfy[lvl];
        float cxl = mpCam->mcx[lvl];
        float cyl = mpCam->mcy[lvl];

        Matrix3f RKi =
                (refToNew.rotationMatrix().cast<float>() * mpCam->mKi[lvl]); // R_new_ref * K^-1
        Vector3f t = (refToNew.translation()).cast<float>();      // t_new_ref

        float sumSquaredShiftT = 0;
        float sumSquaredShiftRT = 0;
        float sumSquaredShiftNum = 0;

        float maxEnergy =
                2 * setting::huberTH * cutoffTH -
                setting::huberTH *
                setting::huberTH; // energy for r=setting_coarseCutoffTH.

        int nl = mpCurrentFrame->pc_buffer[lvl].size();


        for (int i = 0; i < nl; i++) {
            float id = mpCurrentFrame->pc_buffer[lvl][i].idepth;
            float x = mpCurrentFrame->pc_buffer[lvl][i].u;
            float y = mpCurrentFrame->pc_buffer[lvl][i].v;
            // (x,y,1) is the point in reference

            Vector3f pt = RKi * Vector3f(x, y, 1) + t * id;
            // u,v 是在newframe中的投影
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            float new_idepth = id / pt[2];

            if (lvl == 0 && i % 32 == 0) {
                // translation only (positive)
                Vector3f ptT = mpCam->mKi[lvl] * Vector3f(x, y, 1) + t * id;
                float uT = ptT[0] / ptT[2];
                float vT = ptT[1] / ptT[2];
                float KuT = fxl * uT + cxl;
                float KvT = fyl * vT + cyl;

                // translation only (negative)
                Vector3f ptT2 = mpCam->mKi[lvl] * Vector3f(x, y, 1) - t * id;
                float uT2 = ptT2[0] / ptT2[2];
                float vT2 = ptT2[1] / ptT2[2];
                float KuT2 = fxl * uT2 + cxl;
                float KvT2 = fyl * vT2 + cyl;

                // translation and rotation (negative)
                Vector3f pt3 = RKi * Vector3f(x, y, 1) - t * id;
                float u3 = pt3[0] / pt3[2];
                float v3 = pt3[1] / pt3[2];
                float Ku3 = fxl * u3 + cxl;
                float Kv3 = fyl * v3 + cyl;

                // translation and rotation (positive)
                // already have it.

                sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
                sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                sumSquaredShiftNum += 2;
            }

            if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0)) {
                continue;
            }

            float refColor = mpCurrentFrame->pc_buffer[lvl][i].color;
            Vector3f hitColor;
            hitColor[0] = mpCurrentFrame->GetInterpolatedImgElement33(mpCurrentFrame->mPyramidLeft[lvl], Ku, Kv);
            hitColor[1] = mpCurrentFrame->GetInterpolatedGradElement33(mpCurrentFrame->mGradxPyramidLeft[lvl], Ku, Kv);
            hitColor[2] = mpCurrentFrame->GetInterpolatedGradElement33(mpCurrentFrame->mGradyPyramidLeft[lvl], Ku, Kv);
            if (!std::isfinite((float) hitColor[0])) {
                continue;
            }
            float residual = hitColor[0] - refColor; // 残差：参数－当前
            float hw = fabs(residual) < setting::huberTH
                       ? 1
                       : setting::huberTH / fabs(residual); // huber residual

            // 超出阈值，取最大值
            if (fabs(residual) > cutoffTH) {

                E += maxEnergy;
                numTermsInE++;
                numSaturated++;
            } else {
                // 小于最大值, 有效的残差

                E += hw * residual * residual * (2 - hw);
                numTermsInE++;

                res_point_buffer pt;
                pt.idepth = new_idepth;
                pt.u = u;
                pt.v = v;
                pt.dx = hitColor[1];
                pt.dy = hitColor[2];
                pt.residual = residual;
                pt.refColor = mpCurrentFrame->pc_buffer[lvl][i].color;
                pt.weight = hw;

                res_pt_buf.push_back(pt);

            }
        }

        // 凑成4的整数倍，便于加速
        // 往后头添加一些0
        while (numTermsInWarped % 4 != 0) {

            res_point_buffer pt;
            pt.idepth = 0;
            pt.u = 0;
            pt.v = 0;
            pt.dx = 0;
            pt.dy = 0;
            pt.residual = 0;
            pt.refColor = 0;
            pt.weight = 0;

            res_pt_buf.push_back(pt);
        }
//        buf_warped_n = numTermsInWarped;

        Vector6d rs;
        rs[0] = E;
        rs[1] = numTermsInE;
        rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1); // 避免0分母
        rs[3] = 0;
        rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1); // 避免0分母
        rs[5] = numSaturated / (float) numTermsInE;              // 饱和点百分比

        return rs;
    }

    bool CoarseTracker::TrackNewCoarse() {

        assert(mpLastFrame != nullptr);
        // set pose initialization.
        Vector5d achievedRes = Vector5d::Constant(NAN);


        mpCurrentFrame->mTWC = SE3d(Matrix4d::Identity());
        bool bOK = false;
        bOK = TrackNewestCoarse(mpCurrentFrame, mpCurrentFrame->mTWC, setting::numPyramid - 1, achievedRes);


        if (bOK &&
            std::isfinite((float) mLastResiduals[0]) &&
            !(mLastResiduals[0] >= achievedRes[0])) {
            for (int i = 0; i < 5; i++) {
                if (!std::isfinite((float) achievedRes[i]) ||
                    achievedRes[i] > mLastResiduals[i])
                    achievedRes[i] = mLastResiduals[i];
            }
        }


        Vector4d(achievedRes[0], mLastFlowIndicators[0], mLastFlowIndicators[1], mLastFlowIndicators[2]);

        if (!std::isfinite((double) achievedRes[0]) || !std::isfinite((double) mLastFlowIndicators[0]) ||
            !std::isfinite((double) mLastFlowIndicators[1]) || !std::isfinite((double) mLastFlowIndicators[2])) {
            printf("Initial Tracking failed: LOST!\n");

            return false;
        } else
            return true;

    }


    /** 选点并构造sparePoint
     * 1. 针对第一层 使用makemaps
     * 2. 针对其它层使用makepixelstatus
     * 3. 针对选出的点，进行trace right
     * 4. 针对trace right匹配好的点，进行保存到spare point容器中
     * @ 当前帧
     * */
    void CoarseTracker::SelectPoints(shared_ptr<Frame> newFrame) {

        shared_ptr<Frame> firstFrame = newFrame;

        PixelSelector sel(mpCam->mw[0], mpCam->mh[0]);

        Mat float_map, bool_map;
        float_map.create(mpCam->mh[0], mpCam->mw[0], CV_32F);
        bool_map.create(mpCam->mh[0], mpCam->mw[0], cv::DataType<bool>::type);

        float *floatStatusMap = (float *) float_map.data;
        bool *boolStatusMap = (bool *) bool_map.data;

        //金字塔每一层采集像素的密度
        float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
        //对于金字塔的每一层
        for (int lvl = 0; lvl < setting::numPyramid; lvl++) {
            sel.currentPotential = 3;
            int npts;
            if (lvl == 0) {
                npts = sel.makeMaps(firstFrame, floatStatusMap, densities[lvl] * mpCam->mw[0] * mpCam->mh[0], 1, false, 2);
                newFrame->FloatMapToPoint(float_map, newFrame->vPointsLeft);
            }
            else
                npts = sel.makePixelStatus(firstFrame->mGradxPyramidLeft[lvl], firstFrame->mGradyPyramidLeft[lvl],
                                           boolStatusMap, mpCam->mw[lvl], mpCam->mh[lvl], densities[lvl] * mpCam->mw[0] * mpCam->mh[0]);
        }
    }


} // namespace ygz
