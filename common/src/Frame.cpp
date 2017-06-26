#include "ygz/Frame.h"
#include "ygz/Settings.h"
#include "ygz/MapPoint.h"
#include "ygz/Feature.h"
#include "ygz/Memory.h"
#include "ygz/DSOCoarseTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ygz/SparePoint.h"
#include <opencv2/highgui/highgui.hpp>
#include "lsd/lsd.h"

using namespace cv;

namespace ygz {

    // static variables
    long unsigned int Frame::nNextId = 0;
    long unsigned int Frame::nNextKFId = 0;
    shared_ptr<ORBVocabulary> Frame::pORBvocabulary = nullptr;

    // copy constructor
    Frame::Frame(const Frame &frame)
            :
            mTimeStamp(frame.mTimeStamp), mpCam(frame.mpCam),
            mFeaturesLeft(frame.mFeaturesLeft), mFeaturesRight(frame.mFeaturesRight), mnId(frame.mnId),
            mpReferenceKF(frame.mpReferenceKF), mImLeft(frame.mImLeft), mImRight(frame.mImRight),
            mMargCovInv(frame.mMargCovInv) {
        SetPose(SE3d(frame.mRwb, frame.mTwb));
    }

    // normal constructor
    Frame::Frame(const cv::Mat &left, const cv::Mat &right, const double &timestamp, shared_ptr<CameraParam> cam,
                 const VecIMU &IMUSinceLastFrame)
            : mTimeStamp(timestamp), mImLeft(left.clone()), mImRight(right.clone()), mpCam(cam),
              mvIMUDataSinceLastFrame(IMUSinceLastFrame) {
        SetPose(SE3d());
        mnId = nNextId++;
        ComputeImagePyramid();
    }

    Frame::~Frame() {
        LOG(INFO) << "Deconstructing frame " << mnId << endl;
        for (shared_ptr<Feature> &feature : mFeaturesLeft) {
            feature = nullptr;
        }
        for (shared_ptr<Feature> &feature : mFeaturesRight)
            feature = nullptr;
        mFeaturesLeft.clear();
        mFeaturesRight.clear();
        LOG(INFO) << "Done." << endl;
    }

    void Frame::SetDelete(shared_ptr<Frame> pNextKF) {
        // TODO: delete operation in Memory
        // TODO: after change to shared_ptr, re-think about this function

        //1. if feat->mpPoint is nullptr, then do nothing
        //2. if feat->mpPoint is not null
        //2.1 if feat->mpPoint is bad, delete this MapPoint (call delete and set nullptr)
        //2.2 if feat->mpPoint is not bad
        //2.2.1 if feat->mpPoint observations == 1 (only mpRefKF == frameToDel), then delete this MapPoint
        //2.2.2 if feat->mpPoint observations > 1, then erase frameToDel in mObservations, and change mpRefKF
        //3. !! remember to erase MapPoint in mpPoints !!
        // no need to setMarg for the moment.
        // at last, delete frameToDel (clear mObservation/mFeaturesLeft/mFeaturesRight)

        LOG(INFO) << "set delete frame " << mnId << " kf id = " << mnKFId << endl;

        // 将最老的关键帧移除出窗口
        // 删掉此帧以及从此帧出发的地图点
        LOG(INFO) << "Cleaning features" << endl;
        for (shared_ptr<Feature> feat: mFeaturesLeft) {
            shared_ptr<MapPoint> mp = feat->mpPoint;
            if (!mp)
                continue;
            else
                mp = nullptr;

            weak_ptr<Frame> thisptr(shared_ptr<Frame>(this));
            if (!mp->mObservations.count(thisptr))
                LOG(ERROR) << "!mp->mObservations.count(this) in Frame::SetDelete" << endl;

            mp->mObservations.erase(thisptr);
            feat->mpPoint = nullptr;

            bool toDelThisMP = false;
            if (mp->isBad()) {
                // delete this MapPoint
                toDelThisMP = true;
            } else {
                if (mp->Observations() < 1) {
                    // delete this MapPoint
                    toDelThisMP = true;
                } else {
                    // change mpRefKF
                    // compute new invDepth?
                    mp->mpRefKF = mp->mObservations.begin()->first;
                    if (mp->mpRefKF.expired())
                        LOG(ERROR) << "new reference KF expired?" << endl;
                    shared_ptr<Feature> newfeat = mp->mpRefKF.lock()->mFeaturesLeft[mp->mObservations[mp->mpRefKF]];
                    if (newfeat->mfInvDepth < 0)
                        LOG(INFO) << "new reference KF, inv depth: " << newfeat->mfInvDepth << endl;
                }
            }

            if (toDelThisMP) {
                // erase MapPoint in mpPoints
                // delete MapPoint
                // set feat->mpPoint = nullptr
                // TODO: marginalize this point (if use marginalization)
                mp->SetBadFlag();
                //mpPoints.erase(mp);
                //memory::DeleteMapPoint(mp);
            }

        }

        // change reference KF
        /*
        if (pNextKF) {
            assert(this == pNextKF->mpReferenceKF.get());
            if (this != pNextKF->mpReferenceKF.lock())
                LOG(ERROR) << "this != pNextKF->mpReferenceKF" << endl;
            // change reference KeyFrame of next KF
            pNextKF->mpReferenceKF = mpReferenceKF;
            //mpReferenceKF = nullptr;
        }
         */
    }

    void Frame::SetPose(const SE3d &Twb) {
        mRwb = Twb.rotationMatrix();
        mTwb = Twb.translation();

        SE3d TWC = Twb * setting::TBC;
        SE3d TCW = TWC.inverse();
        mRcw = TCW.rotationMatrix();
        mtcw = TCW.translation();
        mOw = TWC.translation();
        mRwc = TWC.rotationMatrix();
    }

    bool Frame::SetThisAsKeyFrame() {
        if (mbIsKeyFrame == true)
            return true;

        // 置关键帧
        mbIsKeyFrame = true;
        mnKFId = nNextKFId++;
        // TODO 对KeyFrame需要的东西加锁

        return true;
    }

    void Frame::ComputeIMUPreIntSinceLastFrame(const shared_ptr<Frame> pLastF, IMUPreIntegration &IMUPreInt) {
        // Reset pre-integrator first
        IMUPreInt.reset();

        const VecIMU &vIMUSInceLastFrame = mvIMUDataSinceLastFrame;

        Vector3d bg = pLastF->BiasG();
        Vector3d ba = pLastF->BiasA();

        // remember to consider the gap between the last KF and the first IMU
        {
            const IMUData &imu = vIMUSInceLastFrame.front();
            double dt = std::max(0., imu.mfTimeStamp - pLastF->mTimeStamp);
            IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }
        // integrate each imu
        for (size_t i = 0; i < vIMUSInceLastFrame.size(); i++) {
            const IMUData &imu = vIMUSInceLastFrame[i];
            double nextt;
            if (i == vIMUSInceLastFrame.size() - 1)
                nextt = mTimeStamp;         // last IMU, next is this KeyFrame
            else
                nextt = vIMUSInceLastFrame[i + 1].mfTimeStamp;  // regular condition, next is imu data

            // delta time
            double dt = std::max(0., nextt - imu.mfTimeStamp);
            // update pre-integrator
            IMUPreInt.update(imu.mfGyro - bg, imu.mfAcce - ba, dt);
        }
    }

    void Frame::UpdatePoseFromPreintegration(const IMUPreIntegration &imupreint, const Vector3d &gw) {

        Matrix3d dR = imupreint.getDeltaR();
        Vector3d dP = imupreint.getDeltaP();
        Vector3d dV = imupreint.getDeltaV();
        double dt = imupreint.getDeltaTime();

        Vector3d Pwbpre = mTwb;     // 平移
        Matrix3d Rwbpre = mRwb.matrix();
        Vector3d Vwbpre = Speed();

        Matrix3d Rwb = Rwbpre * dR;
        Vector3d Pwb = Pwbpre + Vwbpre * dt + 0.5 * gw * dt * dt + Rwbpre * dP;
        Vector3d Vwb = Vwbpre + gw * dt + Rwbpre * dV;

        // Here assume that the pre-integration is re-computed after bias updated, so the bias term is ignored
        mTwb = Pwb;
        mSpeedAndBias.segment<3>(0) = Vwb;
        mRwb = SO3d(Rwb);

        // 普通帧的bias横竖不知道，不用管

        // 更新相机的坐标系
        SE3d TWC = SE3d(mRwb, mTwb) * setting::TBC;
        SE3d TCW = TWC.inverse();
        mRcw = TCW.rotationMatrix();
        mtcw = TCW.translation();
        mOw = TWC.translation();
        mRwc = TWC.rotationMatrix();
    }

    vector<size_t> Frame::GetFeaturesInArea(
            const float &x, const float &y, const float &r, const int minLevel,
            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(mFeaturesLeft.size());

        const int nMinCellX = max(0, (int) floor((x - r) * setting::GridElementWidthInv));
        if (nMinCellX >= setting::FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) setting::FRAME_GRID_COLS - 1,
                                  (int) ceil((x + r) * setting::GridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - r) * setting::GridElementHeightInv));
        if (nMinCellY >= setting::FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) setting::FRAME_GRID_ROWS - 1,
                                  (int) ceil((y + r) * setting::GridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {

                    const shared_ptr<Feature> feature = mFeaturesLeft[vCell[j]];
                    if (bCheckLevels) {
                        if (feature->mLevel < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (feature->mLevel > maxLevel)
                                continue;
                    }

                    const float distx = feature->mPixel[0] - x;
                    const float disty = feature->mPixel[1] - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    void Frame::AssignFeaturesToGrid() {
        if (mFeaturesLeft.empty())
            return;

        int nReserve = 0.5f * mFeaturesLeft.size() / (setting::FRAME_GRID_COLS * setting::FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < setting::FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < setting::FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (size_t i = 0; i < mFeaturesLeft.size(); i++) {
            shared_ptr<Feature> f = mFeaturesLeft[i];
            int nGridPosX, nGridPosY;
            if (PosInGrid(f, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    bool Frame::isInFrustum(shared_ptr<MapPoint> pMP, float viewingCosLimit, int boarder) {

        // 3D in absolute coordinates
        Vector3d P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Vector3d Pc = mRcw * P + mtcw;
        const float &PcX = Pc[0];
        const float &PcY = Pc[1];
        const float &PcZ = Pc[2];

        // Check valid depth
        if (PcZ < setting::minPointDis || PcZ > setting::maxPointDis) {
            return false;
        }

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = mpCam->fx * PcX * invz + mpCam->cx;
        const float v = mpCam->fy * PcY * invz + mpCam->cy;

        if (u < boarder || u > (setting::imageWidth - boarder))
            return false;
        if (v < boarder || v > (setting::imageHeight - boarder))
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const Vector3d PO = P - mOw;
        const float dist = PO.norm();

        // Check viewing angle
        Vector3d Pn = pMP->GetNormal();
        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit) {
            return false;
        }

        // Data used by the tracking
        pMP->mTrackProjX = u;
        pMP->mTrackProjY = v;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    void Frame::ComputeBoW() {
        if (pORBvocabulary && mBowVec.empty() && !mFeaturesLeft.empty()) {
            vector<cv::Mat> vCurrentDesc = this->GetAllDescriptor();
            pORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    vector<Mat> Frame::GetAllDescriptor() const {
        vector<Mat> ret;
        ret.reserve(mFeaturesLeft.size());
        for (size_t i = 0; i < mFeaturesLeft.size(); i++) {
            ret.push_back(cv::Mat(1, 32, CV_8UC1, mFeaturesLeft[i]->mDesc));
        }
        return ret;
    }

    Vector3d Frame::UnprojectStereo(const int &i) {
        const float z = 1.0 / mFeaturesLeft[i]->mfInvDepth;
        if (z > 0) {
            const float u = mFeaturesLeft[i]->mPixel[0];
            const float v = mFeaturesLeft[i]->mPixel[1];
            const float x = (u - mpCam->cx) * z * mpCam->fxinv;
            const float y = (v - mpCam->cy) * z * mpCam->fyinv;
            Vector3d x3Dc(x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return Vector3d(0, 0, 0);
    }

    void Frame::ComputeImagePyramid() {

        mPyramidLeft.resize(setting::numPyramid);
        mPyramidRight.resize(setting::numPyramid);

        for (size_t level = 0; level < setting::numPyramid; ++level) {
            float scale = setting::invScaleFactors[level];
            Size sz(cvRound((float) mImLeft.cols * scale), cvRound((float) mImLeft.rows * scale));
            Size wholeSize(sz.width + setting::EDGE_THRESHOLD * 2, sz.height + setting::EDGE_THRESHOLD * 2);

            Mat tempL(wholeSize, mImLeft.type()), masktempL;
            Mat tempR(wholeSize, mImRight.type()), masktempR;

            mPyramidLeft[level] = tempL(Rect(setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, sz.width, sz.height));

            mPyramidRight[level] = tempR(Rect(setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if (level != 0) {
                resize(mPyramidLeft[level - 1], mPyramidLeft[level], sz, 0, 0, INTER_LINEAR);

                resize(mPyramidRight[level - 1], mPyramidRight[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mPyramidLeft[level], tempL, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);

                copyMakeBorder(mPyramidRight[level], tempR, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);
            } else {
                copyMakeBorder(mImLeft, tempL, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, BORDER_REFLECT_101);
                copyMakeBorder(mImRight, tempR, setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD,
                               setting::EDGE_THRESHOLD, setting::EDGE_THRESHOLD, BORDER_REFLECT_101);
            }

        }
    }

    bool Frame::PosInGrid(const shared_ptr<Feature> feature, int &posX, int &posY) {
        posX = round(feature->mPixel[0] * setting::GridElementWidthInv);
        posY = round(feature->mPixel[1] * setting::GridElementHeightInv);
        if (posX < 0 || posX >= setting::FRAME_GRID_COLS
            || posY < 0 || posY >= setting::FRAME_GRID_ROWS)
            return false;
        return true;
    }

    double Frame::ComputeSceneMedianDepth(const int &q) {

        vector<double> vDepth;
        for (auto feat: mFeaturesLeft) {
            if (feat->mfInvDepth > 0)
                vDepth.push_back(1.0 / feat->mfInvDepth);
        }

        if (vDepth.empty())
            return 0;

        sort(vDepth.begin(), vDepth.end());
        return vDepth[(vDepth.size() - 1) / q];
    }

    void Frame::EraseMapPointMatch(shared_ptr<MapPoint> mp) {
        /*
        int idx = mp->mObservations[this];
        assert(idx >= 0 && idx < mFeaturesLeft.size());
        shared_ptr<Feature> feat = mFeaturesLeft[idx];
        if (idx >= 0)
            feat->mpPoint = nullptr;
        */
    }

    void Frame::EraseMapPointMatch(const size_t &idx) {
        //TODO: add mutex lock
        auto feat = mFeaturesLeft[idx];
        assert(idx >= 0 && idx < mFeaturesLeft.size());

        if (idx >= 0)
            feat->mpPoint = nullptr;
    }

    int Frame::TrackedMapPoints(const int &minObs) {
        int nPoints = 0;
        const bool bCheckObs = minObs > 0;
        int N = mFeaturesLeft.size();
        for (int i = 0; i < N; i++) {
            shared_ptr<MapPoint> pMP = mFeaturesLeft[i]->mpPoint;
            if (pMP) {
                if (!pMP->isBad()) {
                    if (bCheckObs) {
                        if (pMP->Observations() >= minObs)
                            nPoints++;
                    } else
                        nPoints++;
                }
            }
        }

        return nPoints;
    }

    void Frame::makeGradient(const vector<Mat> &mPyramid, vector<Mat> &mGradxPyramid, vector<Mat> &mGradyPyramid,
                             vector<Mat> &mAbsSquaredGrad) {

        mGradxPyramid.resize(setting::numPyramid);
        mGradyPyramid.resize(setting::numPyramid);
        mAbsSquaredGrad.resize(setting::numPyramid);

        int w = mpCam->mw[0];
        int h = mpCam->mh[0];

        for (int lvl = 0; lvl < setting::numPyramid; lvl++) {
            int wl = mpCam->mw[lvl], hl = mpCam->mh[lvl];
            mGradxPyramid[lvl].create(hl, wl, CV_32F);
            mGradyPyramid[lvl].create(hl, wl, CV_32F);
            mAbsSquaredGrad[lvl].create(hl, wl, CV_32F);
            unsigned char *pPyramidLeft = mPyramid[lvl].data;
            float *pGradxPyramid = (float *) mGradxPyramid[lvl].data;
            float *pGradyPyramid = (float *) mGradyPyramid[lvl].data;
            float *pAbsSquaredGrad = (float *) mAbsSquaredGrad[lvl].data;

            for (int idx = wl; idx < wl * (hl - 1); idx++) {
                float dx = 0.5f * (pPyramidLeft[idx + 1] - pPyramidLeft[idx - 1]);
                float dy = 0.5f * (pPyramidLeft[idx + wl] - pPyramidLeft[idx - wl]);

                if (!std::isfinite(dx))
                    dx = 0;
                if (!std::isfinite(dy))
                    dy = 0;

                pGradxPyramid[idx] = dx;
                pGradyPyramid[idx] = dy;

                pAbsSquaredGrad[idx] = sqrtf(dx * dx + dy * dy);
            }
        }
    }


    double Frame::GetInterpolatedImgElement33(const Mat &img, const float x, const float y) {
        assert(img.channels() == 1);
        int istep = img.step;
        int ix = (int) x;
        int iy = (int) y;
        float dx = x - ix;
        float dy = y - iy;
        float dxdy = dx * dy;
        const unsigned char *bp = img.data + ix + iy * istep;

        return dxdy * *(bp + 1 + istep) +
               (dy - dxdy) * *(bp + istep) +
               (dx - dxdy) * *(bp + 1) +
               (1 - dx - dy + dxdy) * *(bp);
    }

    Eigen::Vector3f Frame::getInterpolatedElement33BiLin(const float x, const float y) {
        int ix = (int) x;
        int iy = (int) y;
        int width = mImLeft.cols;
        const float *bp = (float *) (mImLeft.data + ix + iy * mImLeft.step);

        float tl = (*(bp));
        float tr = (*(bp + 1));
        float bl = (*(bp + width));
        float br = (*(bp + width + 1));

        float dx = x - ix;
        float dy = y - iy;
        float topInt = dx * tr + (1 - dx) * tl;
        float botInt = dx * br + (1 - dx) * bl;
        float leftInt = dy * bl + (1 - dy) * tl;
        float rightInt = dy * br + (1 - dy) * tr;

        return Eigen::Vector3f(dx * rightInt + (1 - dx) * leftInt, rightInt - leftInt,
                               botInt - topInt);
    }

    float Frame::GetInterpolatedGradElement33(const Mat &img, const float x, const float y) {
        assert(img.channels() == 1);
        int istep = img.step;
        int ix = (int) x;
        int iy = (int) y;
        float dx = x - ix;
        float dy = y - iy;
        float dxdy = dx * dy;
        const float *bp = (const float *) (img.data + ix + iy * istep);

        return dxdy * *(bp + 1 + istep) +
               (dy - dxdy) * *(bp + istep) +
               (dx - dxdy) * *(bp + 1) +
               (1 - dx - dy + dxdy) * *(bp);


    }


    void Frame::FloatMapToPoint(cv::Mat floatMap, vector<shared_ptr<SparePoint>> &vecSparePoints) {
        for (int i = 0; i < floatMap.rows; ++i) {
            for (int j = 0; j < floatMap.cols; ++j) {
                if (i < setting::boarder || j < setting::boarder || i > floatMap.rows - setting::boarder ||
                    j > floatMap.cols - setting::boarder)
                    continue;
                float type = floatMap.at<float>(i, j);

                if (type < 1e-5)
                    continue;

                shared_ptr<SparePoint> pt(new SparePoint(j, i, type, mpCam));
                pt->u_r = pt->u;
                pt->v_r = pt->v;
                pt->idepth_min_r = 0;
                pt->idepth_max_r = NAN;
                PointStatus stat = pt->TraceRight(shared_ptr<Frame>(this), mpCam->K, mBaseLine);

                if (stat == PointStatus::IPS_GOOD) {
                    vecSparePoints.push_back(pt);
                }
            }
        }
    }

    void Frame::makeIDepthWeightL() {
        //对于金字塔的每一层
        for (int lvl = 1; lvl < setting::numPyramid; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = mpCam->mw[lvl], hl = mpCam->mh[lvl], wlm1 = mpCam->mw[lvlm1];

            float *idepth_l = (float *) (mIDepthLeft[lvl].data);
            float *weightSums_l = (float *) (mWeightSumsLeft[lvl].data);

            float *idepth_lm = (float *) (mIDepthLeft[lvlm1].data);
            float *weightSums_lm = (float *) (mWeightSumsLeft[lvlm1].data);
            //得出每个点的深度，以及权重
            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++) {
                    int bidx = 2 * x + 2 * y * wlm1;
                    idepth_l[x + y * wl] = idepth_lm[bidx] + idepth_lm[bidx + 1] +
                                           idepth_lm[bidx + wlm1] +
                                           idepth_lm[bidx + wlm1 + 1];

                    weightSums_l[x + y * wl] =
                            weightSums_lm[bidx] + weightSums_lm[bidx + 1] +
                            weightSums_lm[bidx + wlm1] + weightSums_lm[bidx + wlm1 + 1];
                }
        }
    }

// dilate idepth by 1 (2 on lower levels).
    void Frame::dialteIDepth0To2() {

        for (int lvl = 0; lvl < 2; lvl++) {
            int numIts = 1;
            if (setting::dilateDoubleCoarse)
                numIts = 2;

            for (int it = 0; it < numIts; it++) {
                int wh = mpCam->mw[lvl] * mpCam->mh[lvl] - mpCam->mw[lvl];
                int wl = mpCam->mw[lvl];
                float *weightSumsl = (float *) (mWeightSumsLeft[lvl].data);
                float *weightSumsl_bak = (float *) (mWeightSumsBakLeft[lvl].data);
                memcpy(weightSumsl_bak, weightSumsl, mpCam->mw[lvl] * mpCam->mh[lvl] * sizeof(float));
                float *idepthl = (float *) (mIDepthLeft[lvl].data); // dotnt need to make a temp copy of depth, since I only
                // read values with weightSumsl>0, and write ones with weightSumsl<=0.
                for (int i = mpCam->mw[lvl]; i < wh; i++) {
                    if (weightSumsl_bak[i] <= 0) {
                        float sum = 0, num = 0, numn = 0;
                        if (weightSumsl_bak[i + 1 + wl] > 0) {
                            sum += idepthl[i + 1 + wl];
                            num += weightSumsl_bak[i + 1 + wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i - 1 - wl] > 0) {
                            sum += idepthl[i - 1 - wl];
                            num += weightSumsl_bak[i - 1 - wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i + wl - 1] > 0) {
                            sum += idepthl[i + wl - 1];
                            num += weightSumsl_bak[i + wl - 1];
                            numn++;
                        }
                        if (weightSumsl_bak[i - wl + 1] > 0) {
                            sum += idepthl[i - wl + 1];
                            num += weightSumsl_bak[i - wl + 1];
                            numn++;
                        }
                        if (numn > 0) {
                            idepthl[i] = sum / numn;
                            weightSumsl[i] = num / numn;
                        }
                    }
                }
            }
        }
    }

    // dilate idepth by 1 (2 on lower levels).
    void Frame::dialteIDepth2ToTop() {
        for (int lvl = 2; lvl < setting::numPyramid; lvl++) {
            int wh = mpCam->mw[lvl] * mpCam->mh[lvl] - mpCam->mw[lvl];
            int wl = mpCam->mw[lvl];
            float *weightSumsl = (float *) (mWeightSumsLeft[lvl].data);
            float *weightSumsl_bak = (float *) (mWeightSumsBakLeft[lvl].data);
            memcpy(weightSumsl_bak, weightSumsl, mpCam->mw[lvl] * mpCam->mh[lvl] * sizeof(float));
            float *idepthl = (float *) (mIDepthLeft[lvl].data); // dotnt need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = mpCam->mw[lvl]; i < wh; i++) {
                if (weightSumsl_bak[i] <= 0) {
                    float sum = 0, num = 0, numn = 0;
                    if (weightSumsl_bak[i + 1] > 0) {
                        sum += idepthl[i + 1];
                        num += weightSumsl_bak[i + 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i - 1] > 0) {
                        sum += idepthl[i - 1];
                        num += weightSumsl_bak[i - 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i + wl] > 0) {
                        sum += idepthl[i + wl];
                        num += weightSumsl_bak[i + wl];
                        numn++;
                    }
                    if (weightSumsl_bak[i - wl] > 0) {
                        sum += idepthl[i - wl];
                        num += weightSumsl_bak[i - wl];
                        numn++;
                    }
                    if (numn > 0) {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }
    }

    // normalize idepths and weights.
    void Frame::NormalizeDepth(std::vector<Mat> PyramidImg) {

        pc_buffer.resize(setting::numPyramid);
        for (int lvl = 0; lvl < setting::numPyramid; lvl++) {
            assert(PyramidImg[lvl].channels() == 1);
            float *weightSumsl = (float *) (mWeightSumsLeft[lvl].data);
            float *idepthl = (float *) (mIDepthLeft[lvl].data);
            //Eigen::Vector3f *dIRefl = mLastRef->dIp[lvl];
            //bool *overRefl = mLastRef->overexposedMapp[lvl];

            int wl = mpCam->mw[lvl], hl = mpCam->mh[lvl];

//            int lpc_n = 0;
//            float *lpc_u = pc_u[lvl];
//            float *lpc_v = pc_v[lvl];
//            float *lpc_idepth = pc_idepth[lvl];
//            float *lpc_color = pc_color[lvl];

            for (int y = 2; y < hl - 2; y++)
                for (int x = 2; x < wl - 2; x++) {
                    int i = x + y * wl;

                    if (weightSumsl[i] > 0) {
                        point_buffer ptbuf;
                        idepthl[i] /= weightSumsl[i];
                        ptbuf.u = x;
                        ptbuf.v = y;
                        ptbuf.idepth = idepthl[i];
                        ptbuf.color = PyramidImg[lvl].at < unsigned
                        char > (y, x);
                        pc_buffer[lvl].push_back(ptbuf);

                        if (!std::isfinite(ptbuf.color) || !(idepthl[i] > 0)) {
                            idepthl[i] = -1;
                            continue; // just skip if something is wrong.
                        }
                    } else
                        idepthl[i] = -1;

                    weightSumsl[i] = 1;
                }
        }
    }


    // make coarse tracking templates for latstRef.
    void Frame::makeCoarseDepthFromStereo(shared_ptr<Frame> frame) {

        mIDepthLeft.resize(setting::numPyramid);
        for (int lvl = 0; lvl < setting::numPyramid; ++lvl) {
            mIDepthLeft[lvl] = Mat::zeros(mpCam->mh[lvl], mpCam->mw[lvl], CV_32F);
            mWeightSumsLeft[lvl] = Mat::zeros(mpCam->mh[lvl], mpCam->mw[lvl], CV_32F);

        }
        float *idepth = (float *) (mIDepthLeft[0].data);
        float *weightSums = (float *) (mWeightSumsLeft[0].data);

        for (shared_ptr<SparePoint> pPoint : frame->vPointsLeft) {
            Vector3f KliP = Vector3f((pPoint->u - mpCam->mcx[0]) * mpCam->mfxi[0],
                                     (pPoint->v - mpCam->mcy[0]) * mpCam->mfyi[0],
                                     1);

            Vector3f ptp = frame->mRcw.cast<float>() * KliP + frame->mtcw.cast<float>() * pPoint->idepth_r;
            float drescale = 1.0f / ptp[2];
            float new_idepth = pPoint->idepth_r * drescale;

            if (!(drescale > 0)) continue;

            int u0 = ptp[0] * drescale;
            int v0 = ptp[1] * drescale;
            int u = u0 * mpCam->mfx[0] + mpCam->mcx[0];
            int v = v0 * mpCam->mfy[0] + mpCam->mcy[0];

            float weight = sqrtf(1e-3 / (pPoint->HdiF + 1e-12));

            idepth[u + mpCam->mw[0] * v] += new_idepth * weight;
            weightSums[u + mpCam->mw[0] * v] += weight;
        }

        makeIDepthWeightL();
        dialteIDepth0To2();
        dialteIDepth2ToTop();
        NormalizeDepth(frame->mPyramidLeft);
    }

    /**
     * 构造当前帧的深度，
     * 1. 对于帧中的每个点
     * @param 当前要处理的帧
     * @return
     */
    void Frame::makeCoarseDepthL0(shared_ptr<Frame> frame) {

        float *idepth = (float *) (mIDepthLeft[0].data);
        float *weightSums = (float *) (mWeightSumsLeft[0].data);
        // make coarse tracking templates for latstRef.
        memset(idepth, 0, sizeof(float) * mpCam->mw[0] * mpCam->mh[0]);
        memset(weightSums, 0, sizeof(float) * mpCam->mw[0] * mpCam->mh[0]);


        for (shared_ptr<SparePoint> pPoint : frame->vPointsLeft) {

            Vector3f KliP = Vector3f((pPoint->u - mpCam->mcx[0]) * mpCam->mfxi[0],
                                     (pPoint->v - mpCam->mcy[0]) * mpCam->mfyi[0],
                                     1);

            Vector3f ptp = frame->mRcw.cast<float>() * KliP + frame->mtcw.cast<float>() * pPoint->idepth_r;
            float drescale = 1.0f / ptp[2];
            float new_idepth = pPoint->idepth_r * drescale;

            if (!(drescale > 0)) continue;

            int u0 = ptp[0] * drescale;
            int v0 = ptp[1] * drescale;
            int u = u0 * mpCam->mfx[0] + mpCam->mcx[0];
            int v = v0 * mpCam->mfy[0] + mpCam->mcy[0];

            float weight = sqrtf(1e-3 / (pPoint->HdiF + 1e-12));

            idepth[u + mpCam->mw[0] * v] += new_idepth * weight;
            weightSums[u + mpCam->mw[0] * v] += weight;
        }
        makeIDepthWeightL();
        dialteIDepth0To2();
        dialteIDepth2ToTop();
        NormalizeDepth(frame->mPyramidLeft);
    }



} //namespace ygz


