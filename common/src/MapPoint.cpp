#include "ygz/MapPoint.h"
#include "ygz/Frame.h"
#include "ygz/Feature.h"
#include "ygz/ORBMatcher.h"
#include "ygz/Memory.h"

#include <thread>
#include <mutex>

using namespace std;

namespace ygz {

    long unsigned int MapPoint::nNextId = 0;

    // 从关键帧新建地图点
    MapPoint::MapPoint(shared_ptr<Frame> pFrame, const size_t &idxF) : mpRefKF(pFrame) {
        assert(pFrame != nullptr);
        assert(idxF < pFrame->mFeaturesLeft.size() && idxF >= 0);

        shared_ptr<Feature> feat = pFrame->mFeaturesLeft[idxF];
        assert(feat->mpPoint == nullptr);

        // 计算该特征点的投影
        Vector3d ptFrame = pFrame->mpCam->Img2Cam(feat->mPixel) * (1.0 / double(feat->mfInvDepth));
        Vector3d Ow = pFrame->mOw;
        mWorldPos = pFrame->mRwc * ptFrame + Ow;

        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector / mNormalVector.norm();

        for (size_t i = 0; i < 32; i++)
            mDescriptor[i] = pFrame->mFeaturesLeft[idxF]->mDesc[i];

        mnId = nNextId++;
    }

    MapPoint::~MapPoint() {
    }

    void MapPoint::ComputeDistinctiveDescriptor() {
        // Retrieve all observed descriptors
        vector<uchar *> vDescriptors;
        if (mObservations.empty())
            return;

        vDescriptors.reserve(mObservations.size());

        for (auto mit = mObservations.begin(), mend = mObservations.end(); mit != mend; mit++) {
            weak_ptr<Frame> pKF = mit->first;
            if (pKF.expired() == false)
                vDescriptors.push_back(pKF.lock()->mFeaturesLeft[mit->second]->mDesc);
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++) {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++) {
                int distij = ORBMatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++) {
            vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian) {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);
            memcpy(mDescriptor, vDescriptors[BestIdx], 32);
        }
    }

    bool MapPoint::SetAnotherRef() {
        for ( auto& obs: mObservations ) {
            if ( obs.first.expired() == false && obs.first.lock() != mpRefKF.lock() ) {
                mpRefKF = obs.first;
                return true;
            }
        }
        return false;
    }

    int MapPoint::GetObsFromKF(shared_ptr<Frame> pKF) {
        weak_ptr<Frame> idx(pKF);
        auto iter = mObservations.find( idx );
        if (iter == mObservations.end())
            return -1;
        return iter->second;
    }

    int MapPoint::Observations() {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations.size();
    }

    void MapPoint::SetWorldPos(const Vector3d &Pos) {
        unique_lock<mutex> lock(mMutexPos);
        mWorldPos = Pos;
    }

    void MapPoint::UpdateWorldPos() {
        if (mpRefKF.expired() == false)
            mWorldPos = mpRefKF.lock()->UnprojectStereo(mObservations[mpRefKF]);
    }

    Vector3d MapPoint::GetWorldPos() {
        return mWorldPos;
    }

    Vector3d MapPoint::GetNormal() {
        return mNormalVector;
    }

    void MapPoint::AddObservation(shared_ptr<Frame> pKF, size_t idx) {
        assert(pKF);
        assert(pKF->mbIsKeyFrame);
        assert(idx < pKF->mFeaturesLeft.size());

        if (mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;
    }

    MapPoint::ObsMap MapPoint::GetObservations() {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    void MapPoint::SetBadFlag() {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mState = BAD;
        mObservations.clear();
    }

    void MapPoint::IncreaseVisible(int n) {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapPoint::SetMarged() {
        mState = MapPoint::MARGED;
    }

    void MapPoint::IncreaseFound(int n) {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapPoint::GetFoundRatio() {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }


    void MapPoint::UpdateNormalAndDepth() {

        // 更新地图点的法线和深度
        if (mObservations.empty()) return;

        Vector3d normal(0, 0, 0);
        int n = 0;
        for (auto &obs: mObservations) {
            weak_ptr<Frame> kf = obs.first;
            if (kf.expired() == true)
                continue;
            Vector3d Owi = kf.lock()->mOw;
            Vector3d normali = mWorldPos - Owi;
            normali.normalize();
            normal = normal + normali;
            n++;
        }
        mNormalVector = normal / n;
    }

}
