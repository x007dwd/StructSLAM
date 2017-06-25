#ifndef YGZ_COARSE_TRACKER_H
#define YGZ_COARSE_TRACKER_H

/**
 * 下面欢迎来自DSO的CoarseTracker选手
 */

#include "ygz/Settings.h"
#include "ygz/NumTypes.h"
#include "ygz/Camera.h"
#include "ygz/Tracker.h"
#include "ygz/getElement.h"
#include "ygz/PixelSelector.h"
#include "ygz/Settings.h"


#define PYR_LEVELS 4
namespace ygz {
    struct Frame;

    // the coarse tracker is used to estimate the coarse pose of current frame
    class CoarseTracker : public Tracker {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // constuctor, allocate memory and send the pointers into ptrToDelete
        CoarseTracker();

        virtual ~CoarseTracker();

        // 向Tracker插入一对新的图像，返回估计的位姿 Twb
        // override the ORB tracker
        virtual SE3d InsertStereo(
                const cv::Mat &imRectLeft, const cv::Mat &imRectRight,  // 左右两图
                const double &timestamp,                                // 时间
                const VecIMU &vimu) override;                      // 自上个帧采集到的IMU

        void TestTrackerLK();

//        shared_ptr<Frame> mLastRef = nullptr;     // the reference frame
//        shared_ptr<Frame> mNewFrame = nullptr;    // the new coming frame

        // act as pure ouptut
        Vector6d mLastResiduals;
        Vector3d mLastFlowIndicators;
        double mFirstCoarseRMSE = 0;

    private:

        /**
        * 实际的Track函数
        */
        virtual void Track() override;

        /**
         * use lk optical flow to track the features in last frame
         * @return
         */
        virtual bool TrackLastFrame( bool usePoseInfo = false ) override;

        /**
         * Track local map using image alignment
         */
        virtual bool TrackLocalMap(int &inliers) override;

        // 与 Reference KF 的比较
        virtual bool TrackReferenceKF(bool usePoseInfomation = true) override;

        // 插入关键帧
        virtual void InsertKeyFrame() override;

        // reset
        virtual void Reset() override;


        /** @brief track the new coming frame and estimate its pose and light parameters
         * @param[in] newFrameHessian the new frame
         * @param[out] lastToNew_out T_new_last
         * @param[in] coarsestLvl the first pyramid level (default=5)
         * @param[in] minResForAbort if residual > 1.5* minResForAbort, return false
         * @return true if track is good
         */
        bool TrackNewestCoarse(
                shared_ptr<Frame> newFrame,
                SE3d &lastToNew_out,
                int coarsestLvl,
                const Vector5d &minResForAbort);

    public:

        /** 对当前帧的图像金字塔选择点
         * @param[in] newFrame frame pointer
         * @return
         */
        void SelectPoints(shared_ptr<Frame> newFrame);

        // 计算残差
        // compute the residual
        /**
         * @param[in] lvl the pyramid level
         * @param[in] refToNew pose from reference to current
         * @param[in] cutoffTH cut off threshold, if residual > cutoffTH, then make residual = max energy. Similar with robust kernel in g2o.
         * @return the residual vector (a bit complicated, see the the last lines in this func.)
         */
        Vector6d CalcRes(int lvl, const SE3d &refToNew, float cutoffTH);

        // Gauss-Newton(or L-M because we have lambda in tracking)
        /** @brief Gauss-Newton
         * NOTE it uses some cache data in "warped buffers"
         * @param[in] lvl image pyramid level
         * @param[out] H_out Hessian matrix
         * @param[out] b_out bias vector
         * @param[in] refToNew Transform matrix from ref to new
         */
        void CalcGS(int lvl, ygz::Matrix6d &H_out, ygz::Vector6d &b_out, const Sophus::SE3d &refToNew);

        // DSO 跟踪器
        /** @brief DSO tracking
         * NOTE it uses some cache data in "warped buffers"
         * @param[in] frame pointer
         * @param[out] resiudal in Vector4d
         */
        bool TrackNewCoarse();



        // warped buffers
        // 缓存用的buffer, w*h 那么大，也可以看成图像
//        float *buf_warped_idepth;
//        float *buf_warped_u;
//        float *buf_warped_v;
//        float *buf_warped_dx;
//        float *buf_warped_dy;
//        float *buf_warped_residual;
//        float *buf_warped_weight;
//        float *buf_warped_refColor;
        vector<res_point_buffer> res_pt_buf;
//        int buf_warped_n;

//        float *idepth[PYR_LEVELS];
//        float *weightSums[PYR_LEVELS];
//        float *weightSums_bak[PYR_LEVELS];

        std::vector<float *> ptrToDelete;    // all allocated memory, will be deleted in deconstructor
        Accumulator6 acc;


        cv::Mat mImgGradx, mImgGrady;
        PixelSelector *pPixelSelector;


    public:
        void TestTracker();
        int TestStereoMatch();

    };

}


#endif
