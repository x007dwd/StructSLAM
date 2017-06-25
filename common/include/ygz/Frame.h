#ifndef YGZ_FRAME_H_
#define YGZ_FRAME_H_

#include <opencv2/core/core.hpp>
#include <vector>

#include "ygz/NumTypes.h"
#include "ygz/Settings.h"
#include "ygz/Camera.h"
#include "ygz/IMUData.h"
#include "ygz/IMUPreIntegration.h"
#include "ygz/SparePoint.h"

using cv::Mat;
using namespace std;

/**
 * Frame 统一由Memory管理。
 * 创建Frame的地方，如果不再使用这个Frame，请用memory::DeleteFrame解除对它的引用
 */

namespace ygz {

    struct Frame;

    struct Feature;

    class KeyFrame;

    struct MapPoint;

    class  SparePoint;

    // 帧结构

    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // copy constructor
        Frame(const Frame &frame);

        // constructor from two images
        // 请注意左右图必须是已经校正过的
        Frame(
                const cv::Mat &left, const cv::Mat &right,
                const double &timestamp,
                shared_ptr<CameraParam> cam,
                const VecIMU &imuSinceLastFrame);

        Frame() {
            SetPose(SE3d());
            mnId = nNextId++;
        }

        virtual ~Frame();

        void SetDelete(shared_ptr<Frame> pNextKF = nullptr);

        // 计算图像金字塔
        // TODO 针对缩放率为2的图像进行优化
        void ComputeImagePyramid();

        /**
         * 计算场景场景的深度中位数
         * @param q 百分比
         */
        double ComputeSceneMedianDepth(const int &q);


        // set the pose of this frame and update the inner variables, give Twb
        // 设置位姿，并更新Rcw, tcw等一系列量
        void SetPose(const SE3d &Twb);

        // 设置Ground truth的位姿，debug用
        void SetPoseGT(const SE3d &TwbGT) {
            mTwbGT = TwbGT;
            mbHaveGT = true;
        }

        // 获得位姿 Twb
        SE3d GetPose() const {
            if (mbIsKeyFrame == false)
                return SE3d(mRwb, mTwb);
            else {
                // TODO 上锁
                return SE3d(mRwb, mTwb);
            }
        }

        SE3d GetPoseGT() const {
            if (mbHaveGT)
                return mTwbGT;
            return SE3d();
        }

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        /**
         * 判断路标点是否在视野中
         * 会检测像素坐标、距离和视线角
         * @param pMP   被检测的地图点
         * @param viewingCosLimit   视线夹角cos值，如0.5为60度，即视线夹角大于60度时判定为负
         * @param boarder   图像边界
         * @return
         */
        bool isInFrustum(shared_ptr<MapPoint> pMP, float viewingCosLimit, int boarder = 20);

        /** 计算一个特征是否在grid里
         * @param feature 需要计算的特征
         * @param posX 若在某网格内，返回该网格的x坐标
         * @param posY 若在某网格内，返回该网格的y坐标
         * @return
         */
        bool PosInGrid(const shared_ptr<Feature> feature, int &posX, int &posY);

        /** 获取一定区域内的地图点
         * @param x 查找点的x坐标
         * @param y 查找点的y坐标
         * @param r 半径
         * @param minLevel 最小层数
         * @param maxLevel 最大层数
         * @return
         */
        vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = 0,
                                         const int maxLevel = 0) const;

        /**
         * 将第i个特征投影到世界坐标
         * @param i 特征点的下标
         * @return
         */
        Vector3d UnprojectStereo(const int &i);

        /**
         * 计算某个指定帧过来的预积分
         * @param pLastF 上一个帧（需要知道位移和速度）
         * @param imupreint 预积分器
         */
        void ComputeIMUPreIntSinceLastFrame(shared_ptr<Frame> pLastF, IMUPreIntegration &imupreint);

        /**
         * 计算从参考帧过来的预积分
         * 结果存储在自带的预积分器中
         */
        inline void ComputeIMUPreInt() {
            if (mpReferenceKF.expired() == false)
                ComputeIMUPreIntSinceLastFrame(mpReferenceKF.lock(), mIMUPreInt);
        }

        // update the navigation status using preintegration
        /**
         * @param imupreint 已经积分过的预积分器
         * @param gw 重力向量
         */
        void UpdatePoseFromPreintegration(const IMUPreIntegration &imupreint, const Vector3d &gw);

        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        // 将特征点分配到网格中
        void AssignFeaturesToGrid();

        // 获取所有特征点的描述子
        vector<Mat> GetAllDescriptor() const;

        // 判定是否关键帧
        bool IsKeyFrame() const { return mbIsKeyFrame; }

        // 将当前帧设成关键帧
        bool SetThisAsKeyFrame();

        // 计算词袋
        void ComputeBoW();

        // accessors
        // TODO 请注意上锁的问题

        inline Vector3d Speed() const { return mSpeedAndBias.segment<3>(0); }

        inline Vector3d BiasG() const { return mSpeedAndBias.segment<3>(3); }

        inline Vector3d BiasA() const { return mSpeedAndBias.segment<3>(6); }

        inline void SetSpeedBias(const Vector3d &speed, const Vector3d &biasg, const Vector3d &biasa) {
            mSpeedAndBias.segment<3>(0) = speed;
            mSpeedAndBias.segment<3>(3) = biasg;
            mSpeedAndBias.segment<3>(6) = biasa;
        }

        inline void SetSpeed(const Vector3d &speed) {
            mSpeedAndBias.segment<3>(0) = speed;
        }

        inline void SetBiasG(const Vector3d &biasg) {
            mSpeedAndBias.segment<3>(3) = biasg;
        }

        inline void SetBiasA(const Vector3d &biasa) {
            mSpeedAndBias.segment<3>(6) = biasa;
        }

        // 取得P+R的6维向量
        inline Vector6d PR() const {
            Vector6d pr;
            pr.head<3>() = mTwb;
            pr.tail<3>() = mRwb.log();
            return pr;
        }

        const IMUPreIntegration &GetIMUPreInt() { return mIMUPreInt; }

        void EraseMapPointMatch(shared_ptr<MapPoint> mp);

        void EraseMapPointMatch(const size_t &idx);

        int TrackedMapPoints(const int &minObs);

        // coordinate transform: world, camera, pixel
        inline Vector3d World2Camera(const Vector3d &p_w, const SE3d &T_c_w) const {
            return T_c_w * p_w;
        }

        inline Vector3d Camera2World(const Vector3d &p_c, const SE3d &T_c_w) const {
            return T_c_w.inverse() * p_c;
        }

        inline Vector2d Camera2Pixel(const Vector3d &p_c) const {
            return Vector2d(
                    mpCam->fx * p_c(0, 0) / p_c(2, 0) + mpCam->cx,
                    mpCam->fy * p_c(1, 0) / p_c(2, 0) + mpCam->cy
            );
        }

        inline Vector3d Pixel2Camera(const Vector2d &p_p, float depth = 1) const {
            return Vector3d(
                    (p_p(0, 0) - mpCam->cx) * depth / mpCam->fx,
                    (p_p(1, 0) - mpCam->cy) * depth / mpCam->fy,
                    depth
            );
        }

        inline Vector3d Pixel2World(const Vector2d &p_p, const SE3d &TCW, float depth = 1) const {
            return Camera2World(Pixel2Camera(p_p, depth), TCW);
        }

        inline Vector2d World2Pixel(const Vector3d &p_w, const SE3d &TCW) const {
            return Camera2Pixel(World2Camera(p_w, TCW));
        }

        // ---------------------------------------------------------------
        // DSO Function
        //

        /** 构造梯度
         * 1. 针对图像金字塔中的每一层，计算梯度dxdy以及梯度幅度。
         * 2. 将dxdyabs分别保存于Mat结构中。
         *
         * @param[in] mPyramid the pyramid image
         * @param[out] mGradxPyramid gradient x of the image in pyramid
         * @param[out] mGradxPyramid gradient y of the image in pyramid
         * @param[out] mGradxPyramid gradient amptitude of the image in pyramid
         * @return
         */
        void makeGradient(const std::vector<Mat> &mPyramid, std::vector<Mat> &mGradxPyramid, std::vector<Mat> &mGradyPyramid,vector<Mat>& mAbsSquaredGrad);



        /** 双线性插值获取图像（uchar数据）中点的值
         *
         * **/
        double GetInterpolatedImgElement33(const Mat &img, const float x,const float y);

        /** 双线性插值获取图像（float数据）中点的值
         *
         * **/
        float GetInterpolatedGradElement33(const Mat &img, const float x,const float y);

        /** 双线性插值获取本帧图像中点的值
         *
         * **/
        Eigen::Vector3f getInterpolatedElement33BiLin(const float x, const float y);

        /** 构造sparePoint
         * 1. 针对选出的点，进行trace right
         * 2. 针对trace right匹配好的点，进行保存到spare point容器中
         * @param[in] floatMap/boolmap 选出的点映射
         * @param[out] vecSparePoints 保存的spare point容器
         * */
        void FloatMapToPoint(cv::Mat floatMap, vector<shared_ptr<SparePoint>> &vecSparePoints);

        /** 根据参考帧计算参考图像的深度 temporal stereo
        * @param[in] frame 参考图像
        * @return
        */
        void makeCoarseDepthL0(shared_ptr<Frame> frame);

        /** 使用双目计算参考图像的深度static stereo
         * @param[in] frame 参考图像
         * @return
         */
        void makeCoarseDepthFromStereo(shared_ptr<Frame> frame);


        void makeIDepthWeightL();
        void dialteIDepth0To2();
        void dialteIDepth2ToTop();
        void NormalizeDepth(std::vector<Mat> PyramidImg);


        // ---------------------------------------------------------------
        // Data
        // Frame的数据是不用上锁的

        // time stamp
        double mTimeStamp = 0;

        // image
        Mat mImLeft, mImRight;    // 左/右图像，显示用


        shared_ptr<CameraParam> mpCam = nullptr; // 相机参数
        Vector3f mBaseLine;

        // 请注意特征点以左眼为准
        std::vector<shared_ptr<Feature>> mFeaturesLeft;    // 左眼特征点容器
        std::vector<shared_ptr<Feature>> mFeaturesRight;   // 右眼特征点容器

        // 左右目的金字塔
        std::vector<cv::Mat> mPyramidLeft;
        std::vector<cv::Mat> mPyramidRight;

        long unsigned int mnId = 0;           // id
        static long unsigned int nNextId;     // next id
        long unsigned int mnKFId = 0;         // keyframe id
        static long unsigned int nNextKFId;     // next keyframe id

        // pose, speed and bias
        // 这些都是状态量，优化时候的中间量给我放到优化相关的struct里去
        SO3d mRwb;  // body rotation
        Vector3d mTwb = Vector3d(0, 0, 0); // body translation
        SpeedAndBias mSpeedAndBias = Vector9d::Zero(); // V and bias

        // For pose optimization, use as prior and prior information(inverse covariance)
        // TODO 这个是否需要放在这里？
        Matrix15d mMargCovInv = Matrix15d::Zero();

        // 参考的关键帧
        weak_ptr<Frame> mpReferenceKF;    // 这里应该是weak?否则所有的关键帧都会被保留

        // 是否关键帧
        bool mbIsKeyFrame = false;

        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // 从上一个帧到这里的IMU
        VecIMU mvIMUDataSinceLastFrame;   // 也可能是上一个关键帧
        IMUPreIntegration mIMUPreInt;   // 至上个关键帧的预积分器

        // 位姿相差的中间量
        Matrix3d mRcw = Matrix3d::Identity(); ///< Rotation from world to camera
        Vector3d mtcw = Vector3d::Zero(); ///< Translation from world to camera
        Matrix3d mRwc = Matrix3d::Identity(); ///< Rotation from camera to world
        Vector3d mOw = Vector3d::Zero();  ///< =mtwc,Translation from camera to world

        // 划分特征点的网格
        std::vector<std::size_t> mGrid[setting::FRAME_GRID_COLS][setting::FRAME_GRID_ROWS];

        // 词典
        static shared_ptr<ORBVocabulary> pORBvocabulary;

        // Debug stuffs
        SE3d mTwbGT;    // Ground truth pose
        bool mbHaveGT = false;  // 有没有Ground Truth的位姿

        std::vector<cv::Mat> mGradxPyramidLeft;
        std::vector<cv::Mat> mGradyPyramidLeft;
        std::vector<cv::Mat> mAbsSquaredGradLeft;

        std::vector<cv::Mat> mGradxPyramidRight;
        std::vector<cv::Mat> mGradyPyramidRight;
        std::vector<cv::Mat> mAbsSquaredGradRight;

        std::vector<cv::Mat> mIDepthLeft;
        std::vector<cv::Mat> mWeightSumsLeft;
        std::vector<cv::Mat> mWeightSumsBakLeft;

        SE3d mTWC;
        vector<shared_ptr<SparePoint>> vPointsLeft;

        // pc buffers
        // 每一层都是w*h的矩阵，w,h为该层金字塔的分辨率
        vector<vector<point_buffer>> pc_buffer;
        float *pc_u[PYR_LEVELS];            // x 坐标
        float *pc_v[PYR_LEVELS];            // y 坐标
        float *pc_idepth[PYR_LEVELS];       // 参考帧的 inv depth
        float *pc_color[PYR_LEVELS];
        int pc_n[PYR_LEVELS];               // 每层pyramid的点数量



    };

}

#endif
