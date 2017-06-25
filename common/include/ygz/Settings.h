#ifndef YGZ_SETTINGS_H_
#define YGZ_SETTINGS_H_

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <stdexcept>
#include <glog/logging.h>

#include "Thirdparty/DBoW2/DBoW2/FORB.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include <string>
#include <string.h>
#include "se3.hpp"

// 非常详细的参数调整
// 可以在线调整的，用全局变量；不能的就用常量

#define patternP ygz::setting::staticPattern[8]


namespace ygz {

    typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
            ORBVocabulary;

    namespace setting {

        // 初始化setting里的变量
        void initSettings();

        // 销毁setting里的变量
        void destroySettings();

        // IMU configs
        // 注意这些都是平方过的
        extern double gyrBiasRw2;   // 陀螺仪的随机游走方差
        extern double accBiasRw2;   // 加速度计随机游走方差

        extern double gyrMeasError2;   // 陀螺仪的测量方差
        extern double accMeasError2;   // 加速度计测量方差

        extern double gravity;           // 重力大小

        extern size_t numPyramid;  // 金字塔层数，默认是4

        // 各层金字塔的缩放倍数
        extern float scalePyramid;      // 金字塔每层至下一层的倍数
        extern float *scaleFactors;     // 各层缩放因子
        extern float *invScaleFactors;  // 倒数
        extern float *levelSigma2;      // 缩放倍数平方
        extern float *invLevelSigma2;   // 平方倒数

        // Frame 的格点设置
        // 使用不同分辨率时，请修改此参数
        const int FRAME_GRID_SIZE = 10;
        const int FRAME_GRID_ROWS = 48;    // 480/10
        const int FRAME_GRID_COLS = 76;    // 752/10

        extern float GridElementWidthInv;
        extern float GridElementHeightInv;

        // Local window configs
        extern size_t localWindowSize;  // 滑动窗口大小

        // T_B_C        相机到IMU的外参
        // 讲道理的话这个要上锁，后端会覆盖它
        extern Sophus::SE3d TBC;

        // 原始图像分辨率
        extern int imageWidth;
        extern int imageHeight;

        // 外框
        extern int boarder;

        // ORB Extractor 阈值
        extern int initTHFAST;  // 初始门限
        extern int minTHFAST;   // 纹理较差时使用的门限
        extern int extractFeatures;     // 特征点数量
        const float minShiTomasiScore = 30.0;   // 在计算shi-tomasi分数时的最小阈值

        const int PATCH_SIZE = 31;
        const int HALF_PATCH_SIZE = 15;
        const int EDGE_THRESHOLD = 19;

        // ORBMatcher 的阈值
        const int TH_LOW = 50;
        const int TH_HIGH = 100;
        const int HISTO_LENGTH = 30;

        // 双目匹配的阈值
        const float stereoMatchingTolerance = 2.0;  // 双目匹配时，允许在极线上浮动的v轴偏差，默认+-2

        /*** Tracking 的阈值 ***/
        const int minTrackLastFrameFeatures = 20;   // TrackLastFrame时，允许的最小匹配点
        const int minTrackRefKFFeatures = 20;       // TrackRefKF时，允许的最小匹配点
        const int minPoseOptimizationInliers = 10;  // Tracker过程中，位姿优化时的最小点数
        const int minTrackLocalMapInliers = 30;     // TrackLocalMap 过程中，位姿优化时的最小点数
        // 判断地图点是否在视野内时，最远和最近的阈值
        const float minPointDis = 0.2;
        const float maxPointDis = 20;
        // Keyframe 相关
        extern double keyframeTimeGapInit;      // 初始化时，两个关键帧之间的时间距离
        extern double keyframeTimeGapTracking;  // 正常跟踪，两个关键帧之间的时间距离
        // 初始化阈值
        const size_t minStereoInitFeatures = 500;    // 初始化时当前帧最小特征点数
        const size_t minValidInitFeatures = 100;     // 初始化时，当前帧有深度值的最少特征点数
        const int minInitKFs = 5;                    // IMU初始化时最少关键帧数量
        extern bool trackerComputeMarginals;    // 在Tracker过程中是否要计算Marg

        /*** 后端设置 ***/
        extern int numBackendKeyframes/* = 5*/;      // 后端关键帧的数量
        // 新建地图点时，滤掉深度值不在该区间的点
        const float minNewMapPointInvD = 1 / maxPointDis;  // D = 20m
        const float maxNewMapPointInvD = 1 / minPointDis;   // D = 0.5m

        // Viewer 设置
        const float cameraSize = 0.1;           // 可视化中相机的大小

        /*** DSO Coarse Tracker 设置 ***/
        const float huberTH = 9;              // done
        const float coarseCutoffTH = 20;
        const bool debugout_runquiet = false;
        const bool debugPrint = false;

        // scales internal value to idepth.

        const float SCALE_IDEPTH = 1.0f;
        const float SCALE_XI_ROT = 1.0f;
        const float SCALE_XI_TRANS = 0.5f;
        const float SCALE_F = 50.0f;
        const float SCALE_C = 50.0f;
        const float SCALE_W = 1.0f;

        const int gammaWeightsPixelSelect = 1;
        extern bool dilateDoubleCoarse;
        extern int sparsityFactor;

        extern float minGradHistCut;
        extern float minGradHistAdd;
        extern float fixGradTH;
        extern float gradDownweightPerLevel;
        extern bool selectDirectionDistribution;
        extern int pixelSelectionUseFast;

        extern const int staticPattern[10][40][2];
        extern const int staticPatternNum[10];
        extern const int staticPatternPadding[10];
//        extern const int **patternP;

        const int patternPadding = 2;
        const int patternNum = 8;

        extern float outlierTH;
        extern float outlierTHSumComponent;
        extern float overallEnergyTHWeight;
        extern float desiredImmatureDensity;

        extern int minTraceTestRadius;
        extern int trace_GNIterations;

        extern float maxPixSearch;
        extern float trace_slackInterval;
        extern float trace_stepsize;
        extern float trace_minImprovementFactor;
        extern float trace_GNThreshold;
        extern float trace_extraSlackOnTH;

    }
}

#endif
