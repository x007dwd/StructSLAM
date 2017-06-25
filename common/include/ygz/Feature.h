#ifndef YGZ_FEATURE_H_
#define YGZ_FEATURE_H_

#include "ygz/NumTypes.h"
#include "ygz/Settings.h"

using namespace std;

namespace ygz {

    // forward declare
    struct MapPoint;

    // Feature 一般是指一个ORB点，参数化中有它的invDepth。没有关联地图点时，mpPoint为空
    struct Feature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Vector2f mPixel = Vector2f(0, 0);    // 像素位置
        size_t mLevel = 0;    // 所在金字塔位置
        float mScore = 0;    // 评分
        float mAngle = 0;    // 角度

        shared_ptr<MapPoint> mpPoint = nullptr;  // 对应的地图点
        float mfInvDepth = -1;        // 逆深度,小于零说明无效
        uchar mDesc[32];              // 256 bits of ORB feature (32x8)
        float mfRightU = -1;           // 如果是左眼的特征，当右眼存在有效值时，此处为右眼特征的u轴坐标

        // flags
        bool mbOutlier = false;       // true if it is an outlier

    };
}


#endif
