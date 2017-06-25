#ifndef YGZ_BACKEND_H
#define YGZ_BACKEND_H

#include <set>
#include <deque>
#include "ygz/NumTypes.h"

/**
 * 后端的接口，具体函数请在子类实现
 * 后端需要做的事：
 * 1. 接受前端给出的Key-frame，有单独线程处理Keyframe-Mappoint的优化
 * 2. 管理地图点，向前端提供一个较好的局部地图
 * 3. 管理关键帧，删除时间久远的帧
 */
namespace ygz {

    // forward declare
    struct Frame;
    struct MapPoint;

    // this is the virtual interface class
    class BackendInterface {
    public:

        BackendInterface() {}

        virtual ~BackendInterface() {}

        // 插入新关键帧
        virtual int InsertKeyFrame(shared_ptr<Frame> new_kf) =0;

        // 查询后端状态
        virtual bool IsBusy() =0;

        // 关闭后端
        virtual void Shutdown() =0;

        // 获取局部地图（前端追踪用）
        virtual std::set<shared_ptr<MapPoint>> GetLocalMap() =0;

        // 获取所有的关键帧
        virtual std::deque<shared_ptr<Frame>> GetAllKF() =0;

        // 初始化完成时，更新整张地图
        // 在初始化之前，相机的Twb仅相对第一个帧，但初始化完成后，b系至w系的旋转被估计出来，需要将整个地图转到world系下
        virtual void UpdateWholeMap(const SE3d &Twb) =0;

        virtual void Reset() =0;

    };

}

#endif