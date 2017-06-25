#ifndef YGZ_BACKEND_SLIDING_WINDOW_G2O_H
#define YGZ_BACKEND_SLIDING_WINDOW_G2O_H

#include "ygz/NumTypes.h"
#include "ygz/Settings.h"
#include "ygz/BackendInterface.h"

#include <deque>
#include <set>


/**
 * 用G2O实现的一个滑窗后端
 * （其实G2O并不适合处理滑窗……）
 * 没有Marg和FEJ，啦啦啦
 *
 * 这货后期要改成单独一个线程用的
 */

namespace ygz {

    // forward declare
    class Tracker;

    class BackendSlidingWindowG2O : public BackendInterface {

    public:
        BackendSlidingWindowG2O(shared_ptr<Tracker> tracker) : BackendInterface(), mpTracker(tracker) {}

        virtual ~BackendSlidingWindowG2O() {}

        // 插入新关键帧
        virtual int InsertKeyFrame(shared_ptr<Frame> newKF) override;

        // 查询后端状态
        virtual bool IsBusy() override;

        // 关闭后端
        virtual void Shutdown() override;

        // 获取局部地图（前端追踪用）
        virtual std::set<shared_ptr<MapPoint >> GetLocalMap() override;

        // 获取所有的关键帧
        virtual std::deque<shared_ptr<Frame>> GetAllKF() override { return mpKFs; }

        // 初始化完成时，更新整张地图
        // 在初始化之前，相机的Twb仅相对第一个帧，但初始化完成后，b系至w系的旋转被估计出来，需要将整个地图转到world系下
        virtual void UpdateWholeMap(const SE3d &Twb) override;

        virtual void Reset() override ;
    private: // 一些中间函数
        // 创建新地图点
        int CreateNewMapPoints();

        // 向队列中新加一个帧
        int AddNewKF(shared_ptr<Frame> frameAdd);
        void DeleteKF(int idx);

        bool DeleteRedundantKF();

        // 清理地图点，若ref失效则重新找ref
        int CleanMapPoint();

        // Local BA，分带IMU和不带IMU两个版本。Tracker未初始化时用不带IMU的，初始化之后用带IMU的
        void LocalBAWithIMU(bool verbose = false);

        void LocalBAWithoutIMU(bool verbose = false);

        // 计算两个帧之间的Funcdamental
        Matrix3d ComputeF12(shared_ptr<Frame> f1, shared_ptr<Frame> f2);

    private:
        shared_ptr<Tracker> mpTracker = nullptr; // Tracker指针，需要向Tracker通报一些状态
        shared_ptr<Frame> mpCurrent = nullptr;      // 当前正处理的帧

        std::deque<shared_ptr<Frame>> mpKFs;   // 关键帧队列，会保持一定长度
        std::set<shared_ptr<MapPoint>> mpPoints;   // 局部地图点，同样会操持一定长度

    public:
        // 测试函数
        void testLocalBA(); // Local BA的测试
        void testLocalBAIMU(); // Local BA with IMU的测试

        // Debug only
        // 输出所有KF的信息
        void PrintKFInfo();
    };
}

#endif
