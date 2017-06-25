#ifndef YGZ_MEMORY_H_
#define YGZ_MEMORY_H_

#include "ygz/Settings.h"
#include "ygz/NumTypes.h"

// memory class is used to store all the recorded frames and map points as we don't use shared_ptr now
// 统一管理省得出现一些莫名其妙的问题，也请不要在Tracker或Backend里随随便便delete东西

namespace ygz {

    struct Frame;
    struct MapPoint;

    namespace memory {

        // 新建一个帧
        void InsertFrame( Frame* frame );

        // 删掉一个帧，请注意这个帧的引用计数会减一，只有当它引用计数为零时，才会真正从内存中删去
        void DeleteFrame( Frame* frame );

        // 新建一个地图点
        void InsertMapPoint( MapPoint* mp );

        // 删掉一个地图点
        void DeleteMapPoint( MapPoint* mp );

        /**
         * 整理内存（这会删掉引用为零的地图点和帧）
         * @param removeKF  是否需要删除关键帧（默认不删除）
         */
        void OptimizeMemory( bool removeKF = false );

        // 清理内存，这会删掉整个地图
        void CleanAllMemory();

    };
}

#endif