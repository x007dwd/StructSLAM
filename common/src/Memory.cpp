#include "ygz/Memory.h"
#include "ygz/Frame.h"
#include "ygz/MapPoint.h"

#include <unordered_map>

using namespace std;

namespace ygz {

    namespace memory {

        // 真正存储的地方
        std::unordered_map<Frame *, int> allFrames;
        std::unordered_map<MapPoint *, int> allMapPoints;

        std::mutex mutexFrame;
        std::mutex mutexPoint;

        void InsertFrame(Frame *frame) {
            unique_lock<mutex> lockFrame;

            if (allFrames.count(frame) > 0)
                allFrames[frame]++;
            else
                allFrames[frame] = 1;
        }

        void InsertMapPoint(MapPoint *mp) {
            unique_lock<mutex> lockPoint;
            if (allMapPoints.count(mp) > 0)
                allMapPoints[mp]++;
            else
                allMapPoints[mp] = 1;
        }

        void DeleteFrame(Frame *frame) {
            unique_lock<mutex> lockFrame;
            if (allFrames.count(frame) == 0) {
                LOG(WARNING) << "Frame doesn't exist" << endl;
                return;
            }
            allFrames[frame]--;
        }

        void DeleteMapPoint(MapPoint *mp) {
            unique_lock<mutex> lockPoint;
            if (allMapPoints.count(mp) == 0) {
                LOG(INFO) << "MapPoint doesn't exist" << endl;
                return;
            }
            allMapPoints[mp]--;
            //LOG(INFO) << "delete mappoint " << mp->mnId << ", count " << allMapPoints[mp] << endl;
        }

        void OptimizeMemory(bool removeKF) {

            for (auto iter = allFrames.begin(); iter != allFrames.end();) {
                if (iter->second <= 0) {
                    if (removeKF) {
                        delete iter->first;
                        iter = allFrames.erase(iter);
                    } else if (iter->first->mbIsKeyFrame) {
                        iter++;
                    } else {
                        delete iter->first;
                        iter = allFrames.erase(iter);
                    }
                } else {
                    iter++;
                }
            }

            // TODO 在删除地图点时，考虑是否会留有野指针的情况
            for (auto iter = allMapPoints.begin(); iter != allMapPoints.end();) {
                if (iter->second <= 0) {
                    delete iter->first;
                    iter = allMapPoints.erase(iter);
                } else {
                    iter++;
                }
            }
        }

        void CleanAllMemory() {
            for (auto &f: allFrames) {
                delete f.first;
            }
            allFrames.clear();

            for (auto &p: allMapPoints) {
                delete p.first;
            }
            allMapPoints.clear();
        }


    }

}
