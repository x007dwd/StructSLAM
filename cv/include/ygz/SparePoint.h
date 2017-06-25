//
// Created by bobin on 17-6-12.
//

#ifndef YGZ_STEREO_POINT_H
#define YGZ_STEREO_POINT_H



#include "ygz/NumTypes.h"
#include <ygz/Settings.h>
#include "ygz/Camera.h"
#define MAX_RES_PER_POINT 8


using namespace std;
namespace ygz {
    struct Frame;

    enum ResLocation {
        ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE
    };
    enum ResState {
        IN = 0, OOB, OUTLIER
    };

    enum PointStatus{
        IPS_GOOD=0,					// traced well and good
        IPS_OOB,					// OOB: end tracking & marginalize!
        IPS_OUTLIER,				// energy too high: if happens again: outlier!
        IPS_SKIPPED,				// traced well and good (but not actually traced).
        IPS_BADCONDITION,			// not traced because of bad condition.
        IPS_UNINITIALIZED			// not even traced once.
    };

    struct point_buffer{
        float u;
        float v;
        float idepth;
        float color;
    };

    struct res_point_buffer {
        float idepth;
        float u;
        float v;
        float dx;
        float dy;
        float residual;
        float weight;
        float refColor;
    };
    class PointInterface{
    public:
        PointInterface(int u_, int v_);
        // data

        static  int instanceCounter;
        float color[MAX_RES_PER_POINT];
        float  weights[MAX_RES_PER_POINT];
        float  u,v;
        int idx;
        shared_ptr<Frame> host;
        float  energy_TH;

    };


    class SparePoint : public  PointInterface{
    public:
        enum PtStatus {
            ACTIVE = 0, INACTIVE, OUTLIER, OOB, MARGINALIZED
        };

        inline void setPointStatus(PtStatus s) { status = s; };
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        SparePoint(int u, int v, float type, shared_ptr<CameraParam> cam);
        ~SparePoint();
        PointStatus TraceLeft(shared_ptr<Frame> frame, Matrix3f K, Vector3f bl);

        PointStatus TraceRight(shared_ptr<Frame> frame, Matrix3f K, Vector3f bl);

        PointStatus TraceOn(shared_ptr<Frame> frame, Matrix3f hostToFrame_KRKi, Vector3f hostToFrame_Kt,
                                    shared_ptr<CameraParam> cam);


        // data
        Matrix2f gradH;  // gradient hessian
        /* | dx*dx dx*dy |
         * | dx*dy dy*dy |
         *
         */

        float quality;  // 在迭代中标记当前最好的深度质量
        float u_r, v_r;
        float  type;
        float  idepth_r;
        float  idepth_min;
        float  idepth_max;
        float  idepth_min_r;
        float  idepth_max_r;
        float idepth;
        float HdiF; // inverse of idepth Hessian(Hdd).
        /*
         *    (dr/dd)^T(dr/dd)
         * = Hdd_acc = JIdx2 * Jpdd * Jpdd
         *
         *
         * Jidx2是一个4X4的矩阵
         *  = | dx*dx dx*dy|
         *    | dx*dy dy*dy|
         * */
        Vector3f centerProjectedTo;
        PtStatus status;
        PointStatus lastTraceStatus;
        float lastTracePixelInterval;
        Vector2f lastTraceUV;
        shared_ptr<Frame> mpCF;
    };

}

#endif //YGZ_STEREO_POINT_H
