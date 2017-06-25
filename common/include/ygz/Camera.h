#ifndef YGZ_CAMERA_H_
#define YGZ_CAMERA_H_

#include "ygz/NumTypes.h"
#include "ygz/Settings.h"
#define PYR_LEVELS 4
namespace ygz {

    struct CameraParam {

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        CameraParam(const float &_fx, const float &_fy, const float &_cx, const float &_cy, const float _bf = 0)
                : fx(_fx), fy(_fy), cx(_cx), cy(_cy), bf(_bf) {
            K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
            fxinv = 1 / fx;
            fyinv = 1 / fy;
            Kinv = K.inverse();
            f = (fx + fy) * 0.5;
            b = bf / f;
        }

        // 从像素到相机
        inline Vector3d Img2Cam(const Vector2f &px) {
            return Vector3d(
                    fxinv * (px[0] - cx),
                    fyinv * (px[1] - cy),
                    1
            );
        }

        // create camera intrinsics in each level
        /** 构造内参数结构
         * 1. 内参数结构分层表示
         * 2. 图像的大小分层表示
         * @param[in] cam 相机数据结构指针
         * @return
         */
        inline void MakeK() {

            mw[0] = setting::imageWidth;
            mh[0] = setting::imageHeight;

            mfx[0] = fx;
            mfy[0] = fy;
            mcx[0] = cx;
            mcy[0] = cy;

            // 第一层到最高层
            for (int level = 1; level < setting::numPyramid; ++level) {
                mw[level] = mw[0] >> level;
                mh[level] = mh[0] >> level;
                mfx[level] = mfx[level - 1] * 0.5;
                mfy[level] = mfy[level - 1] * 0.5;
                mcx[level] = (mcx[0] + 0.5) / ((int) 1 << level) - 0.5;
                mcy[level] = (mcy[0] + 0.5) / ((int) 1 << level) - 0.5;
            }

            for (int level = 0; level < setting::numPyramid; ++level) {
                mK[level] << mfx[level], 0.0, mcx[level], 0.0, mfy[level], mcy[level], 0.0,
                        0.0, 1.0;
                mKi[level] = mK[level].inverse();
                mfxi[level] = mKi[level](0, 0);
                mfyi[level] = mKi[level](1, 1);
                mcxi[level] = mKi[level](0, 2);
                mcyi[level] = mKi[level](1, 2);
            }
        }


        float fx = 0;
        float fy = 0;
        float fxinv = 0;
        float fyinv = 0;
        float cx = 0;
        float cy = 0;
        float b = 0;   // baseline in stereo
        float f = 0;   // focal length
        float bf = 0;   // baseline*focal

        Matrix3f K = Matrix3f::Identity();     // 内参矩阵
        Matrix3f Kinv = Matrix3f::Identity();  // inverse K

        // camera and image parameters in each pyramid
        // 每个金字塔的参数
        Matrix3f mK[PYR_LEVELS];     // K
        Matrix3f mKi[PYR_LEVELS];   // K inverse
        float mfx[PYR_LEVELS];
        float mfy[PYR_LEVELS];
        float mfxi[PYR_LEVELS];
        float mfyi[PYR_LEVELS];
        float mcx[PYR_LEVELS];
        float mcy[PYR_LEVELS];
        float mcxi[PYR_LEVELS];
        float mcyi[PYR_LEVELS];
        int mw[PYR_LEVELS];
        int mh[PYR_LEVELS];


    };

}

#endif