//
// Created by bobin on 17-6-13.
//

#ifndef YGZ_STEREO_GETELEMENT_H
#define YGZ_STEREO_GETELEMENT_H

#include "ygz/NumTypes.h"

namespace ygz {
    struct Frame;

    class Accumulator6 {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Matrix6d H;
        Vector6d b;
        size_t num;

        inline void initialize() {
            H.setZero();
            b.setZero();
            memset(SSEData, 0, sizeof(float) * 4 * 21);
            memset(SSEData1k, 0, sizeof(float) * 4 * 21);
            memset(SSEData1m, 0, sizeof(float) * 4 * 21);
            // 矩阵的每个元素上保留了4个float数据, 是因为配合SSE优化进行的, 因为SSE的寄存器是128bit的.
            // 所以在最终统计H的时候需要把Data1m中的相邻的四个float数据加起来放入H.
            num = numIn1 = numIn1k = numIn1m = 0;

            // 每调用一次update, numIn1加1. numIn1满1000进位到numIn1k, numIn1k记录K（10^3）上的数字. numIn1满1000进位到numIn1k, numIn1m记录M（10^6）上的数字.
            // Data1m在数据数量增加超过1M的时候才会加入。
        }

        inline void finish() {
            H.setZero();
            shiftUp(true);
            assert(numIn1 == 0);
            assert(numIn1k == 0);

            int idx = 0;
            for (int r = 0; r < 6; r++)
                for (int c = r; c < 6; c++) {
                    float d = SSEData1m[idx + 0] + SSEData1m[idx + 1] + SSEData1m[idx + 2] +
                              SSEData1m[idx + 3];
                    H(r, c) = H(c, r) = d;
                    ///由于H是对称矩阵，因此最后需要将右上角的数据复制到左下角.
                    idx += 4;
                }
            assert(idx == 4 * 21);
        }

        inline void updateSSE(const __m128 J0, const __m128 J1, const __m128 J2,
                              const __m128 J3, const __m128 J4, const __m128 J5,
                              const __m128 J6) {
            float *pt = SSEData;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J0)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J1)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J2)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0, J6)));
            pt += 4;

            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J1)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J2)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1, J6)));
            pt += 4;


            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J2)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2, J6)));
            pt += 4;

            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3, J6)));
            pt += 4;

            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4, J6)));
            pt += 4;

            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6, J6)));
            pt += 4;

            num += 4;
            numIn1++;
            shiftUp(false);
        }

        inline void updateSSE_eighted(const __m128 J0, const __m128 J1,
                                      const __m128 J2, const __m128 J3,
                                      const __m128 J4, const __m128 J5,
                                      const __m128 J6, const __m128 J7,
                                      const __m128 J8, const __m128 w) {
            float *pt = SSEData;

            __m128 J0w = _mm_mul_ps(J0, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J0)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J1)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J2)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J0w, J8)));
            pt += 4;

            __m128 J1w = _mm_mul_ps(J1, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J1)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J2)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J1w, J8)));
            pt += 4;

            __m128 J2w = _mm_mul_ps(J2, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J2)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J2w, J8)));
            pt += 4;

            __m128 J3w = _mm_mul_ps(J3, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J3)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J3w, J8)));
            pt += 4;

            __m128 J4w = _mm_mul_ps(J4, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J4)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J4w, J8)));
            pt += 4;

            __m128 J5w = _mm_mul_ps(J5, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J5)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J5w, J8)));
            pt += 4;

            __m128 J6w = _mm_mul_ps(J6, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J6)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J6w, J8)));
            pt += 4;

            __m128 J7w = _mm_mul_ps(J7, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J7)));
            pt += 4;
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J7w, J8)));
            pt += 4;

            __m128 J8w = _mm_mul_ps(J8, w);
            _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(J8w, J8)));
            pt += 4;

            num += 4;
            numIn1++;
            shiftUp(false);
        }

        inline void updateSingle(const float J0, const float J1, const float J2,
                                 const float J3, const float J4, const float J5,
                                 const float J6, int off = 0) {
            float *pt = SSEData + off;
            *pt += J0 * J0;
            pt += 4;
            *pt += J1 * J0;
            pt += 4;
            *pt += J2 * J0;
            pt += 4;
            *pt += J3 * J0;
            pt += 4;
            *pt += J4 * J0;
            pt += 4;
            *pt += J5 * J0;
            pt += 4;
            *pt += J6 * J0;
            pt += 4;

            *pt += J1 * J1;
            pt += 4;
            *pt += J2 * J1;
            pt += 4;
            *pt += J3 * J1;
            pt += 4;
            *pt += J4 * J1;
            pt += 4;
            *pt += J5 * J1;
            pt += 4;
            *pt += J6 * J1;
            pt += 4;

            *pt += J2 * J2;
            pt += 4;
            *pt += J3 * J2;
            pt += 4;
            *pt += J4 * J2;
            pt += 4;
            *pt += J5 * J2;
            pt += 4;
            *pt += J6 * J2;
            pt += 4;

            *pt += J3 * J3;
            pt += 4;
            *pt += J4 * J3;
            pt += 4;
            *pt += J5 * J3;
            pt += 4;
            *pt += J6 * J3;
            pt += 4;

            *pt += J4 * J4;
            pt += 4;
            *pt += J5 * J4;
            pt += 4;
            *pt += J6 * J4;
            pt += 4;

            *pt += J5 * J5;
            pt += 4;
            *pt += J6 * J5;
            pt += 4;
            *pt += J6 * J6;
            pt += 4;


            num++;
            numIn1++;
            shiftUp(false);
        }

        inline void updateSingleWeighted(float J0, float J1, float J2, float J3,
                                         float J4, float J5, float w, int off = 0) {

            float *pt = SSEData + off;
            *pt += J0 * J0 * w;
            pt += 4;
            J0 *= w;
            *pt += J1 * J0;
            pt += 4;
            *pt += J2 * J0;
            pt += 4;
            *pt += J3 * J0;
            pt += 4;
            *pt += J4 * J0;
            pt += 4;
            *pt += J5 * J0;
            pt += 4;

            *pt += J1 * J1 * w;
            pt += 4;
            J1 *= w;
            *pt += J2 * J1;
            pt += 4;
            *pt += J3 * J1;
            pt += 4;
            *pt += J4 * J1;
            pt += 4;
            *pt += J5 * J1;
            pt += 4;

            *pt += J2 * J2 * w;
            pt += 4;
            J2 *= w;
            *pt += J3 * J2;
            pt += 4;
            *pt += J4 * J2;
            pt += 4;
            *pt += J5 * J2;
            pt += 4;

            *pt += J3 * J3 * w;
            pt += 4;
            J3 *= w;
            *pt += J4 * J3;
            pt += 4;
            *pt += J5 * J3;
            pt += 4;

            *pt += J4 * J4 * w;
            pt += 4;
            J4 *= w;
            *pt += J5 * J4;
            pt += 4;

            *pt += J5 * J5 * w;
            pt += 4;
            J5 *= w;

            num++;
            numIn1++;
            shiftUp(false);
        }

    private:
        EIGEN_ALIGN16 float SSEData[4 * 21];
        EIGEN_ALIGN16 float SSEData1k[4 * 21];
        EIGEN_ALIGN16 float SSEData1m[4 * 21];
        float numIn1, numIn1k, numIn1m;

        void shiftUp(bool force) {
            if (numIn1 > 1000 || force) {
                for (int i = 0; i < 21; i++)
                    _mm_store_ps(SSEData1k + 4 * i,
                                 _mm_add_ps(_mm_load_ps(SSEData + 4 * i),
                                            _mm_load_ps(SSEData1k + 4 * i)));
                numIn1k += numIn1;
                numIn1 = 0;
                memset(SSEData, 0, sizeof(float) * 4 * 21);
            }

            if (numIn1k > 1000 || force) {
                for (int i = 0; i < 21; i++)
                    _mm_store_ps(SSEData1m + 4 * i,
                                 _mm_add_ps(_mm_load_ps(SSEData1k + 4 * i),
                                            _mm_load_ps(SSEData1m + 4 * i)));
                numIn1m += numIn1k;
                numIn1k = 0;
                memset(SSEData1k, 0, sizeof(float) * 4 * 21);
            }
        }
    };
}


#endif //YGZ_STEREO_GETELEMENT_H
