#ifndef YGZ_MINIMAL_IMAGE_H
#define YGZ_MINIMAL_IMAGE_H

#include "ygz/NumTypes.h"

namespace ygz {


    // 简单的图像类，T是图像数据类型，内部用Eigen存储
    template<typename T>
    class MinimalImage {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        inline MinimalImage(int w_, int h_) : w(w_), h(h_) {
            data = new T[w * h];
            ownData = true;
        }

        /*
        * creates minimal image wrapping around existing memory
        */
        inline MinimalImage(int w_, int h_, T *data_) : w(w_), h(h_) {
            data = data_;
            ownData = false;
        }

        inline ~MinimalImage() {
            if (ownData) {
                delete[] data;
            }
        }

        inline MinimalImage *getClone() {
            MinimalImage *clone = new MinimalImage(w, h);
            memcpy(clone->data, data, sizeof(T) * w * h);
            return clone;
        }

        // get pixel by x,y
        inline T &at(int x, int y) {
            return data[(int) x + ((int) y) * w];
        }

        // get pixel by index
        inline T &at(int i) {
            return data[i];
        }

        // set to zero
        inline void setBlack() {
            memset(data, 0, sizeof(T) * w * h);
        }

        // set to a const value
        inline void setConst(T val) {
            for (int i = 0; i < w * h; i++) {
                data[i] = val;
            }
        }

        // set the pixel to val
        inline void setPixel1(const float &u, const float &v, T val) {
            at(u + 0.5f, v + 0.5f) = val;
        }

        // the the 4 pixels around (u,v) to val
        inline void setPixel4(const float &u, const float &v, T val) {
            at(u + 1.0f, v + 1.0f) = val;
            at(u + 1.0f, v) = val;
            at(u, v + 1.0f) = val;
            at(u, v) = val;
        }

        // set the 9 pixels around (u,v) to val
        inline void setPixel9(const int &u, const int &v, T val) {
            at(u + 1, v - 1) = val;
            at(u + 1, v) = val;
            at(u + 1, v + 1) = val;
            at(u, v - 1) = val;
            at(u, v) = val;
            at(u, v + 1) = val;
            at(u - 1, v - 1) = val;
            at(u - 1, v) = val;
            at(u - 1, v + 1) = val;
        }

        // set a circle around (u,v) to val
        /**     u
         *   +++++++
         *   +     +
         *   +  o  +  v
         *   +     +
         *   +++++++
         */
        inline void setPixelCirc(const int &u, const int &v, T val) {
            for (int i = -3; i <= 3; i++) {
                at(u + 3, v + i) = val;
                at(u - 3, v + i) = val;
                at(u + 2, v + i) = val;
                at(u - 2, v + i) = val;

                at(u + i, v - 3) = val;
                at(u + i, v + 3) = val;
                at(u + i, v - 2) = val;
                at(u + i, v + 2) = val;
            }
        }

        int w = 0;      // width
        int h = 0;      // height
        T *data = nullptr; // data buffer
    private:
        bool ownData = false;        // if we really own the image data (false in copy constructor)
    };

    // types defined from minimal image
    typedef MinimalImage<float> MinimalImageF;              // float
    typedef MinimalImage<Vec3f> MinimalImageF3;             // Vector 3f
    typedef MinimalImage<unsigned char> MinimalImageB;      // unsigned char
    typedef MinimalImage<Vec3b> MinimalImageB3;             // Vector 3 unsigned char
    typedef MinimalImage<unsigned short> MinimalImageB16;   // unsigned short

}

#endif
