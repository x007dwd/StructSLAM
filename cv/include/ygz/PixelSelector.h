
#pragma once

#include "ygz/Settings.h"
#include "ygz/NumTypes.h"
#include "opencv2/core/core.hpp"


namespace ygz {

    enum PixelSelectorStatus {
        PIXSEL_VOID = 0, PIXSEL_1, PIXSEL_2, PIXSEL_3
    };

    struct Frame;

    class CoarseTracker;

    class PixelSelector {
    public:
        int makeMaps(shared_ptr<Frame> fh, float *map_out, float density,
                     int recursionsLeft = 1, bool plot = false, float thFactor = 1);

        PixelSelector(int w, int h);

        ~PixelSelector();


        const float minUseGrad_pixsel = 10;

        template<int pot>
        inline int gridMaxSelection(const cv::Mat &ImGradX, const cv::Mat &ImGradY, bool *map_out, int w, int h,
                                    float THFac) {

            memset(map_out, 0, sizeof(bool) * w * h);

            int numGood = 0;
            for (int y = 1; y < h - pot; y += pot) {
                for (int x = 1; x < w - pot; x += pot) {
                    int bestXXID = -1;
                    int bestYYID = -1;
                    int bestXYID = -1;
                    int bestYXID = -1;

                    float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

                    //Eigen::Vector3f *grads0; = grads + x + y * w;
                    for (int dx = 0; dx < pot; dx++)
                        for (int dy = 0; dy < pot; dy++) {
                            int idx = dx + dy * w;
                            //Eigen::Vector3f g = grads0[idx];
                            float gdx = ImGradX.at<float>(y + dy, x + dx);
                            float gdy = ImGradY.at<float>(y + dy, x + dx);
                            float sqgd = sqrtf(gdx * gdx + gdy * gdy);
                            float TH = THFac * minUseGrad_pixsel * (0.75f);

                            if (sqgd > TH * TH) {
                                float agx = fabs((float) gdx);
                                if (agx > bestXX) {
                                    bestXX = agx;
                                    bestXXID = idx;
                                }

                                float agy = fabs((float) gdy);
                                if (agy > bestYY) {
                                    bestYY = agy;
                                    bestYYID = idx;
                                }

                                float gxpy = fabs((float) (gdx - gdy));
                                if (gxpy > bestXY) {
                                    bestXY = gxpy;
                                    bestXYID = idx;
                                }

                                float gxmy = fabs((float) (gdx + gdy));
                                if (gxmy > bestYX) {
                                    bestYX = gxmy;
                                    bestYXID = idx;
                                }
                            }
                        }

                    bool *map0 = map_out + x + y * w;

                    if (bestXXID >= 0) {
                        if (!map0[bestXXID])
                            numGood++;
                        map0[bestXXID] = true;
                    }
                    if (bestYYID >= 0) {
                        if (!map0[bestYYID])
                            numGood++;
                        map0[bestYYID] = true;
                    }
                    if (bestXYID >= 0) {
                        if (!map0[bestXYID])
                            numGood++;
                        map0[bestXYID] = true;
                    }
                    if (bestYXID >= 0) {
                        if (!map0[bestYXID])
                            numGood++;
                        map0[bestYXID] = true;
                    }
                }
            }

            return numGood;
        }

        inline int gridMaxSelection(const cv::Mat &ImGradX, const cv::Mat &ImGradY, bool *map_out, int w, int h,
                                    int pot, float THFac) {

            memset(map_out, 0, sizeof(bool) * w * h);

            int numGood = 0;
            for (int y = 1; y < h - pot; y += pot) {
                for (int x = 1; x < w - pot; x += pot) {
                    int bestXXID = -1;
                    int bestYYID = -1;
                    int bestXYID = -1;
                    int bestYXID = -1;

                    float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

//                Eigen::Vector3f *grads0 = grads + x + y * w;
                    for (int dx = 0; dx < pot; dx++)
                        for (int dy = 0; dy < pot; dy++) {
                            int idx = dx + dy * w;
//                        Eigen::Vector3f g = grads0[idx];
                            float gdx = ImGradX.at<float>(y + dy, x + dx);
                            float gdy = ImGradY.at<float>(y + dy, x + dx);
                            float sqgd = sqrtf(gdx * gdx + gdy * gdy);
                            float TH = THFac * minUseGrad_pixsel * (0.75f);

                            if (sqgd > TH * TH) {
                                float agx = fabs((float) gdx);
                                if (agx > bestXX) {
                                    bestXX = agx;
                                    bestXXID = idx;
                                }

                                float agy = fabs((float) gdy);
                                if (agy > bestYY) {
                                    bestYY = agy;
                                    bestYYID = idx;
                                }

                                float gxpy = fabs((float) (gdx - gdy));
                                if (gxpy > bestXY) {
                                    bestXY = gxpy;
                                    bestXYID = idx;
                                }

                                float gxmy = fabs((float) (gdx + gdy));
                                if (gxmy > bestYX) {
                                    bestYX = gxmy;
                                    bestYXID = idx;
                                }
                            }
                        }

                    bool *map0 = map_out + x + y * w;

                    if (bestXXID >= 0) {
                        if (!map0[bestXXID])
                            numGood++;
                        map0[bestXXID] = true;
                    }
                    if (bestYYID >= 0) {
                        if (!map0[bestYYID])
                            numGood++;
                        map0[bestYYID] = true;
                    }
                    if (bestXYID >= 0) {
                        if (!map0[bestXYID])
                            numGood++;
                        map0[bestXYID] = true;
                    }
                    if (bestYXID >= 0) {
                        if (!map0[bestYXID])
                            numGood++;
                        map0[bestYXID] = true;
                    }
                }
            }

            return numGood;
        }

        inline int makePixelStatus(const cv::Mat &ImGradX, const cv::Mat &ImGradY, bool *map, int w, int h,
                                   float desiredDensity, int recsLeft = 5,
                                   float THFac = 1) {
            if (setting::sparsityFactor < 1)
                setting::sparsityFactor = 1;

            int numGoodPoints;

            if (setting::sparsityFactor == 1)
                numGoodPoints = gridMaxSelection<1>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 2)
                numGoodPoints = gridMaxSelection<2>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 3)
                numGoodPoints = gridMaxSelection<3>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 4)
                numGoodPoints = gridMaxSelection<4>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 5)
                numGoodPoints = gridMaxSelection<5>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 6)
                numGoodPoints = gridMaxSelection<6>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 7)
                numGoodPoints = gridMaxSelection<7>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 8)
                numGoodPoints = gridMaxSelection<8>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 9)
                numGoodPoints = gridMaxSelection<9>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 10)
                numGoodPoints = gridMaxSelection<10>(ImGradX, ImGradY, map, w, h, THFac);
            else if (setting::sparsityFactor == 11)
                numGoodPoints = gridMaxSelection<11>(ImGradX, ImGradY, map, w, h, THFac);
            else
                numGoodPoints = gridMaxSelection(ImGradX, ImGradY, map, w, h, setting::sparsityFactor, THFac);

            /*
             * #points is approximately proportional to sparsityFactor^2.
             */

            float quotia = numGoodPoints / (float) (desiredDensity);

            int newSparsity = (setting::sparsityFactor * sqrtf(quotia)) + 0.7f;

            if (newSparsity < 1)
                newSparsity = 1;

            float oldTHFac = THFac;
            if (newSparsity == 1 && setting::sparsityFactor == 1)
                THFac = 0.5;

            if ((abs(newSparsity - setting::sparsityFactor) < 1 && THFac == oldTHFac) ||
                (quotia > 0.8 && 1.0f / quotia > 0.8) || recsLeft == 0) {

                //		printf(" \n");
                // all good
                setting::sparsityFactor = newSparsity;
                return numGoodPoints;
            } else {
                //		printf(" -> re-evaluate! \n");
                // re-evaluate.
                setting::sparsityFactor = newSparsity;
                return makePixelStatus(ImGradX, ImGradY, map, w, h, desiredDensity, recsLeft - 1,
                                       THFac);
            }
        }


        int currentPotential;

        bool allowFast;

        void makeHists(shared_ptr<Frame> frame);

    private:
        Eigen::Vector3i select(shared_ptr<Frame> frame, float *map_out, int pot,
                               float thFactor = 1);

        unsigned char *randomPattern;

        int *gradHist;
        float *ths;
        float *thsSmoothed;
        int thsStep;
        shared_ptr<Frame> gradHistFrame = nullptr;
};
}
