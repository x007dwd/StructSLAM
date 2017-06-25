#include "ygz/PixelSelector.h"
#include "ygz/Frame.h"
#include "ygz/DSOCoarseTracker.h"
#include "ygz/NumTypes.h"


namespace ygz {

    PixelSelector::PixelSelector(int w, int h) {
        randomPattern = new unsigned char[w * h];
        std::srand(3141592); // want to be deterministic.
        for (int i = 0; i < w * h; i++)
            randomPattern[i] = rand() & 0xFF;
        currentPotential = 3;
        gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
        ths = new float[(w / 32) * (h / 32) + 100];
        thsSmoothed = new float[(w / 32) * (h / 32) + 100];
        allowFast = false;
    }

    PixelSelector::~PixelSelector() {
        delete[] randomPattern;
        delete[] gradHist;
        delete[] ths;
        delete[] thsSmoothed;
    }

    int computeHistQuantil(int *hist, float below) {
        int th = hist[0] * below + 0.5f;
        for (int i = 0; i < 90; i++) {
            th -= hist[i + 1];
            if (th < 0)
                return i;
        }
        return 90;
    }

    void PixelSelector::makeHists(shared_ptr<Frame> frame) {

        gradHistFrame = frame;
        float *mapmax0 = (float *) (frame->mAbsSquaredGradLeft[0].data);

        int w = frame->mImLeft.cols;
        int h = frame->mImLeft.rows;

        int w32 = w / 32;
        int h32 = h / 32;
        thsStep = w32;

        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float *map0 = mapmax0 + 32 * x + 32 * y * w;
                int *hist0 = gradHist; // + 50*(x+y*w32);
                memset(hist0, 0, sizeof(int) * 50);

                for (int j = 0; j < 32; j++)
                    for (int i = 0; i < 32; i++) {
                        int it = i + 32 * x;
                        int jt = j + 32 * y;
                        if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1)
                            continue;
                        int g = sqrtf(map0[i + j * w]);
                        if (g > 48)
                            g = 48;
                        hist0[g + 1]++;
                        hist0[0]++;
                    }

                ths[x + y * w32] = computeHistQuantil(hist0, setting::minGradHistCut) +
                                   setting::minGradHistAdd;
            }

        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float sum = 0, num = 0;
                if (x > 0) {
                    if (y > 0) {
                        num++;
                        sum += ths[x - 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x - 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x - 1 + (y) * w32];
                }

                if (x < w32 - 1) {
                    if (y > 0) {
                        num++;
                        sum += ths[x + 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x + 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x + 1 + (y) * w32];
                }

                if (y > 0) {
                    num++;
                    sum += ths[x + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + (y + 1) * w32];
                }
                num++;
                sum += ths[x + y * w32];

                thsSmoothed[x + y * w32] = (sum / num) * (sum / num);

                if (setting::fixGradTH > 0)
                    thsSmoothed[x + y * w32] = setting::fixGradTH * setting::fixGradTH;
            }
    }


    int PixelSelector::makeMaps(shared_ptr<Frame> frame, float *map_out, float density, int recursionsLeft,
                                bool plot, float thFactor) {
        float numHave = 0;
        float numWant = density;
        float quotia;
        int idealPotential = currentPotential;


        if (frame != gradHistFrame)
            makeHists(frame);

        // select!
        Eigen::Vector3i n = this->select(frame, map_out, currentPotential, thFactor);

        // sub-select!
        numHave = n[0] + n[1] + n[2];
        quotia = numWant / numHave;

        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential + 1) * (currentPotential + 1);
        idealPotential = sqrtf(K / numWant) - 1; // round down.
        if (idealPotential < 1)
            idealPotential = 1;

        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
            // re-sample to get more points!
            // potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            currentPotential = idealPotential;
            return makeMaps(frame, map_out, density, recursionsLeft - 1, plot, thFactor);
        } else if (recursionsLeft > 0 && quotia < 0.25) {
            // re-sample to get less points!

            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;

            currentPotential = idealPotential;
            return makeMaps(frame, map_out, density, recursionsLeft - 1, plot, thFactor);
        }


        int numHaveSub = numHave;
        if (quotia < 0.95) {
            int wh = frame->mImLeft.cols * frame->mImLeft.rows;
            int rn = 0;
            unsigned char charTH = 255 * quotia;
            for (int i = 0; i < wh; i++) {
                if (map_out[i] != 0) {
                    if (randomPattern[rn] > charTH) {
                        map_out[i] = 0;
                        numHaveSub--;
                    }
                    rn++;
                }
            }
        }

        currentPotential = idealPotential;
        return numHaveSub;
    }

    Eigen::Vector3i PixelSelector::select(shared_ptr<Frame> frame,
                                          float *map_out, int pot, float thFactor) {

        unsigned char const *const map0 = frame->mImLeft.data;

        float *mapmax0 = (float *) (frame->mAbsSquaredGradLeft[0].data);
        float *mapmax1 = (float *) (frame->mAbsSquaredGradLeft[1].data);
        float *mapmax2 = (float *) (frame->mAbsSquaredGradLeft[2].data);

        int w = frame->mpCam->mw[0];
        int w1 = frame->mpCam->mw[1];
        int w2 = frame->mpCam->mw[2];
        int h = frame->mpCam->mh[0];

        const Vector2f directions[16] = {
                Vector2f(0, 1.0000), Vector2f(0.3827, 0.9239), Vector2f(0.1951, 0.9808),
                Vector2f(0.9239, 0.3827), Vector2f(0.7071, 0.7071), Vector2f(0.3827, -0.9239),
                Vector2f(0.8315, 0.5556), Vector2f(0.8315, -0.5556), Vector2f(0.5556, -0.8315),
                Vector2f(0.9808, 0.1951), Vector2f(0.9239, -0.3827), Vector2f(0.7071, -0.7071),
                Vector2f(0.5556, 0.8315), Vector2f(0.9808, -0.1951), Vector2f(1.0000, 0.0000),
                Vector2f(0.1951, -0.9808)};

        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));

        float dw1 = setting::gradDownweightPerLevel;
        float dw2 = dw1 * dw1;

        int n3 = 0, n2 = 0, n4 = 0;
        for (int y4 = 0; y4 < h; y4 += (4 * pot))
            for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
                int my3 = std::min((4 * pot), h - y4);
                int mx3 = std::min((4 * pot), w - x4);
                int bestIdx4 = -1;
                float bestVal4 = 0;
                Vector2f dir4 = directions[randomPattern[n2] & 0xF];
                for (int y3 = 0; y3 < my3; y3 += (2 * pot))
                    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                        int x34 = x3 + x4;
                        int y34 = y3 + y4;
                        int my2 = std::min((2 * pot), h - y34);
                        int mx2 = std::min((2 * pot), w - x34);
                        int bestIdx3 = -1;
                        float bestVal3 = 0;
                        Vector2f dir3 = directions[randomPattern[n2] & 0xF];
                        for (int y2 = 0; y2 < my2; y2 += pot)
                            for (int x2 = 0; x2 < mx2; x2 += pot) {
                                int x234 = x2 + x34;
                                int y234 = y2 + y34;
                                int my1 = std::min(pot, h - y234);
                                int mx1 = std::min(pot, w - x234);
                                int bestIdx2 = -1;
                                float bestVal2 = 0;
                                Vector2f dir2 = directions[randomPattern[n2] & 0xF];
                                for (int y1 = 0; y1 < my1; y1 += 1)
                                    for (int x1 = 0; x1 < mx1; x1 += 1) {
                                        assert(x1 + x234 < w);
                                        assert(y1 + y234 < h);
                                        int idx = x1 + x234 + w * (y1 + y234);
                                        int xf = x1 + x234;
                                        int yf = y1 + y234;

                                        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4)
                                            continue;

                                        float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];
                                        float pixelTH1 = pixelTH0 * dw1;
                                        float pixelTH2 = pixelTH1 * dw2;

                                        float ag0 = mapmax0[idx];
                                        if (ag0 > pixelTH0 * thFactor) {
                                            Vector2f ag0d;
                                            ag0d[0] = frame->mGradxPyramidLeft[0].at<float>(yf, xf);
                                            ag0d[1] = frame->mGradyPyramidLeft[0].at<float>(yf, xf);

                                            float dirNorm = fabsf((float) (ag0d.dot(dir2)));
                                            if (!setting::selectDirectionDistribution)
                                                dirNorm = ag0;

                                            if (dirNorm > bestVal2) {
                                                bestVal2 = dirNorm;
                                                bestIdx2 = idx;
                                                bestIdx3 = -2;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx3 == -2)
                                            continue;

                                        float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) +
                                                            (int) (yf * 0.5f + 0.25f) * w1];
                                        if (ag1 > pixelTH1 * thFactor) {
                                            Vector2f ag0d;
                                            ag0d[0] = frame->mGradxPyramidLeft[0].at<float>(yf, xf);
                                            ag0d[1] = frame->mGradyPyramidLeft[0].at<float>(yf, xf);

                                            float dirNorm = fabsf((float) (ag0d.dot(dir3)));
                                            if (!setting::selectDirectionDistribution)
                                                dirNorm = ag1;

                                            if (dirNorm > bestVal3) {
                                                bestVal3 = dirNorm;
                                                bestIdx3 = idx;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx4 == -2)
                                            continue;

                                        float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                                                            (int) (yf * 0.25f + 0.125) * w2];
                                        if (ag2 > pixelTH2 * thFactor) {
                                            Vector2f ag0d;
                                            ag0d[0] = frame->mGradxPyramidLeft[0].at<float>(yf, xf);
                                            ag0d[1] = frame->mGradyPyramidLeft[0].at<float>(yf, xf);

                                            float dirNorm = fabsf((float) (ag0d.dot(dir4)));
                                            if (!setting::selectDirectionDistribution)
                                                dirNorm = ag2;

                                            if (dirNorm > bestVal4) {
                                                bestVal4 = dirNorm;
                                                bestIdx4 = idx;
                                            }
                                        }
                                    }

                                if (bestIdx2 > 0) {
                                    map_out[bestIdx2] = 1;
                                    bestVal3 = 1e10;
                                    n2++;
                                }
                            }

                        if (bestIdx3 > 0) {
                            map_out[bestIdx3] = 2;
                            bestVal4 = 1e10;
                            n3++;
                        }
                    }

                if (bestIdx4 > 0) {
                    map_out[bestIdx4] = 4;
                    n4++;
                }
            }

        return Eigen::Vector3i(n2, n3, n4);
    }
}
