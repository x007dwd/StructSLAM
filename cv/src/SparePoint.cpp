//
// Created by bobin on 17-6-12.
//

#include "ygz/SparePoint.h"
#include "ygz/Settings.h"
#include "ygz/Frame.h"
#include "ygz/getElement.h"


namespace ygz {

    PointInterface::PointInterface(int u_, int v_) : u(u_), v(v_) {}

    SparePoint::SparePoint(int u, int v, float type, shared_ptr<CameraParam> cam)
            : PointInterface(u, v), idepth_min(0), idepth_max(NAN), type(type) {
        gradH.setZero();
        for (int i = 0; i < setting::patternNum; ++i) {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];

            Vector3f ptc;
            bool ptcOver = false;
            ptc = host->getInterpolatedElement33BiLin(u + dx, v + dy);
            color[idx] = ptc[0];
            if (!std::isfinite(color[idx])) {
                energy_TH = NAN;
                return;
            }
            // gradH 梯度平方和
            gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
            weights[idx] = sqrtf(
                    setting::outlierTHSumComponent / (setting::outlierTHSumComponent + ptc.tail<2>().squaredNorm()));

        }
        energy_TH = setting::patternNum * setting::outlierTH;
        energy_TH *= setting::overallEnergyTHWeight * setting::overallEnergyTHWeight;

        quality = 10000;
    }


    SparePoint::~SparePoint() {}

    PointStatus SparePoint::TraceLeft(shared_ptr<Frame> frame, Matrix3f K, Vector3f bl) {
        // KRKi为单位矩阵
        Matrix3f KRKi = Matrix3f::Identity().cast<float>();
        // Kt为baseline平移矩阵

        Vector3f Kt;

        Vector2f lastTraceUV;
        float lastTracePixelInterval;

        Kt = K * bl;
        //简化情况的aff
        Vector2f aff;
        aff << 1, 0;

        float bf = -K(0, 0) * bl[0];
        //        printf("bf %f \n", bf);

        Vector3f pr = KRKi * Vector3f(u_r, v_r, 1);
        Vector3f ptpMin = pr + Kt * idepth_min_r;

        float uMin = ptpMin[0] / ptpMin[2];
        float vMin = ptpMin[1] / ptpMin[2];

        unsigned width = frame->mImLeft.cols;
        unsigned height = frame->mImLeft.rows;

        if (!(uMin > setting::boarder && vMin > setting::boarder && uMin < width - setting::boarder &&
              vMin < height - setting::boarder)) {
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        float dist;
        float uMax;
        float vMax;
        Vector3f ptpMax;
        float maxPixSearch = (width + height) * setting::maxPixSearch;

        if (std::isfinite(idepth_max_r)) {
            ptpMax = pr + Kt * idepth_max_r;
            uMax = ptpMax[0] / ptpMax[2];
            vMax = ptpMax[1] / ptpMax[2];

            if (!(uMax > setting::boarder && vMax > setting::boarder && uMax < width - setting::boarder &&
                  vMax < height - setting::boarder)) {
                lastTraceUV = Vector2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = PointStatus::IPS_OOB;
            }

            // ============== check their distance. everything below 2px is OK (->
            // skip). ===================
            dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
            dist = sqrtf(dist);
            if (dist < setting::trace_slackInterval) {
                //				lastTraceUV_Stereo = Vec2f(uMax+uMin,
                //vMax+vMin)*0.5;
                //				lastTracePixelInterval_Stereo=dist;
                //				idepth_stereo = (u_stereo -
                //0.5*(uMax+uMin))/bf;
                //				return lastTraceStatus_Stereo =
                //ImmaturePointStatus::IPS_SKIPPED;
                return lastTraceStatus = PointStatus::IPS_SKIPPED;
            }
            assert(dist > 0);
        } else {
            dist = maxPixSearch;

            // project to arbitrary depth to get direction.
            ptpMax = pr + Kt * 0.01;
            uMax = ptpMax[0] / ptpMax[2];
            vMax = ptpMax[1] / ptpMax[2];

            // direction.
            float dx = uMax - uMin;
            float dy = vMax - vMin;
            float d = 1.0f / sqrtf(dx * dx + dy * dy);

            // set to [setting_maxPixSearch].
            uMax = uMin + dist * dx * d;
            vMax = vMin + dist * dy * d;

            // may still be out!
            if (!(uMax > setting::boarder && vMax > setting::boarder && uMax < width - setting::boarder &&
                  vMax < height - setting::boarder)) {
                lastTraceUV = Vector2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = PointStatus::IPS_OOB;
            }
            assert(dist > 0);
        }

        //		 set OOB if scale change too big.
        if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        // ============== compute error-bounds on result in pixel. if the new interval
        // is not at least 1/2 of the old, SKIP ===================
        float dx = setting::trace_stepsize * (uMax - uMin);
        float dy = setting::trace_stepsize * (vMax - vMin);

        float a = (Vector2f(dx, dy).transpose() * gradH * Vector2f(dx, dy));
        float b = (Vector2f(dy, -dx).transpose() * gradH * Vector2f(dy, -dx));
        float errorInPixel = 0.2f + 0.2f * (a + b) / a;

        if (errorInPixel * setting::trace_minImprovementFactor > dist &&
            std::isfinite(idepth_max_r)) {
            //			lastTraceUV_Stereo = Vec2f(uMax+uMin, vMax+vMin)*0.5;
            //			lastTracePixelInterval_Stereo=dist;
            //			idepth_stereo = (u_stereo - 0.5*(uMax+uMin))/bf;
            //			return lastTraceStatus_Stereo =
            //ImmaturePointStatus::IPS_BADCONDITION;
            //            lastTraceUV = Vec2f(u, v);
            //            lastTracePixelInterval = dist;
            return lastTraceStatus = PointStatus::IPS_BADCONDITION;
        }

        if (errorInPixel > 10)
            errorInPixel = 10;

        // ============== do the discrete search ===================
        dx /= dist;
        dy /= dist;

        if (dist > maxPixSearch) {
            uMax = uMin + maxPixSearch * dx;
            vMax = vMin + maxPixSearch * dy;
            dist = maxPixSearch;
        }

        int numSteps = 1.9999f + dist / setting::trace_stepsize;
        Matrix2f Rplane = KRKi.topLeftCorner<2, 2>();

        float randShift = uMin * 1000 - floorf(uMin * 1000);
        float ptx = uMin - randShift * dx;
        float pty = vMin - randShift * dy;

        Vector2f rotatetPattern[MAX_RES_PER_POINT];
        for (int idx = 0; idx < setting::patternNum; idx++)
            rotatetPattern[idx] = Rplane * Vector2f(patternP[idx][0], patternP[idx][1]);

        if (!std::isfinite(dx) || !std::isfinite(dy)) {
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        float errors[100];
        float bestU = 0, bestV = 0, bestEnergy = 1e10;
        int bestIdx = -1;
        if (numSteps >= 100)
            numSteps = 99;

        for (int i = 0; i < numSteps; i++) {
            float energy = 0;
            for (int idx = 0; idx < setting::patternNum; idx++) {
                float hitColor = frame->GetInterpolatedImgElement33(frame->mImLeft,
                                                                    (float) (ptx + rotatetPattern[idx][0]),
                                                                    (float) (pty + rotatetPattern[idx][1]));

                if (!std::isfinite(hitColor)) {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor - (float) (aff[0] * color[idx] + aff[1]);
                float hw = fabs(residual) < setting::huberTH
                           ? 1
                           : setting::huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);
            }

            errors[i] = energy;
            if (energy < bestEnergy) {
                bestU = ptx;
                bestV = pty;
                bestEnergy = energy;
                bestIdx = i;
            }

            ptx += dx;
            pty += dy;
        }

        // find best score outside a +-2px radius.
        float secondBest = 1e10;
        for (int i = 0; i < numSteps; i++) {
            if ((i < bestIdx - setting::minTraceTestRadius ||
                 i > bestIdx + setting::minTraceTestRadius) &&
                errors[i] < secondBest)
                secondBest = errors[i];
        }
        float newQuality = secondBest / bestEnergy;
        if (newQuality < quality || numSteps > 10)
            quality = newQuality;

        // ============== do GN optimization ===================
        float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
        if (setting::trace_GNIterations > 0)
            bestEnergy = 1e5;
        int gnStepsGood = 0, gnStepsBad = 0;
        for (int it = 0; it < setting::trace_GNIterations; it++) {
            float H = 1, b = 0, energy = 0;
            for (int idx = 0; idx < setting::patternNum; idx++) {
                float hitColor = frame->GetInterpolatedImgElement33(frame->mImLeft,
                                                                    (float) (bestU + rotatetPattern[idx][0]),
                                                                    (float) (bestV + rotatetPattern[idx][1]));

                if (!std::isfinite((float) hitColor)) {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor - color[idx];
                float gdx = frame->GetInterpolatedGradElement33(frame->mGradxPyramidLeft[0],
                                                                (float) (bestU + rotatetPattern[idx][0]),
                                                                (float) (bestV + rotatetPattern[idx][1]));

                float gdy = frame->GetInterpolatedGradElement33(frame->mGradyPyramidLeft[0],
                                                                (float) (bestU + rotatetPattern[idx][0]),
                                                                (float) (bestV + rotatetPattern[idx][1]));
                float dResdDist = dx * gdx + dy * gdy;
                float hw = fabs(residual) < setting::huberTH
                           ? 1
                           : setting::huberTH / fabs(residual);

                H += hw * dResdDist * dResdDist;
                b += hw * residual * dResdDist;
                energy +=
                        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
            }

            if (energy > bestEnergy) {
                gnStepsBad++;

                // do a smaller step from old point.
                stepBack *= 0.5;
                bestU = uBak + stepBack * dx;
                bestV = vBak + stepBack * dy;
            } else {
                gnStepsGood++;

                float step = -gnstepsize * b / H;
                if (step < -0.5)
                    step = -0.5;
                else if (step > 0.5)
                    step = 0.5;

                if (!std::isfinite(step))
                    step = 0;

                uBak = bestU;
                vBak = bestV;
                stepBack = step;

                bestU += step * dx;
                bestV += step * dy;
                bestEnergy = energy;
            }

            if (fabsf(stepBack) < setting::trace_GNThreshold)
                break;
        }

        if (!(bestEnergy < energy_TH * setting::trace_extraSlackOnTH)) {

            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            if (lastTraceStatus == PointStatus::IPS_OUTLIER)
                return lastTraceStatus = PointStatus::IPS_OOB;
            else
                return lastTraceStatus = PointStatus::IPS_OUTLIER;
        }

        // ============== set new interval ===================
        if (dx * dx > dy * dy) {
            idepth_min_r = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
                                (Kt[0] - Kt[2] * (bestU - errorInPixel * dx));
            idepth_max_r = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
                                (Kt[0] - Kt[2] * (bestU + errorInPixel * dx));
        } else {
            idepth_min_r = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
                                (Kt[1] - Kt[2] * (bestV - errorInPixel * dy));
            idepth_max_r = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
                                (Kt[1] - Kt[2] * (bestV + errorInPixel * dy));
        }
        if (idepth_min_r > idepth_max_r)
            std::swap<float>(idepth_min_r, idepth_max_r);

        //  printf("the idpeth_min is %f, the idepth_max is %f \n", idepth_min,
        //  idepth_max);

        if (!std::isfinite(idepth_min_r) || !std::isfinite(idepth_max_r) ||
            (idepth_max_r < 0)) {
            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            return lastTraceStatus = PointStatus::IPS_OUTLIER;
        }

        lastTracePixelInterval = 2 * errorInPixel;
        lastTraceUV = Vector2f(bestU, bestV);
        idepth_r = (u_r - bestU) / bf;
        return lastTraceStatus = PointStatus::IPS_GOOD;
    }

    PointStatus SparePoint::TraceOn(shared_ptr<Frame> frame, Matrix3f hostToFrame_KRKi, Vector3f hostToFrame_Kt,
                                    shared_ptr<CameraParam> cam) {
        //如果已经是OOB了，则不再动它
        if (lastTraceStatus == PointStatus::IPS_OOB)
            return lastTraceStatus;

        bool debugPrint = false;
        unsigned width = frame->mImLeft.cols;
        unsigned height = frame->mImLeft.rows;

        float maxPixSearch = (width + height) * setting::maxPixSearch;

        if (debugPrint)
            printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f "
                           "%f!\n",
                   u, v, mpCF->mnId, frame->mnId, idepth_min, idepth_max,
                   hostToFrame_Kt[0], hostToFrame_Kt[1], hostToFrame_Kt[2]);


        // ============== project min and max. return if one of them is OOB
        // =================== 把二维坐标u,v按照KRKi,
        Vector3f pr = hostToFrame_KRKi * Vector3f(u, v, 1);
        // 由于深度具有不确定性，这里可以获得一个ptpMin，即深度最低限时的点坐标
        Vector3f ptpMin = pr + hostToFrame_Kt * idepth_min;

        float uMin = ptpMin[0] / ptpMin[2];
        float vMin = ptpMin[1] / ptpMin[2];


        if (!(uMin > setting::boarder && vMin > setting::boarder && uMin < width - setting::boarder &&
              vMin < height - setting::boarder)) {
            if (debugPrint)
                printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n", u, v, uMin, vMin,
                       ptpMin[2], idepth_min, idepth_max);
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        float dist;
        float uMax;
        float vMax;
        Vector3f ptpMax;
        if (std::isfinite(idepth_max)) {
            ptpMax = pr + hostToFrame_Kt * idepth_max;
            uMax = ptpMax[0] / ptpMax[2];
            vMax = ptpMax[1] / ptpMax[2];

            if (!(uMax > setting::boarder && vMax > setting::boarder && uMax < width - setting::boarder &&
                  vMax < height - setting::boarder)) {
                if (debugPrint)
                    printf("OOB uMax  %f %f - %f %f!\n", u, v, uMax, vMax);
                lastTraceUV = Vector2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = PointStatus::IPS_OOB;
            }

            // ============== check their distance. everything below 2px is OK (->
            // skip). ===================
            dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
            dist = sqrtf(dist);
            if (dist < setting::trace_slackInterval) {
                if (debugPrint)
                    printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

                lastTraceUV = Vector2f(uMax + uMin, vMax + vMin) * 0.5;
                lastTracePixelInterval = dist;
                return lastTraceStatus = PointStatus::IPS_SKIPPED;
            }
            assert(dist > 0);
        } else {
            dist = maxPixSearch;

            // project to arbitrary depth to get direction.
            ptpMax = pr + hostToFrame_Kt * 0.01;
            uMax = ptpMax[0] / ptpMax[2];
            vMax = ptpMax[1] / ptpMax[2];

            // direction.
            float dx = uMax - uMin;
            float dy = vMax - vMin;
            float d = 1.0f / sqrtf(dx * dx + dy * dy);

            // set to [setting_maxPixSearch].
            uMax = uMin + dist * dx * d;
            vMax = vMin + dist * dy * d;

            // may still be out!
            if (!(uMax > setting::boarder && vMax > setting::boarder && uMax < width - setting::boarder &&
                  vMax < height - setting::boarder)) {
                if (debugPrint)
                    printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax, ptpMax[2]);
                lastTraceUV = Vector2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = PointStatus::IPS_OOB;
            }
            assert(dist > 0);
        }

        // set OOB if scale change too big.
        if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
            if (debugPrint)
                printf("OOB SCALE %f %f %f!\n", uMax, vMax, ptpMin[2]);
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        // ============== compute error-bounds on result in pixel. if the new interval
        // is not at least 1/2 of the old, SKIP ===================
        float dx = setting::trace_stepsize * (uMax - uMin);
        float dy = setting::trace_stepsize * (vMax - vMin);

        float a = (Vector2f(dx, dy).transpose() * gradH * Vector2f(dx, dy));
        float b = (Vector2f(dy, -dx).transpose() * gradH * Vector2f(dy, -dx));
        float errorInPixel = 0.2f + 0.2f * (a + b) / a;

        if (errorInPixel * setting::trace_minImprovementFactor > dist &&
            std::isfinite(idepth_max)) {
            if (debugPrint)
                printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
            lastTraceUV = Vector2f(uMax + uMin, vMax + vMin) * 0.5;
            lastTracePixelInterval = dist;
            return lastTraceStatus = PointStatus::IPS_BADCONDITION;
        }

        if (errorInPixel > 10)
            errorInPixel = 10;

        // ============== do the discrete search ===================
        dx /= dist;
        dy /= dist;

        if (debugPrint)
            printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> "
                           "%f (%.1f %.1f)! ErrorInPixel %.1f!\n",
                   u, v, mpCF->mnId, frame->mnId, idepth_min, uMin, vMin,
                   idepth_max, uMax, vMax, errorInPixel);

        if (dist > maxPixSearch) {
            uMax = uMin + maxPixSearch * dx;
            vMax = vMin + maxPixSearch * dy;
            dist = maxPixSearch;
        }

        int numSteps = 1.9999f + dist / setting::trace_stepsize;
        Matrix2f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

        float randShift = uMin * 1000 - floorf(uMin * 1000);
        float ptx = uMin - randShift * dx;
        float pty = vMin - randShift * dy;

        Vector2f rotatetPattern[MAX_RES_PER_POINT];
        for (int idx = 0; idx < setting::patternNum; idx++)
            rotatetPattern[idx] = Rplane * Vector2f(patternP[idx][0], patternP[idx][1]);

        if (!std::isfinite(dx) || !std::isfinite(dy)) {
            // printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        float errors[100];
        float bestU = 0, bestV = 0, bestEnergy = 1e10;
        int bestIdx = -1;
        if (numSteps >= 100)
            numSteps = 99;

        for (int i = 0; i < numSteps; i++) {
            float energy = 0;
            for (int idx = 0; idx < setting::patternNum; idx++) {
                float hitColor = frame->GetInterpolatedImgElement33(frame->mImLeft,
                                                                    (float) (ptx + rotatetPattern[idx][0]),
                                                                    (float) (pty + rotatetPattern[idx][1]));

                if (!std::isfinite(hitColor)) {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor - (float) (color[idx]);
                float hw = fabs(residual) < setting::huberTH
                           ? 1
                           : setting::huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);
            }

            if (debugPrint)
                printf("step %.1f %.1f (id %f): energy = %f!\n", ptx, pty, 0.0f, energy);

            errors[i] = energy;
            if (energy < bestEnergy) {
                bestU = ptx;
                bestV = pty;
                bestEnergy = energy;
                bestIdx = i;
            }

            ptx += dx;
            pty += dy;
        }

        // find best score outside a +-2px radius.
        float secondBest = 1e10;
        for (int i = 0; i < numSteps; i++) {
            if ((i < bestIdx - setting::minTraceTestRadius ||
                 i > bestIdx + setting::minTraceTestRadius) &&
                errors[i] < secondBest)
                secondBest = errors[i];
        }
        float newQuality = secondBest / bestEnergy;
        if (newQuality < quality || numSteps > 10)
            quality = newQuality;

        // ============== do GN optimization ===================
        float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
        if (setting::trace_GNIterations > 0)
            bestEnergy = 1e5;
        int gnStepsGood = 0, gnStepsBad = 0;
        for (int it = 0; it < setting::trace_GNIterations; it++) {
            float H = 1, b = 0, energy = 0;
            for (int idx = 0; idx < setting::patternNum; idx++) {
                float hitColor = frame->GetInterpolatedImgElement33(frame->mImLeft,
                                                                    (float) (bestU + rotatetPattern[idx][0]),
                                                                    (float) (bestV + rotatetPattern[idx][1]));

                if (!std::isfinite((float) hitColor)) {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor - color[idx];
                float gdx = frame->GetInterpolatedGradElement33(frame->mGradxPyramidLeft[0],
                                                                (float) (bestU + rotatetPattern[idx][0]),
                                                                (float) (bestV + rotatetPattern[idx][1]));
                float gdy = frame->GetInterpolatedGradElement33(frame->mGradyPyramidLeft[0],
                                                                (float) (bestU + rotatetPattern[idx][0]),
                                                                (float) (bestV + rotatetPattern[idx][1]));
                float dResdDist = dx * gdx + dy * gdy;
                float hw = fabs(residual) < setting::huberTH
                           ? 1
                           : setting::huberTH / fabs(residual);

                H += hw * dResdDist * dResdDist;
                b += hw * residual * dResdDist;
                energy +=
                        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
            }

            if (energy > bestEnergy) {
                gnStepsBad++;

                // do a smaller step from old point.
                stepBack *= 0.5;
                bestU = uBak + stepBack * dx;
                bestV = vBak + stepBack * dy;
                if (debugPrint)
                    printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                           it, energy, H, b, stepBack, uBak, vBak, bestU, bestV);
            } else {
                gnStepsGood++;

                float step = -gnstepsize * b / H;
                if (step < -0.5)
                    step = -0.5;
                else if (step > 0.5)
                    step = 0.5;

                if (!std::isfinite(step))
                    step = 0;

                uBak = bestU;
                vBak = bestV;
                stepBack = step;

                bestU += step * dx;
                bestV += step * dy;
                bestEnergy = energy;

                if (debugPrint)
                    printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
                           it, energy, H, b, step, uBak, vBak, bestU, bestV);
            }

            if (fabsf(stepBack) < setting::trace_GNThreshold)
                break;
        }

        // ============== detect energy-based outlier. ===================
        if (!(bestEnergy < energy_TH * setting::trace_extraSlackOnTH))
        {
            if (debugPrint)
                printf("OUTLIER!\n");

            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            if (lastTraceStatus == PointStatus::IPS_OUTLIER)
                return lastTraceStatus = PointStatus::IPS_OOB;
            else
                return lastTraceStatus = PointStatus::IPS_OUTLIER;
        }

        // ============== set new interval ===================
        if (dx * dx > dy * dy) {
            idepth_min =
                    (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
                    (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
            idepth_max =
                    (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
                    (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
        } else {
            idepth_min =
                    (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
                    (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
            idepth_max =
                    (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
                    (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
        }
        if (idepth_min > idepth_max)
            std::swap<float>(idepth_min, idepth_max);

        //  printf("the idpeth_min is %f, the idepth_max is %f \n", idepth_min,
        //  idepth_max);

        if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) ||
            (idepth_max < 0)) {
            // printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min,
            // idepth_max);

            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            return lastTraceStatus = PointStatus::IPS_OUTLIER;
        }

        lastTracePixelInterval = 2 * errorInPixel;
        lastTraceUV = Vector2f(bestU, bestV);
        float H = bestEnergy;
        HdiF = 1.0 / H;
        return lastTraceStatus = PointStatus::IPS_GOOD;
    }

    PointStatus SparePoint::TraceRight(shared_ptr<Frame> frame, Matrix3f K, Vector3f bl) {

        // KRKi为单位矩阵
        Matrix3f KRKi = Matrix3f::Identity().cast<float>();
        // Kt为baseline平移矩阵

        Vector3f Kt;
        Kt = K * bl;
        float bf = K(0, 0) * bl[0];

        Vector3f pr = KRKi * Vector3f(u_r, v_r, 1);
        Vector3f ptpMin = pr + Kt * idepth_min_r;
        float uMin = ptpMin[0] / ptpMin[2];
        float vMin = ptpMin[1] / ptpMin[2];


        unsigned width = frame->mImRight.cols;
        unsigned height = frame->mImRight.rows;

        if (!(uMin > setting::boarder && vMin > setting::boarder && uMin < width - setting::boarder &&
              vMin < height - setting::boarder)) {
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        float dist;
        float uMax;
        float vMax;
        Vector3f ptpMax;
        float maxPixSearch = (width + height) * setting::maxPixSearch;

        if (std::isfinite(idepth_max_r)) {
            ptpMax = pr + Kt * idepth_max_r;
            uMax = ptpMax[0] / ptpMax[2];
            vMax = ptpMax[1] / ptpMax[2];

            if (!(uMax > setting::boarder && vMax > setting::boarder && uMax < width - setting::boarder &&
                  vMax < height - setting::boarder)) {
                lastTraceUV = Vector2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = PointStatus::IPS_OOB;
            }

            // ============== check their distance. everything below 2px is OK (->
            // skip). ===================
            dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
            dist = sqrtf(dist);
            if (dist < setting::trace_slackInterval) {
                return lastTraceStatus = PointStatus::IPS_SKIPPED;
            }
            assert(dist > 0);
        } else {
            dist = maxPixSearch;

            // project to arbitrary depth to get direction.
            ptpMax = pr + Kt * 0.01;
            uMax = ptpMax[0] / ptpMax[2];
            vMax = ptpMax[1] / ptpMax[2];

            // direction.
            float dx = uMax - uMin;
            float dy = vMax - vMin;
            float d = 1.0f / sqrtf(dx * dx + dy * dy);

            // set to [setting_maxPixSearch].
            uMax = uMin + dist * dx * d;
            vMax = vMin + dist * dy * d;

            // may still be out!
            if (!(uMax > setting::boarder && vMax > setting::boarder && uMax < width - setting::boarder &&
                  vMax < height - setting::boarder)) {
                lastTraceUV = Vector2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = PointStatus::IPS_OOB;
            }
            assert(dist > 0);
        }

        //		 set OOB if scale change too big.
        if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        // ============== compute error-bounds on result in pixel. if the new interval
        // is not at least 1/2 of the old, SKIP ===================
        float dx = setting::trace_stepsize * (uMax - uMin);
        float dy = setting::trace_stepsize * (vMax - vMin);
        // gradH 在SparePOint的构造函数中赋值。
        float a = (Vector2f(dx, dy).transpose() * gradH * Vector2f(dx, dy));
        float b = (Vector2f(dy, -dx).transpose() * gradH * Vector2f(dy, -dx));
        float errorInPixel = 0.2f + 0.2f * (a + b) / a;

        if (errorInPixel * setting::trace_minImprovementFactor > dist &&
            std::isfinite(idepth_max_r)) {
            return lastTraceStatus = PointStatus::IPS_BADCONDITION;
        }

        if (errorInPixel > 10)
            errorInPixel = 10;

        // ============== do the discrete search ===================
        dx /= dist;
        dy /= dist;

        if (dist > maxPixSearch) {
            uMax = uMin + maxPixSearch * dx;
            vMax = vMin + maxPixSearch * dy;
            dist = maxPixSearch;
        }
        int numSteps = 1.9999f + dist / setting::trace_stepsize;
        Matrix2f Rplane = KRKi.topLeftCorner<2, 2>();

        float randShift = uMin * 1000 - floorf(uMin * 1000);
        float ptx = uMin - randShift * dx;
        float pty = vMin - randShift * dy;

        Vector2f rotatetPattern[MAX_RES_PER_POINT];
        for (int idx = 0; idx < setting::patternNum; idx++)
            rotatetPattern[idx] = Rplane * Vector2f(patternP[idx][0], patternP[idx][1]);

        if (!std::isfinite(dx) || !std::isfinite(dy)) {

            lastTraceUV = Vector2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = PointStatus::IPS_OOB;
        }

        float errors[100];
        float bestU = 0, bestV = 0, bestEnergy = 1e10;
        int bestIdx = -1;
        if (numSteps >= 100)
            numSteps = 99;

        for (int i = 0; i < numSteps; i++) {
            float energy = 0;
            for (int idx = 0; idx < setting::patternNum; idx++) {
                float hitColor = frame->GetInterpolatedImgElement33(frame->mImRight,
                                                                    (float) (ptx + rotatetPattern[idx][0]),
                                                                    (float) (pty + rotatetPattern[idx][1]));

                if (!std::isfinite(hitColor)) {
                    energy += 1e5;
                    continue;
                }
                // color[8]在构造函数中使用点周围的8个点初始化。
                float residual = hitColor - color[idx];
                float hw = fabs(residual) < setting::huberTH
                           ? 1
                           : setting::huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);
            }

            errors[i] = energy;
            if (energy < bestEnergy) {
                bestU = ptx;
                bestV = pty;
                bestEnergy = energy;
                bestIdx = i;
            }

            ptx += dx;
            pty += dy;
        }

        // find best score outside a +-2px radius.
        float secondBest = 1e10;
        for (int i = 0; i < numSteps; i++) {
            if ((i < bestIdx - setting::minTraceTestRadius ||
                 i > bestIdx + setting::minTraceTestRadius) &&
                errors[i] < secondBest)
                secondBest = errors[i];
        }
        float newQuality = secondBest / bestEnergy;
        if (newQuality < quality || numSteps > 10)
            quality = newQuality;

        // ============== do GN optimization ===================
        float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
        if (setting::trace_GNIterations > 0)
            bestEnergy = 1e5;
        int gnStepsGood = 0, gnStepsBad = 0;
        for (int it = 0; it < setting::trace_GNIterations; it++) {
            float H = 1, b = 0, energy = 0;
            for (int idx = 0; idx < setting::patternNum; idx++) {
                float hitColor = frame->GetInterpolatedImgElement33(frame->mImRight,
                                                                    (float) (bestU + rotatetPattern[idx][0]),
                                                                    (float) (bestV + rotatetPattern[idx][1]));

                if (!std::isfinite((float) hitColor)) {
                    energy += 1e5;
                    continue;
                }
                float residual = hitColor - color[idx];
                float gdx = frame->GetInterpolatedGradElement33(frame->mGradxPyramidRight[0],
                                                                (float) (bestU + rotatetPattern[idx][0]),
                                                                (float) (bestV + rotatetPattern[idx][1]));
                float gdy = frame->GetInterpolatedGradElement33(frame->mGradyPyramidRight[0],
                                                                (float) (bestU + rotatetPattern[idx][0]),
                                                                (float) (bestV + rotatetPattern[idx][1]));
                float dResdDist = dx * gdx + dy * gdy;
                float hw = fabs(residual) < setting::huberTH
                           ? 1
                           : setting::huberTH / fabs(residual);

                H += hw * dResdDist * dResdDist;
                b += hw * residual * dResdDist;
                energy +=
                        weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
            }

            if (energy > bestEnergy) {
                gnStepsBad++;

                // do a smaller step from old point.
                stepBack *= 0.5;
                bestU = uBak + stepBack * dx;
                bestV = vBak + stepBack * dy;
            } else {
                gnStepsGood++;

                float step = -gnstepsize * b / H;
                if (step < -0.5)
                    step = -0.5;
                else if (step > 0.5)
                    step = 0.5;

                if (!std::isfinite(step))
                    step = 0;

                uBak = bestU;
                vBak = bestV;
                stepBack = step;

                bestU += step * dx;
                bestV += step * dy;
                bestEnergy = energy;
            }

            if (fabsf(stepBack) < setting::trace_GNThreshold)
                break;
        }
        // energy_TH 在构造函数中初始化。
        if (!(bestEnergy < energy_TH * setting::trace_extraSlackOnTH)) {

            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            if (lastTraceStatus == PointStatus::IPS_OUTLIER)
                return lastTraceStatus = PointStatus::IPS_OOB;
            else
                return lastTraceStatus = PointStatus::IPS_OUTLIER;
        }

        // ============== set new interval ===================
        if (dx * dx > dy * dy) {
            idepth_min_r = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
                                (Kt[0] - Kt[2] * (bestU - errorInPixel * dx));
            idepth_max_r = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
                                (Kt[0] - Kt[2] * (bestU + errorInPixel * dx));
        } else {
            idepth_min_r = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
                                (Kt[1] - Kt[2] * (bestV - errorInPixel * dy));
            idepth_max_r = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
                                (Kt[1] - Kt[2] * (bestV + errorInPixel * dy));
        }
        if (idepth_min_r > idepth_max_r)
            std::swap<float>(idepth_min_r, idepth_max_r);


        if (!std::isfinite(idepth_min_r) || !std::isfinite(idepth_max_r) ||
            (idepth_max_r < 0)) {
            lastTracePixelInterval = 0;
            lastTraceUV = Vector2f(-1, -1);
            return lastTraceStatus = PointStatus::IPS_OUTLIER;
        }

        lastTracePixelInterval = 2 * errorInPixel;
        lastTraceUV = Vector2f(bestU, bestV);
        idepth_r = (u_r - bestU) / bf;
        return lastTraceStatus = PointStatus::IPS_GOOD;
    }


}