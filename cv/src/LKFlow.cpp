#include "ygz/LKFlow.h"
#include "ygz/Align.h"
#include "ygz/Frame.h"
#include "ygz/Feature.h"

#include <opencv2/video/video.hpp>

namespace ygz {

    int LKFlow(
            const shared_ptr<Frame> ref,
            const shared_ptr<Frame> current,
            VecVector2f &trackPts
    ) {

        if (ref->mPyramidLeft.empty())
            ref->ComputeImagePyramid();

        trackPts.reserve(ref->mFeaturesLeft.size());

        // 匹配局部地图用的 patch, 默认8x8
        uchar patch[align_patch_area] = {0};
        // 带边界的，左右各1个像素
        uchar patch_with_border[(align_patch_size + 2) * (align_patch_size + 2)] = {0};

        int successPts = 0;
        for (shared_ptr<Feature> feat: ref->mFeaturesLeft) {
            // from coarse to fine
            Vector2f trackedPos = feat->mPixel;  // 第零层分辨率下的位置
            Vector2f refPixel = trackedPos;
            bool success = true;

            for (int lvl = setting::numPyramid - 1; lvl >= 0; lvl--) {

                float scale = setting::scaleFactors[lvl];
                float invScale = setting::invScaleFactors[lvl];

                Vector2f posLvl = trackedPos * invScale;   // 第lvl层下的位置
                Vector2f refPixelLvl = refPixel * invScale;

                cv::Mat &img_ref = ref->mPyramidLeft[lvl];

                // outside
                if (refPixelLvl[0] < align_halfpatch_size + 2 || refPixelLvl[1] < align_halfpatch_size + 2 ||
                    refPixelLvl[0] > img_ref.cols - align_halfpatch_size - 2 ||
                    refPixel[1] > img_ref.rows - align_halfpatch_size - 2) {
                    success = false;
                    continue;
                }

                // copy the patch with boarder
                uchar *patch_ptr = patch_with_border;
                for (int y = 0; y < align_patch_size + 2; y++) {
                    for (int x = 0; x < align_patch_size + 2; x++, ++patch_ptr) {
                        Vector2f delta(x - align_halfpatch_size - 1, y - align_halfpatch_size - 1);
                        const Vector2f px(refPixelLvl + delta);
                        if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
                            *patch_ptr = 0;
                        } else {
                            *patch_ptr = GetBilateralInterpUchar(px[0], px[1], img_ref);
                        }
                    }
                }

                // remove the boarder
                uint8_t *ref_patch_ptr = patch;
                for (int y = 1; y < align_patch_size + 1; ++y, ref_patch_ptr += align_patch_size) {
                    uint8_t *ref_patch_border_ptr = patch_with_border + y * (align_patch_size + 2) + 1;
                    for (int x = 0; x < align_patch_size; ++x)
                        ref_patch_ptr[x] = ref_patch_border_ptr[x];
                }

                success = Align2D(
                        current->mPyramidLeft[lvl],
                        patch_with_border,
                        patch,
                        10,
                        posLvl
                );

                if (success == false)
                    break;
                // set the tracked pos
                trackedPos = posLvl * scale;
            }

            if (success) {
                // copy the results
                trackPts.push_back(trackedPos);
                successPts++;
            } else {
                trackPts.push_back(Vector2f(-1, -1));
            }
        }

        return successPts;
    }

    int LKFlow1D(const shared_ptr<Frame> frame) {

        // 匹配局部地图用的 patch, 默认8x8
        uchar patch[align_patch_area] = {0};
        // 带边界的，左右各1个像素
        uchar patch_with_border[(align_patch_size + 2) * (align_patch_size + 2)] = {0};

        int successPts = 0;
        for (shared_ptr<Feature> feat: frame->mFeaturesLeft) {
            // from coarse to fine
            Vector2f trackedPos = feat->mPixel;  // 第零层分辨率下的位置
            Vector2f refPixel = trackedPos;
            Vector2f direction(-1, 0);  // 右图中点应该出现在左侧
            bool success = true;

            for (int lvl = setting::numPyramid - 1; lvl >= 0; lvl--) {

                float scale = setting::scaleFactors[lvl];
                float invScale = setting::invScaleFactors[lvl];

                Vector2f posLvl = trackedPos * invScale;   // 第lvl层下的位置
                Vector2f refPixelLvl = refPixel * invScale;

                cv::Mat &img_ref = frame->mPyramidLeft[lvl];

                // copy the patch with boarder
                uchar *patch_ptr = patch_with_border;
                for (int y = 0; y < align_patch_size + 2; y++) {
                    for (int x = 0; x < align_patch_size + 2; x++, ++patch_ptr) {
                        Vector2f delta(x - align_halfpatch_size - 1, y - align_halfpatch_size - 1);
                        const Vector2f px(refPixelLvl + delta);
                        if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
                            *patch_ptr = 0;
                        } else {
                            *patch_ptr = GetBilateralInterpUchar(px[0], px[1], img_ref);
                        }
                    }
                }

                // remove the boarder
                uint8_t *ref_patch_ptr = patch;
                for (int y = 1; y < align_patch_size + 1; ++y, ref_patch_ptr += align_patch_size) {
                    uint8_t *ref_patch_border_ptr = patch_with_border + y * (align_patch_size + 2) + 1;
                    for (int x = 0; x < align_patch_size; ++x)
                        ref_patch_ptr[x] = ref_patch_border_ptr[x];
                }

                /* 一维align的调用方法，但是对双目校正要求过高，不太现实
                double hinv = 0;
                success = Align1D(
                        frame->mPyramidRight[lvl],
                        direction,
                        patch_with_border,
                        patch,
                        10,
                        posLvl,
                        hinv
                );
                 */

                success = Align2D(
                        frame->mPyramidRight[lvl],
                        patch_with_border,
                        patch,
                        10,
                        posLvl
                );

                if (success == false)
                    break;
                // set the tracked pos
                trackedPos = posLvl * scale;
            }

            if (success) {
                // compute the disparity
                float disparity = refPixel[0] - trackedPos[0];
                feat->mfInvDepth = disparity / frame->mpCam->bf;

                successPts++;
            } else {
                feat->mfInvDepth = -1;
            }
        }

        return successPts;
    }

    int LKFlowCV(
            const shared_ptr<Frame> ref,
            const shared_ptr<Frame> current,
            VecVector2f &trackedPts
    ) {
        vector<cv::Point2f> refPts, currPts;
        for (shared_ptr<Feature> feat: ref->mFeaturesLeft) {
            refPts.push_back(cv::Point2f(feat->mPixel[0], feat->mPixel[1]));
        }
        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(ref->mImLeft, current->mImLeft, refPts, currPts, status, err,
                                 cv::Size(21, 21), 3);
        int successPts = 0;
        for (int i = 0; i < currPts.size(); i++) {
            if (status[i] && (currPts[i].x > setting::boarder && currPts[i].y > setting::boarder &&
                              currPts[i].x < setting::imageWidth - setting::boarder &&
                              currPts[i].y < setting::imageHeight - setting::boarder)) {
                // succeed
                trackedPts.push_back(Vector2f(currPts[i].x, currPts[i].y));
                successPts++;
            } else {
                // failed
                trackedPts.push_back(Vector2f(-1, -1));
            }
        }
        return successPts;
    }

}