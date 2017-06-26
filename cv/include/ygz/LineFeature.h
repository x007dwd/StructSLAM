//
// Created by bobin on 17-6-25.
//

#ifndef YGZ_STEREO_LINEFEATURE_H
#define YGZ_STEREO_LINEFEATURE_H


#include <ml.h>
#include "ygz/NumTypes.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

namespace ygz {
    struct LineFeature {
        LineFeature() { mGlobalID = -1; }

        LineFeature(Vector2d start_pt, Vector2d end_pt, int lId) {
            mStartPt = start_pt;
            mEndPt = end_pt;
            haveDepth = false;
            mGlobalID = -1;
            mLocalID = lId;
            CalcLinePara();
        }

        static inline Vector3d getHomoPoint(Vector2d pt) {
            Vector3d pt3;
            pt3.head<2>() = pt;
            pt3(3) = 1;
        }

        inline void CalcLinePara() {

            Vector3d StartPt = getHomoPoint(mStartPt);
            Vector3d EndPt = getHomoPoint(mEndPt);
            mLineParam = StartPt.cross(EndPt);
            mLineParam /= mLineParam.head<2>().norm();

        }

        double getLineLen() const { return (mStartPt - mEndPt).norm(); }

        inline Vector2d getGradient(const cv::Mat &xGradient, const cv::Mat &yGradient) const {
            cv::Point2d start(mStartPt(0), mStartPt(1));
            cv::Point2d end(mEndPt(0), mEndPt(1));
            double xsum, ysum;
            cv::LineIterator it(xGradient, start, end, 8);
            for (int i = 0; i < it.count; ++i, ++it) {
                xsum += xGradient.at<float>(it.pos());
                ysum += yGradient.at<float>(it.pos());
            }
            double len = std::sqrt(xsum * xsum + ysum * ysum);
            return Vector2d(xsum / len, ysum / len);

        }


        // data
        Vector2d mStartPt, mEndPt;
        Vector3d mLineParam;
        bool haveDepth;
        Vector2d mDirection;
        cv::Mat mDescriptor;
        unsigned mLocalID;
        long mGlobalID;
    };

    class LineExtract {
        LineExtract();

        /**
            * detect line segment from image
            * @param[in] Image
            */
        void DetectLine(const cv::Mat &Image, std::vector<LineFeature> &LineFeatures);

        /***
         * compute the line descriptor
         * @param lf line feature
         * @param xGradient
         * @param yGradient
         */
        bool ComputeLineDesc(LineFeature &lf, const cv::Mat &xGradient, const cv::Mat &yGradient);

        double ptLineDist(Vector2d pt, Vector3d line);

        double LinesDist(const LineFeature &lf1, const LineFeature &lf2);

        void setFastTrackParam();

        void setDarkLightingParam();

        void setTrackParam();
        void setMatchParam();
        void line_match(const std::vector<LineFeature> &vlf1, const std::vector<LineFeature> &vlf2, std::vector<std::vector<int>>& matches);

        void MatchLine(const std::vector<LineFeature> &lf1, const std::vector<LineFeature> &lf2,
                       std::vector<std::vector<int>>& matches);

        void TrackLine(const std::vector<LineFeature> &lf1, const std::vector<LineFeature> &lf2,
                       std::vector<std::vector<int>>& matches);


        // data

        bool fast_motion = false;
        bool dark_lighting = false;

        double lineDistThresh = 80;
        double lineAngleThresh = 25 * CV_PI / 180.0;
        double desDiffThresh = 0.7;
        double lineOverlapThresh = -1;
        double ratioDist1st2nd = 0.7;

    };

}

#endif //YGZ_STEREO_LINEFEATURE_H
