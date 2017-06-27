//
// Created by bobin on 17-6-25.
//

#include "cv/include/ygz/LineFeature.h"
#include "opencv2/core/core.hpp"
#include "lsd/lsd.h"
#include "ygz/Settings.h"
#include <vector>
#include "ygz/NumTypes.h"

using namespace cv;
using namespace std;
namespace ygz {

    LineExtract::LineExtract() {
        fast_motion = false;
        dark_lighting = false;

    }

    void LineExtract::DetectLine(const Mat &Image, vector<LineFeature> &LineFeatures) {

        assert(Image.type() == CV_32F);
        Mat Gray;
        if (Image.channels() == 3) {
            cvtColor(Image, Gray, CV_RGB2GRAY);
        } else {
            Gray = Image.clone();
        }
        Mat GrayFloat;
        Gray.convertTo(GrayFloat, CV_64F);
        double *lsd_out;
        int nout;
        lsd_out = lsd(&nout, (double *) (GrayFloat.data), GrayFloat.cols, GrayFloat.rows);
        LineFeatures.resize(nout);
        double a, b, c, d;
        // lsd output x1,y1,x2,y2,width,p,-log10(NFA)
        int out_dim = 7;
        for (int i = 0; i < nout; ++i) {
            a = lsd_out[i * out_dim];
            b = lsd_out[i * out_dim + 1];
            c = lsd_out[i * out_dim + 2];
            d = lsd_out[i * out_dim + 3];

            if (std::sqrt((a - c) * (a - c) + (b - d) * (b - d)) >= setting::lineLenThreshold) {
                Vector2d start, end;
                start << a, b;
                end << c, d;
                LineFeatures.push_back(LineFeature(start, end, i));
            }
        }
        cv::Mat xGradImg, yGradImg;
        int ddepth = CV_64F;
        cv::Sobel(GrayFloat, xGradImg, ddepth, 1, 0, 3); // Gradient X
        cv::Sobel(GrayFloat, yGradImg, ddepth, 0, 1, 3); // Gradient Y

        for (int i = 0; i < LineFeatures.size(); ++i) {
            //	computeMSLD(lines[i], &xGradImg, &yGradImg);
            LineFeatures[i].mDirection = LineFeatures[i].getGradient(xGradImg, yGradImg);
            ComputeLineDesc(LineFeatures[i], xGradImg, yGradImg);
        }
    }

    int computeSubPSR(const Mat &xGradient, const Mat &yGradient, cv::Point2d p, double s, cv::Point2d g,
                      vector<double> &vs) {

        /* input: p - 2D point position
        s - side length of square region
        g - unit vector of gradient of line
        output: vs = (v1, v2, v3, v4)
        */

        int width, height;
        assert(xGradient.rows == yGradient.rows);
        assert(xGradient.cols == yGradient.cols);
        width = xGradient.cols;
        height = xGradient.rows;

        double tl_x = floor(p.x - s / 2), tl_y = floor(p.y - s / 2);
        if (tl_x < 0 || tl_y < 0 ||
            tl_x + s + 1 > width || tl_y + s + 1 > height)
            return 0; // out of image
        double v1 = 0, v2 = 0, v3 = 0, v4 = 0;
        for (int y = tl_y; y < tl_y + s; ++y) {
            for (int x = tl_x; x < tl_x + s; ++x) {
                double tmp1 =
                        xGradient.at<float>(y, x) * g.x + yGradient.at<float>(y, x) * g.y;
                double tmp2 =
                        xGradient.at<float>(y, x) * (-g.y) + yGradient.at<float>(y, x) * g.x;
                if (tmp1 >= 0)
                    v1 = v1 + tmp1;
                else
                    v2 = v2 - tmp1;
                if (tmp2 >= 0)
                    v3 = v3 + tmp2;
                else
                    v4 = v4 - tmp2;
            }
        }
        vs.resize(4);
        vs[0] = v1;
        vs[1] = v2;
        vs[2] = v3;
        vs[3] = v4;
        return 1;
    }

    bool LineExtract::ComputeLineDesc(LineFeature &lf, const cv::Mat &xGradient, const Mat &yGradient) {

        lf.mDirection = lf.getGradient(xGradient, yGradient);
        int s = 5 * xGradient.cols / 800.0;
        double lineLen = (lf.mStartPt - lf.mEndPt).norm();
        vector<vector<double>> GDM;
        double step = setting::msld_sample_interval;
        for (int i = 0; i * step < lineLen; ++i) {
            vector<double> GDM_cols;
            GDM_cols.reserve(36);
            Vector2d pt = lf.mStartPt + (lf.mStartPt - lf.mEndPt) * (i * step / lineLen);

            bool fail = false;
            for (int j = -4; j < 4; ++j) {
                pt += j * s * lf.mDirection;
                cv::Point2d cvpt(pt(0), pt(1));
                cv::Point2d gd(lf.mDirection(0), lf.mDirection(1));
                vector<double> psr(4);
                if (computeSubPSR(xGradient, yGradient, cvpt, s, gd, psr)) {
                    GDM_cols.push_back(psr[0]);
                    GDM_cols.push_back(psr[1]);
                    GDM_cols.push_back(psr[2]);
                    GDM_cols.push_back(psr[3]);
                } else {
                    fail = true;
                    continue;
                }

            }
            GDM.push_back(GDM_cols);

        }
        cv::Mat MS(72, 1, CV_32F);
        if (GDM.size() == 0) {
            for (int i = 0; i < MS.rows; ++i)
                MS.at<double>(i, 0) = rand(); // if not computable, assign random num
            lf.mDescriptor = MS;
            return 0;
        }

        double gauss[9] = {0.24142, 0.30046, 0.35127, 0.38579, 0.39804,
                           0.38579, 0.35127, 0.30046, 0.24142};
        for (int i = 0; i < 36; ++i) {
            double sum = 0, sum2 = 0, mean, std;
            for (int j = 0; j < GDM.size(); ++j) {
                GDM[j][i] = GDM[j][i] * gauss[i / 4];
                sum += GDM[j][i];
                sum2 += GDM[j][i] * GDM[j][i];
            }
            mean = sum / GDM.size();
            std = sqrt(sum2 / GDM.size() - mean * mean);
            MS.at<double>(i, 0) = mean;
            MS.at<double>(i + 36, 0) = std;
        }
        // normalize mean and std vector, respcectively
        MS.rowRange(0, 36) = MS.rowRange(0, 36) / cv::norm(MS.rowRange(0, 36));
        MS.rowRange(36, 72) = MS.rowRange(36, 72) / cv::norm(MS.rowRange(36, 72));
        for (int i = 0; i < MS.rows; ++i) {
            if (MS.at<double>(i, 0) > 0.4)
                MS.at<double>(i, 0) = 0.4;
        }
        MS = MS / cv::norm(MS);
        lf.mDescriptor = MS;
        return 1;
    }

    double LineExtract::ptLineDist(Vector2d pt, Vector3d line) {
        return std::fabs(LineFeature::getHomoPoint(pt).dot(line) / line.head<2>().norm());
    }

    double LineExtract::LinesDist(const LineFeature &lf1, const LineFeature &lf2) {
        return 0.25 * (ptLineDist(lf1.mStartPt, lf2.mLineParam) +
                       ptLineDist(lf1.mEndPt, lf2.mLineParam) +
                       ptLineDist(lf2.mStartPt, lf1.mLineParam) +
                       ptLineDist(lf2.mEndPt, lf1.mLineParam)
        );
    }

    double projectPt2Line(Vector2d pt, Vector2d lineEnd1, Vector2d lineEnd2) {
        Vector2d ptToEnd1 = pt - lineEnd1;
        Vector2d ptToEnd2 = pt - lineEnd2;
        return ptToEnd1.dot(ptToEnd2) / ptToEnd1.norm() / ptToEnd2.norm();
    }

    double lineSegOverlap(const LineFeature &lf1, const LineFeature &lf2) {
        if (lf1.getLineLen() < lf2.getLineLen()) {
            double lambda_start = projectPt2Line(lf1.mStartPt, lf2.mStartPt, lf2.mEndPt);
            double lambda_end = projectPt2Line(lf1.mEndPt, lf2.mStartPt, lf2.mEndPt);

            if ((lambda_start < 0 && lambda_end < 0) || (lambda_start > 1 && lambda_end > 1)) {
                return -1;
            } else {
                return std::fabs(lambda_start - lambda_end) * lf1.getLineLen();
            }

        } else {
            double lambda_start = projectPt2Line(lf2.mStartPt, lf1.mStartPt, lf1.mEndPt);
            double lambda_end = projectPt2Line(lf2.mEndPt, lf1.mStartPt, lf1.mEndPt);

            if ((lambda_start < 0 && lambda_end < 0) || (lambda_start > 1 && lambda_end > 1)) {
                return -1;
            } else {
                return std::fabs(lambda_start - lambda_end) * lf2.getLineLen();
            }
        }
    }

    void LineExtract::setMatchParam() {
        lineDistThresh = 80;
        lineAngleThresh = 25 * CV_PI / 180.0;
        desDiffThresh = 0.7;
        lineOverlapThresh = -1;
        ratioDist1st2nd = 0.7;
    }


    void LineExtract::setTrackParam() {
        lineDistThresh = 25; // pixel
        lineAngleThresh = 25 * CV_PI / 180; // 30 degree
        desDiffThresh = 0.85;
        lineOverlapThresh = 3; // pixels
        ratioDist1st2nd = 0.7;
    }

    void LineExtract::setFastTrackParam() {

        lineDistThresh = 45;
        lineAngleThresh = 30 * CV_PI / 180;
        desDiffThresh = 0.85;
        lineOverlapThresh = -1;
        ratioDist1st2nd = 0.7;
    }

    void LineExtract::setDarkLightingParam() {

        lineDistThresh = 25; // pixel
        lineAngleThresh = 10 * CV_PI / 180;
        desDiffThresh = 1.5;
        lineDistThresh = 20;
        ratioDist1st2nd = 0.85;
    }

    void LineExtract::MatchLine(const vector<LineFeature> &vlf1, const vector<LineFeature> &vlf2,
                                vector<vector<int>> &matches) {
        setMatchParam();
        line_match(vlf1, vlf2, matches);
    }

    void LineExtract::line_match(const vector<LineFeature> &vlf1, const vector<LineFeature> &vlf2,
                                 vector<vector<int>> &matches) {

        cv::Mat desDiff = cv::Mat::zeros(vlf1.size(), vlf2.size(), CV_64F) + 100;

        for (int i = 0; i < vlf1.size(); ++i) {
            for (int j = 0; j < vlf2.size(); ++j) {
                if ((vlf1[i].mDirection.dot(vlf2[j].mDirection) > std::cos(lineAngleThresh)) &&
                    (LinesDist(vlf1[i], vlf2[j]) < lineDistThresh) &&
                    (lineSegOverlap(vlf1[i], vlf2[j]) > lineOverlapThresh)) {
                    desDiff.at<double>(i, j) = cv::norm(vlf1[i].mDescriptor, vlf2[j].mDescriptor);
                }
            }
        }
        for (int k = 0; k < desDiff.rows; ++k) {
            vector<int> onePairIdx;
            double minXVal;
            cv::Point minXPos;
            cv::minMaxLoc(desDiff.row(k), &minXVal, NULL, &minXPos, NULL);
            if (minXVal < desDiffThresh) {
                double minYVal;
                cv::Point minYPos;
                cv::minMaxLoc(desDiff.col(minXPos.x), &minYVal, NULL, &minXPos, NULL);
                if (k == minYPos.y) {
                    double rowmin2 = 100, colsmin2 = 100;
                    for (int i = 0; i < desDiff.cols; ++i) {
                        if (i == minXPos.x) continue;
                        if (rowmin2 > desDiff.at<double>(k, i)) {
                            rowmin2 = desDiff.at<double>(k, i);
                        }
                    }
                    for (int j = 0; j < desDiff.rows; ++j) {
                        if (j == minYPos.y) continue;
                        if (colsmin2 > desDiff.at<double>(j, minXPos.x)) {
                            colsmin2 = desDiff.at<double>(j, minXPos.x);
                        }
                    }

                    if (rowmin2 * ratioDist1st2nd > minXVal && colsmin2 * ratioDist1st2nd > minXVal) {
                        onePairIdx.push_back(k);
                        onePairIdx.push_back(minXPos.x);
                        matches.push_back(onePairIdx);

                    }

                }
            }
        }
    }


    void LineExtract::TrackLine(const vector<LineFeature> &vlf1, const vector<LineFeature> &vlf2,
                                vector<vector<int>> &matches) {

        if (fast_motion)
            setFastTrackParam();
        if (dark_lighting)
            setDarkLightingParam();
        line_match(vlf1, vlf2, matches);
    }

    void
    LineExtract::GetTransformPtsLineRansac(const vector<vector<int>> lineMatches, vector<vector<int>> &outMatches) {
        int nln = lineMatches.size();

    }

    /**
     * input needs at least 2 correspondences of non-parallel lines
     * the resulting R and t works as below: x'=Rx+t for point pair(x,x');
     * @param vLineA
     * @param vLineB
     * @param R
     * @param t
     */
    void
    LineExtract::ComputeRelativeMotion_svd(vector<Line3d> vLineA, vector<Line3d> vLineB, Matrix3d &R, Vector3d &t) {
        if (vLineA.size() < 2) {
            cerr << "Error in computeRelativeMotion_svd: input needs at least 2 pairs!\n";
            return ;
        }
        // convert to the representation of Zhang's paper
        for (int i = 0; i < vLineA.size(); ++i) {
            Vector3d l, m;
            if (vLineA[i].u.norm() < 0.9) {
                l = vLineA[i].EndB - vLineA[i].EndA;
                m = (vLineA[i].EndA + vLineA[i].EndB) * 0.5;
                vLineA[i].u = l / l.norm();
                vLineA[i].d = vLineA[i].u.cross(m);
                //	cout<<"in computeRelativeMotion_svd compute \n";
            }
            if (vLineB[i].u.norm() < 0.9) {
                l = vLineB[i].EndB - vLineB[i].EndA;
                m = (vLineB[i].EndA + vLineB[i].EndB) * 0.5;
                vLineB[i].u = l * (1 / l.norm());
                vLineB[i].d = vLineB[i].u.cross(m);
            }
        }

        Matrix4d A = Matrix4d::Zero();
        for (int i = 0; i < vLineA.size(); ++i) {
            Matrix4d Ai = Matrix4d::Zero();
            Ai.block<1,3>(0,1) = vLineA[i].u - vLineB[i].u;
            Ai.block<3,1>(1,0) = vLineB[i].u - vLineA[i].u;

            Ai.bottomRightCorner<3, 3>(1, 1) = SO3d::hat((vLineA[i].u + vLineB[i].u)).matrix();
            A = A + Ai.transpose() * Ai;
        }
        Eigen::JacobiSVD<Matrix4d> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullV);

        Vector4d q = svd.matrixU().col(3);
        R = Eigen::Quaterniond(q).matrix();

        Matrix3d uu = Matrix3d::Zero();
        Vector3d udr = Vector3d::Zero();
        for (int i = 0; i < vLineA.size(); ++i) {
            uu = uu + SO3d::hat(vLineB[i].u) * SO3d::hat(vLineB[i].u).matrix().transpose();
            udr = udr + SO3d::hat(vLineB[i].u).transpose() * (vLineB[i].d - R * vLineA[i].d);
        }
        t = uu.inverse() * udr;
    }

    void LineExtract::drawLine(const cv::Mat &src, cv::Mat& output, std::vector<LineFeature>&LineFeatures){
        output = src.clone();

        for (auto line : LineFeatures) {
            cv::Point stt(line.mStartPt(0),line.mStartPt(1));
            cv::Point end(line.mEndPt(0), line.mEndPt(1));
            cv::line(output,stt, end,Scalar(0,0,255),1,8,0);
        }
    }
}