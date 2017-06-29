//
// Created by bobin on 17-6-26.
//

#ifndef YGZ_STEREO_EDGE_SE3_LINEENDPTS_H
#define YGZ_STEREO_EDGE_SE3_LINEENDPTS_H
#include "parameter_se3_offset.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/vertex_se3.h"

#include "iostream"
#include "Eigen/src/SVD/JacobiSVD.h"
#include "edge_se3_lineendpts.h"
#include "vertex_lineendpts.h"

namespace g2o {
    using namespace std;

    class EdgeSE3LineEndpts : public BaseBinaryEdge<6, Vector6d, VertexSE3, VertexLineEndpts> {
    public:
        EdgeSE3LineEndpts();

        virtual bool read(std::istream &is);

        virtual bool write(std::ostream &os);

        void computeError();

        virtual void setMeasurement(const Vector6d &m) {
            _measurement = m;
        }

        virtual bool setMeasurementData(const double *m) {
            Map<const Vector6d> mes(m);
            _measurement = mes;
            return true;
        }

        virtual bool getMeasurementData(double *d) const {
            Map<Vector6d> v(d);
            v = _measurement;
            return true;
        }

        virtual int measurementDimension() const {
            return 6;
        }

        virtual bool setMeasurementFromState();

        virtual double initialEstimatePossible(const OptimizableGraph::VertexSet&from, OptimizableGraph::Vertex *to) {
            return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
        }

        virtual void initialEstimate(const OptimizableGraph::VertexSet&from, OptimizableGraph::Vertex *to);

        Eigen::Matrix<double, 6, 6> endptCov;
        Eigen::Matrix<double, 6, 6> endpt_AffnMat;

    private:
        Eigen::Matrix<double, 6, 12> J;
        ParameterSE3Offset *offsetParam;
        CacheSE3Offset *cache;

        virtual bool resolveCaches();
    };

#ifdef G2O_HAVE_OPENGL
    class EdgeSE3LineEndptsDrawAction:public DrawAction{
    public:
        EdgeSE3LineEndptsDrawAction();
        virtual HyperGraphElementAction*operator()(HyperGraph::HyperGraphElement *element,
        HyperGraphElementAction::Parameters *params_);
    };
#endif
}

#endif //YGZ_STEREO_EDGE_SE3_LINEENDPTS_H
