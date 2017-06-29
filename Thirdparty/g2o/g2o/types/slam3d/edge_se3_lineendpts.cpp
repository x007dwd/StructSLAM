//
// Created by bobin on 17-6-26.
//

#include "edge_se3_lineendpts.h"


namespace g2o {
    using namespace std;

    EdgeSE3LineEndpts::EdgeSE3LineEndpts() : BaseBinaryEdge<6, Vector6d, VertexSE3, VertexLineEndpts>() {
        information().setIdentity();
        J.fill(0);
        J.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
        cache = 0;
        offsetParam = 0;
        resizeParameters(1);
        installParameter(offsetParam, 0, 0);
    }

    bool EdgeSE3LineEndpts::resolveCaches() {
        ParameterVector pv(1);
        pv[0] = offsetParam;
        resolveCache(cache, (OptimizableGraph::Vertex *) _vertices[0], "CACHE_SE3_OFFSET", pv);
        return cache != 0;
    }

    bool EdgeSE3LineEndpts::read(std::istream &is) {
        int pid;
        is >> pid;
        setParameterId(8, pid);

        Vector6d meas;
        for (int i = 0; i < 6; ++i) {
            is >> meas[i];
        }
        setMeasurement(meas);

        if (is.bad()) {
            return false;
        }

        for (int i = 0; i < information().rows() && is.good(); ++i) {
            for (int j = i; j < information().cols() && is.good(); ++j) {
                is >> information()(i, j);
                if (i != 1)
                    information()(j, i) = information()(i, j);
            }
            if (is.bad())
                information().setIdentity();
        }
        return true;
    }

    bool EdgeSE3LineEndpts::write(std::ostream &os) {
        os << offsetParam->id() << " ";
        for (int i = 0; i < 6; ++i) {
            os << measurement()[i] << " ";

        }

        for (int i = 0; i < information().rows(); ++i) {
            for (int j = 0; j < information().cols(); ++j) {
                os << information()(i, j) << " ";
            }
        }
        return os.good();
    }

    void EdgeSE3LineEndpts::computeError() {
        VertexLineEndpts *endpts = static_cast<VertexLineEndpts *>(_vertices[1]);

        Vector3d ptAw(endpts->estimate()[0], endpts->estimate()[1], endpts->estimate()[2]);
        Vector3d ptBw(endpts->estimate()[3], endpts->estimate()[4], endpts->estimate()[5]);

        Vector3d ptA = cache->w2l() * ptAw;
        Vector3d ptB = cache->w2l() * ptBw;

        Vector3d measpt1(_measurement(0), _measurement(1), _measurement(2));
        Vector3d measpt2(_measurement(3), _measurement(4), _measurement(5));

        Eigen::Vector3d Ap1 = endpt_AffnMat.block<3, 3>(3, 3) * (ptA - measpt1);
        Eigen::Vector3d Bp1 = endpt_AffnMat.block<3, 3>(3, 3) * (ptB - measpt1);
        Eigen::Vector3d Bp1_Ap1 = Bp1 - Ap1;

        double t = -Ap1.dot(Bp1_Ap1) / (Bp1_Ap1.dot(Bp1_Ap1));
        Vector3d normalized_pt2line_vec1 = Ap1 + t * Bp1_Ap1;

        Eigen::Vector3d Ap2 = endpt_AffnMat.block<3, 3>(3, 3) * (ptA - measpt2);
        Eigen::Vector3d Bp2 = endpt_AffnMat.block<3, 3>(3, 3) * (ptB - measpt2);
        Eigen::Vector3d Bp2_Ap2 = Bp2 - Ap2;

        t = -Ap2.dot(Bp2_Ap2) / (Bp2_Ap2.dot(Bp2_Ap2));
        Vector3d normalized_pt2line_vec2 = Ap2 + t * Bp2_Ap2;

        _error.resize(6);
        _error(0) = normalized_pt2line_vec1(0);
        _error(1) = normalized_pt2line_vec1(1);
        _error(2) = normalized_pt2line_vec1(2);
        _error(3) = normalized_pt2line_vec2(0);
        _error(4) = normalized_pt2line_vec2(1);
        _error(5) = normalized_pt2line_vec2(2);

    }

    bool EdgeSE3LineEndpts::setMeasurementFromState() {
        VertexLineEndpts *lpts = static_cast<VertexLineEndpts *>(_vertices[1]);

        VertexLineEndpts *endpts = static_cast<VertexLineEndpts *>(_vertices[1]);
        Vector3d ptAw(endpts->estimate()[0], endpts->estimate()[1],
                      endpts->estimate()[2]);
        Vector3d ptBw(endpts->estimate()[3], endpts->estimate()[4],
                      endpts->estimate()[5]);
        Vector3d ptA =
                cache->w2n() * ptAw; // line endpoint tranformed to the camera frame
        Vector3d ptB = cache->w2n() * ptBw;
        _measurement.resize(6);
        _measurement(0) = ptA(0);
        _measurement(1) = ptA(1);
        _measurement(2) = ptA(2);
        _measurement(3) = ptB(0);
        _measurement(4) = ptB(1);
        _measurement(5) = ptB(2);
        return true;
    }

    void EdgeSE3LineEndpts::initialEstimate(
            const OptimizableGraph::VertexSet &from,
            OptimizableGraph::Vertex *to_)
    { // estimate 3d pt world position by cam pose and current meas pt
        (void) from;
        assert(from.size() == 1 && from.count(_vertices[0]) == 1 &&
               "Can not initialize VertexDepthCam position by VertexTrackXYZ");

        VertexSE3 *cam = dynamic_cast<VertexSE3 *>(_vertices[0]);
        VertexLineEndpts *point = dynamic_cast<VertexLineEndpts *>(_vertices[1]);
    }

#ifdef G2O_HAVE_OPENGL
    EdgeSE3LineEndptsDrawAction::EdgeSE3LineEndptsDrawAction()
        : DrawAction(typeid(EdgeSE3LineEndpts).name()) {}

    HyperGraphElementAction *EdgeSE3LineEndptsDrawAction::
    operator()(HyperGraph::HyperGraphElement *element,
               HyperGraphElementAction::Parameters *params_) {
      if (typeid(*element).name() != _typeName)
        return 0;
      refreshPropertyPtrs(params_);
      if (!_previousParams)
        return this;

      if (_show && !_show->value())
        return this;

      EdgeSE3LineEndpts *e = static_cast<EdgeSE3LineEndpts *>(element);
      VertexSE3 *fromEdge = static_cast<VertexSE3 *>(e->vertex(0));
      VertexLineEndpts *toEdge = static_cast<VertexLineEndpts *>(e->vertex(1));
      glColor3f(0.8f, 0.3f, 0.3f);
      glPushAttrib(GL_ENABLE_BIT);
      glDisable(GL_LIGHTING);
      glBegin(GL_LINES);
      glVertex3f((float)fromEdge->estimate().translation().x(),
                 (float)fromEdge->estimate().translation().y(),
                 (float)fromEdge->estimate().translation().z());
      glVertex3f((float)toEdge->estimate()(0), (float)toEdge->estimate()(1),
                 (float)toEdge->estimate()(2));
      glEnd();
      glPopAttrib();
      return this;
    }
#endif
}
