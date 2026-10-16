#include "CudaPCGWrapper.hh"
#include <linear_system/global_linear_system.h>
#include <linear_system/linear_pcg.h>
#include <stdexcept>

namespace MeshFEM {

using uipc::backend::cuda::GlobalLinearSystem;
using uipc::backend::cuda::LinearPCG;

struct CudaPCGWrapper::Impl {
    GlobalLinearSystem system;
    LinearPCG          solver{system};
};

CudaPCGWrapper::CudaPCGWrapper()
    : m_impl(std::make_unique<Impl>())
{
    // libuipc: the SimEngine builds each system once
    m_impl->solver.build();
}

CudaPCGWrapper::~CudaPCGWrapper() = default;

void CudaPCGWrapper::m_buildLinearSystem(int blockRows, int blockCols, size_t tripletCount,
                                         const int *rowIndices, const int *colIndices,
                                         const Eigen::Matrix3d *values) {
    m_impl->system.m_impl.build_linear_system(blockRows, blockCols, tripletCount,
                                              rowIndices, colIndices, values);
}

Eigen::VectorXd CudaPCGWrapper::solve(const Eigen::VectorXd &b) {
    auto &sys = m_impl->system.m_impl;
    const size_t n = size_t(sys.bcoo_A.rows()) * 3;
    if (n == 0)                 throw std::runtime_error("CudaPCGWrapper::solve: call compute() first");
    if (size_t(b.size()) != n)  throw std::runtime_error("CudaPCGWrapper::solve: b has the wrong size");

    sys.b.resize(n);
    sys.b.buffer_view().copy_from(b.data());
    sys.x.resize(n);

    sys.solve_linear_system();   // -> LinearPCG::do_solve -> pcg()

    Eigen::VectorXd x(n);
    sys.x.copy_to(x.data());
    return x;
}

size_t CudaPCGWrapper::iterCount()    const { return m_impl->system.m_impl.last_solve_iterations; }
double CudaPCGWrapper::tolRate()      const { return m_impl->solver.global_tol_rate; }
void   CudaPCGWrapper::setTolRate(double tolRate) { m_impl->solver.global_tol_rate = tolRate; }
double CudaPCGWrapper::maxIterRatio() const { return m_impl->solver.max_iter_ratio; }
void   CudaPCGWrapper::setMaxIterRatio(double ratio) { m_impl->solver.max_iter_ratio = ratio; }

} // namespace MeshFEM
