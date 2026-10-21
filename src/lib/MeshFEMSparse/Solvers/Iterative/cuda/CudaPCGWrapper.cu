#include "CudaPCGWrapper.hh"
#include <linear_system/global_linear_system.h>
#include <linear_system/linear_pcg.h>
#include <linear_system/linear_fused_pcg.h>
#include <stdexcept>

namespace MeshFEM {

using uipc::backend::cuda::GlobalLinearSystem;
using uipc::backend::cuda::LinearPCG;
using uipc::backend::cuda::LinearFusedPCG;

// Exactly one solver is created, as libuipc does for config "linear_system/solver".
struct CudaPCGWrapper::Impl {
    GlobalLinearSystem              system;
    std::unique_ptr<LinearPCG>      linear_pcg;
    std::unique_ptr<LinearFusedPCG> fused_pcg;
    std::string                     solver;
};

CudaPCGWrapper::CudaPCGWrapper(const std::string &solver)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->solver = solver;
    // libuipc: the SimEngine builds each system once
    if (solver == "fused_pcg") {
        m_impl->fused_pcg = std::make_unique<LinearFusedPCG>(m_impl->system);
        m_impl->fused_pcg->build();
    }
    else if (solver == "linear_pcg") {
        m_impl->linear_pcg = std::make_unique<LinearPCG>(m_impl->system);
        m_impl->linear_pcg->build();
    }
    else throw std::runtime_error("CudaPCGWrapper: unknown solver \"" + solver + "\" (expected \"fused_pcg\" or \"linear_pcg\")");
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

    sys.solve_linear_system();   // -> LinearFusedPCG::fused_pcg() or LinearPCG::pcg()

    Eigen::VectorXd x(n);
    sys.x.copy_to(x.data());
    return x;
}

const std::string &CudaPCGWrapper::solver() const { return m_impl->solver; }
size_t CudaPCGWrapper::iterCount() const { return m_impl->system.m_impl.last_solve_iterations; }

double CudaPCGWrapper::tolRate() const {
    return m_impl->fused_pcg ? m_impl->fused_pcg->global_tol_rate : m_impl->linear_pcg->global_tol_rate;
}
void CudaPCGWrapper::setTolRate(double tolRate) {
    if (m_impl->fused_pcg) m_impl->fused_pcg->global_tol_rate = tolRate;
    else                   m_impl->linear_pcg->global_tol_rate = tolRate;
}
double CudaPCGWrapper::maxIterRatio() const {
    return m_impl->fused_pcg ? m_impl->fused_pcg->max_iter_ratio : m_impl->linear_pcg->max_iter_ratio;
}
void CudaPCGWrapper::setMaxIterRatio(double ratio) {
    if (m_impl->fused_pcg) m_impl->fused_pcg->max_iter_ratio = ratio;
    else                   m_impl->linear_pcg->max_iter_ratio = ratio;
}

} // namespace MeshFEM
