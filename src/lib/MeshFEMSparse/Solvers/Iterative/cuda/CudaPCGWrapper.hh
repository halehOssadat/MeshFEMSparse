#ifndef CUDAPCGWRAPPER_HH
#define CUDAPCGWRAPPER_HH

#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <string>

namespace MeshFEM {

struct BlockCSCHessianBase;

// MeshFEM front end for libuipc's CUDA LinearFusedPCG / LinearPCG (block-Jacobi preconditioner).
// Plays the role of libuipc's SimEngine: owns the GlobalLinearSystem and the solver.
class CudaPCGWrapper {
public:
    // `solver`: "fused_pcg" (libuipc default) or "linear_pcg" (libuipc config "linear_system/solver").
    explicit CudaPCGWrapper(const std::string &solver = "fused_pcg");
    ~CudaPCGWrapper();

    const std::string &solver() const;

    // Convert H to libuipc's BCOO on the GPU and assemble the preconditioner.
    void compute(const BlockCSCHessianBase &H);

    // Solve H x = b with libuipc's fused_pcg() or pcg().
    Eigen::VectorXd solve(const Eigen::VectorXd &b);

    size_t iterCount() const;              // PCG iterations of the last solve

    double tolRate() const;                // global_tol_rate of the selected solver
    void   setTolRate(double tolRate);
    double maxIterRatio() const;           // max_iter_ratio of the selected solver
    void   setMaxIterRatio(double ratio);

private:
    // CUDA part of compute() (defined in CudaPCGWrapper.cu).
    void m_buildLinearSystem(int blockRows, int blockCols, size_t tripletCount,
                             const int *rowIndices, const int *colIndices,
                             const Eigen::Matrix3d *values);

    struct Impl;                           // GlobalLinearSystem + the selected solver
    std::unique_ptr<Impl> m_impl;
};

} // namespace MeshFEM

#endif /* end of include guard: CUDAPCGWRAPPER_HH */
