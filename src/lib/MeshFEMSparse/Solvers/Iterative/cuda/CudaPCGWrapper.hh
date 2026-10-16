#ifndef CUDAPCGWRAPPER_HH
#define CUDAPCGWRAPPER_HH

#include <Eigen/Dense>
#include <cstddef>
#include <memory>

namespace MeshFEM {

struct BlockCSCHessianBase;

// MeshFEM front end for libuipc's CUDA LinearPCG (block-Jacobi preconditioner).
// Plays the role of libuipc's SimEngine: owns the GlobalLinearSystem and the solver.
class CudaPCGWrapper {
public:
    CudaPCGWrapper();
    ~CudaPCGWrapper();

    // Convert H to libuipc's BCOO on the GPU and assemble the preconditioner.
    void compute(const BlockCSCHessianBase &H);

    // Solve H x = b with libuipc's pcg().
    Eigen::VectorXd solve(const Eigen::VectorXd &b);

    size_t iterCount() const;              // PCG iterations of the last solve

    double tolRate() const;                // LinearPCG::global_tol_rate
    void   setTolRate(double tolRate);
    double maxIterRatio() const;           // LinearPCG::max_iter_ratio
    void   setMaxIterRatio(double ratio);

private:
    // CUDA part of compute() (defined in CudaPCGWrapper.cu).
    void m_buildLinearSystem(int blockRows, int blockCols, size_t tripletCount,
                             const int *rowIndices, const int *colIndices,
                             const Eigen::Matrix3d *values);

    struct Impl;                           // GlobalLinearSystem + LinearPCG
    std::unique_ptr<Impl> m_impl;
};

} // namespace MeshFEM

#endif /* end of include guard: CUDAPCGWRAPPER_HH */
