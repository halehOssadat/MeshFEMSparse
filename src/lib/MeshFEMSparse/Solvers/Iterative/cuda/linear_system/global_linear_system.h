#pragma once
#include <array>
#include <cuda_tool/cuda_tool.h>
#include <algorithm/matrix_converter.h>
#include <linear_system/spmv.h>
#include <finite_element/fem_diag_preconditioner.h>

namespace MeshFEM
{
class CudaPCGWrapper;
}

namespace uipc::backend::cuda
{
class IterativeSolver;

// MeshFEM stand-in for libuipc's GlobalLinearSystem: keeps only the linear
// algebra (matrix, vectors, SpMV) used by the iterative solvers.
class GlobalLinearSystem
{
  public:
    using DenseVectorView  = cuda_tool::DenseVectorView<Float>;
    using CDenseVectorView = cuda_tool::CDenseVectorView<Float>;

    class Impl;

    class SolvingInfo
    {
      public:
        SolvingInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        DenseVectorView  x() { return m_x; }
        CDenseVectorView b() { return m_b; }
        void iter_count(SizeT iter_count) { m_iter_count = iter_count; }

      private:
        friend class Impl;
        DenseVectorView  m_x;
        CDenseVectorView m_b;
        SizeT            m_iter_count = 0;
        Impl*            m_impl       = nullptr;
    };

    class Impl
    {
      public:
        // Replaces libuipc's _assemble_linear_system(): upload host 3x3 block
        // triplets, then (as in libuipc) ge2sym -> convert -> triplet_count_dev.
        void build_linear_system(int              block_rows,
                                 int              block_cols,
                                 SizeT            triplet_count,
                                 const int*       row_indices,
                                 const int*       col_indices,
                                 const Matrix3x3* values);
        void _assemble_preconditioner();
        void solve_linear_system();

        void apply_preconditioner(cuda_tool::DenseVectorView<Float>  z,
                                  cuda_tool::CDenseVectorView<Float> r,
                                  cuda_tool::CVarView<IndexT>        converged,
                                  cudaStream_t stream = nullptr);

        void spmv(Float                              a,
                  cuda_tool::CDenseVectorView<Float> x,
                  Float                              b,
                  cuda_tool::DenseVectorView<Float>  y);
        void spmv_dot(cuda_tool::CDenseVectorView<Float> x,
                      cuda_tool::DenseVectorView<Float>  y,
                      cuda_tool::VarView<Float>          d_dot,
                      cudaStream_t                       stream = nullptr);

        // data pointers of the assembled matrix buffers (row/col/value),
        // for the FusedPCG CUDA-graph validity key: a reassembly that
        // reallocates any of them invalidates the captured graph
        std::array<const void*, 3> matrix_data_ptrs() const
        {
            auto v = bcoo_A.cview();
            return {v.row_indices().data(), v.col_indices().data(), v.values().data()};
        }

        bool accuracy_statisfied(cuda_tool::DenseVectorView<Float> r);

        // Linear System
        cuda_tool::LinearSystemContext           ctx;
        cuda_tool::DeviceDenseVector<Float>      x;
        cuda_tool::DeviceDenseVector<Float>      b;
        cuda_tool::DeviceTripletMatrix<Float, 3> triplet_A;
        cuda_tool::DeviceBCOOMatrix<Float, 3>    bcoo_A;
        // device-side copy of bcoo_A's triplet count (for graph-stable SpMV)
        cuda_tool::DeviceVar<IndexT> triplet_count_dev;

        Spmv                      spmver;
        MatrixConverter<Float, 3> converter;

        // libuipc: local_preconditioners (one FEMDiagPreconditioner for the whole system)
        FEMDiagPreconditioner fem_diag_preconditioner;

        // libuipc: SimSystemSlot<IterativeSolver>
        IterativeSolver* iterative_solver = nullptr;

        SizeT last_solve_iterations = 0;
    };

  private:
    friend class IterativeSolver;
    friend class ::MeshFEM::CudaPCGWrapper;  // libuipc: friend class SimEngine;

    void add_solver(IterativeSolver* solver);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
