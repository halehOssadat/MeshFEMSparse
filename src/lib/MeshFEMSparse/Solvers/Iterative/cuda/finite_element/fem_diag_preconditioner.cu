#include <type_define.h>
#include <Eigen/Dense>
#include <finite_element/fem_diag_preconditioner.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace eigen = cuda_tool::eigen;

    __global__ void FEMDiagPreconditioner_do_assemble_kernel(
        cuda_tool::CBCOOMatrixView<Float, 3> triplet,
        cuda_tool::BufferView<Matrix3x3>     diag_inv,
        SizeT                                fem_segment_offset,
        SizeT                                fem_segment_count,
        int                                  n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        auto&& [g_i, g_j, H3x3] = triplet(I);

        IndexT i = g_i - fem_segment_offset;
        IndexT j = g_j - fem_segment_offset;

        if(i >= fem_segment_count || j >= fem_segment_count)
        {
            return;
        }

        if(i == j)
        {
            diag_inv(i) = eigen::inverse(H3x3);
        }
    }

    __global__ void FEMDiagPreconditioner_do_apply_kernel(
        cuda_tool::CDenseVectorView<Float> r,
        cuda_tool::DenseVectorView<Float>  z,
        cuda_tool::CDense<IndexT>          converged,
        cuda_tool::BufferView<Matrix3x3>   diag_inv,
        int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        if(*converged != 0)
            return;
        z.segment<3>(i * 3).as_eigen() = diag_inv(i) * r.segment<3>(i * 3).as_eigen();
    }
}  // namespace

void FEMDiagPreconditioner::do_assemble(cuda_tool::CBCOOMatrixView<Float, 3> A,
                                        SizeT dof_offset,
                                        SizeT dof_count)
{
    // libuipc: diag_inv.resize(finite_element_method->xs().size());
    diag_inv.resize(dof_count / 3);

    // 1) collect diagonal blocks
    auto k = FEMDiagPreconditioner_do_assemble_kernel;
    int  n = (int)A.triplet_count();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            A, diag_inv.view(), dof_offset / 3, dof_count / 3, n);
    }
}

void FEMDiagPreconditioner::do_apply(cuda_tool::CDenseVectorView<Float> r,
                                     cuda_tool::DenseVectorView<Float>  z,
                                     cuda_tool::CVarView<IndexT>        converged,
                                     cudaStream_t                       stream)
{
    auto k = FEMDiagPreconditioner_do_apply_kernel;
    int  n = (int)diag_inv.size();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, stream>>>(
            r, z, converged.cviewer(), diag_inv.view(), n);
    }
}
}  // namespace uipc::backend::cuda
