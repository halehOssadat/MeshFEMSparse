#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
// MeshFEM version of libuipc's FEMDiagPreconditioner (block-Jacobi on the 3x3
// diagonal blocks) without the LocalPreconditioner/SimSystem plumbing: the
// values libuipc reads from the assembly/apply info objects are passed directly.
class FEMDiagPreconditioner
{
  public:
    cuda_tool::DeviceBuffer<Matrix3x3> diag_inv;

    // libuipc: info.A(), info.dof_offset(), info.dof_count()
    void do_assemble(cuda_tool::CBCOOMatrixView<Float, 3> A, SizeT dof_offset, SizeT dof_count);

    // libuipc: info.r(), info.z(), info.converged(), info.stream()
    void do_apply(cuda_tool::CDenseVectorView<Float> r,
                  cuda_tool::DenseVectorView<Float>  z,
                  cuda_tool::CVarView<IndexT>        converged,
                  cudaStream_t                       stream);
};
}  // namespace uipc::backend::cuda
