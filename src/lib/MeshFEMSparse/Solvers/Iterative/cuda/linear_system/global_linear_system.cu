#include <linear_system/global_linear_system.h>
#include <linear_system/iterative_solver.h>
#include <uipc/common/log.h>
#include <linear_system/synced_timer.h>

namespace uipc::backend::cuda
{
void GlobalLinearSystem::add_solver(IterativeSolver* solver)
{
    UIPC_ASSERT(solver != nullptr, "The solver should not be nullptr.");
    m_impl.iterative_solver = solver;
}

void GlobalLinearSystem::Impl::build_linear_system(int              block_rows,
                                                   int              block_cols,
                                                   SizeT            triplet_count,
                                                   const int*       row_indices,
                                                   const int*       col_indices,
                                                   const Matrix3x3* values)
{
    // Copy to device (as in libuipc's import_matrix_market)
    triplet_A.reshape(block_rows, block_cols);
    triplet_A.resize_triplets(triplet_count);
    triplet_A.row_indices().copy_from(row_indices);
    triplet_A.col_indices().copy_from(col_indices);
    triplet_A.values().copy_from(values);

    {
        SyncedTimer t("Convert To BCOO");
        converter.ge2sym(triplet_A);
        converter.convert(triplet_A, bcoo_A);
        // upload the nnz count for graph-stable SpMV launches (async on the
        // default stream; drained before any solve reads it)
        triplet_count_dev = (IndexT)bcoo_A.triplet_count();
    }

    {
        SyncedTimer t("Assemble Preconditioner");
        _assemble_preconditioner();
    }
}

void GlobalLinearSystem::Impl::_assemble_preconditioner()
{
    // libuipc loops over the local preconditioners with each subsystem's dof
    // range; here the whole system is one FEM subsystem: offset 0, all dofs.
    fem_diag_preconditioner.do_assemble(bcoo_A.cview(), 0, SizeT(bcoo_A.rows()) * 3);
}

void GlobalLinearSystem::Impl::solve_linear_system()
{
    SyncedTimer timer("Solve Linear System");
    if(iterative_solver)
    {
        SolvingInfo info{this};
        info.m_b = b.cview();
        info.m_x = x.view();
        iterative_solver->solve(info);
        last_solve_iterations = info.m_iter_count;
        logger::info("Iterative linear solver iteration count: {}", info.m_iter_count);
    }
}

void GlobalLinearSystem::Impl::apply_preconditioner(cuda_tool::DenseVectorView<Float> z,
                                                    cuda_tool::CDenseVectorView<Float> r,
                                                    cuda_tool::CVarView<IndexT> converged,
                                                    cudaStream_t stream)
{
    fem_diag_preconditioner.do_apply(r, z, converged, stream);
}

void GlobalLinearSystem::Impl::spmv(Float                              a,
                                    cuda_tool::CDenseVectorView<Float> x,
                                    Float                              b,
                                    cuda_tool::DenseVectorView<Float>  y)
{
    spmver.rbk_sym_spmv(a, bcoo_A.cview(), x, b, y);

    // Just some debug options
    //  * spmver.sym_spmv(a, bcoo_A.cview(), x, b, y);      // Slightly slower
    //  * spmver.cpu_sym_spmv(a, bcoo_A.cview(), x, b, y);  // Much slower
}

void GlobalLinearSystem::Impl::spmv_dot(cuda_tool::CDenseVectorView<Float> x,
                                        cuda_tool::DenseVectorView<Float>  y,
                                        cuda_tool::VarView<Float> d_dot,
                                        cudaStream_t              stream)
{
    spmver.rbk_sym_spmv_dot(1.0,
                            bcoo_A.cview(),
                            x,
                            0.0,
                            y,
                            d_dot,
                            triplet_count_dev.cviewer(),
                            bcoo_A.triplet_capacity(),
                            stream);
}

bool GlobalLinearSystem::Impl::accuracy_statisfied(cuda_tool::DenseVectorView<Float> r)
{
    // libuipc asks each diag subsystem; the FEM subsystem always answers true.
    return true;
}
}  // namespace uipc::backend::cuda
