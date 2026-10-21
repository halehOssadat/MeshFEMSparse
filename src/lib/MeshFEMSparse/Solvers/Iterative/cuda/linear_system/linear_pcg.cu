#include <linear_system/linear_pcg.h>
#include <linear_system/global_linear_system.h>
#include <cuda_tool/linear_reduction.h>
#include <cuda_device/builtin.h>
#include <uipc/common/log.h>
#include <linear_system/synced_timer.h>
namespace uipc::backend::cuda
{
namespace
{
    __global__ void update_xr_kernel(Float                              alpha,
                                     cuda_tool::DenseVectorView<Float>  x,
                                     cuda_tool::CDenseVectorView<Float> p,
                                     cuda_tool::DenseVectorView<Float>  r,
                                     cuda_tool::CDenseVectorView<Float> Ap,
                                     int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        x(i) += alpha * p(i);
        r(i) -= alpha * Ap(i);
    }

    __global__ void update_p_kernel(cuda_tool::DenseVectorView<Float>  p,
                                    cuda_tool::CDenseVectorView<Float> z,
                                    Float                              beta,
                                    int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        p(i) = z(i) + beta * p(i);
    }
}  // namespace

void LinearPCG::do_build(BuildInfo& info)
{
    // TODO: get info from the scene, now we just use the default value
    max_iter_ratio = 2;

    // libuipc: global_tol_rate = config "linear_system/tol_rate" (default 1e-3)
    global_tol_rate = 1e-3;

    logger::info("LinearPCG: max_iter_ratio = {}, tol_rate = {}",
                 max_iter_ratio,
                 global_tol_rate);
}

void LinearPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    x.buffer_view().fill(0);

    auto N = x.size();
    if(z.capacity() < N)
    {
        auto M = reserve_ratio * N;
        z.reserve(M);
        p.reserve(M);
        r.reserve(M);
        Ap.reserve(M);
    }

    z.resize(N);
    p.resize(N);
    r.resize(N);
    Ap.resize(N);
    d_converged_false = 0;

    r0 = r;

    auto max_iter = static_cast<SizeT>(max_iter_ratio * static_cast<Float>(b.size()));
    max_iter  = std::max(max_iter, SizeT{1});
    auto iter = pcg(x, b, max_iter);

    if(iter >= max_iter)
        logger::warn(
            "LinearPCG: reached max_iter = {} (no early convergence); "
            "check preconditioner or linear_system/tol_rate.",
            max_iter);

    info.iter_count(iter);
}

void LinearPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "PCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto norm_r = ctx().norm(r.cview());
        auto norm_z = ctx().norm(z.cview());
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint   = (r_ok && z_bad) ?
                          "preconditioner failed, likely due to inverse matrix calculation failure" :
                          "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "PCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    k,
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void update_xr(Float                              alpha,
               cuda_tool::DenseVectorView<Float>  x,
               cuda_tool::CDenseVectorView<Float> p,
               cuda_tool::DenseVectorView<Float>  r,
               cuda_tool::CDenseVectorView<Float> Ap)
{
    // Fused update of x and r for better performance
    int n = r.size();
    if(n > 0)
    {
        update_xr_kernel<<<cuda_tool::best_grid_dim(n, update_xr_kernel), cuda_tool::best_block_dim(update_xr_kernel), 0, nullptr>>>(
            alpha, x.viewer(), p.cviewer(), r.viewer(), Ap.cviewer(), n);
    }
}

void update_p(cuda_tool::DenseVectorView<Float> p, cuda_tool::CDenseVectorView<Float> z, Float beta)
{
    // Simple axpby
    int n = p.size();
    if(n > 0)
    {
        update_p_kernel<<<cuda_tool::best_grid_dim(n, update_p_kernel), cuda_tool::best_block_dim(update_p_kernel), 0, nullptr>>>(
            p.viewer(), z.cviewer(), beta, n);
    }
}

SizeT LinearPCG::pcg(cuda_tool::DenseVectorView<Float>  x,
                     cuda_tool::CDenseVectorView<Float> b,
                     SizeT                              max_iter)
{
    SyncedTimer pcg_timer("PCG");

    SizeT k = 0;
    // r = b - A * x
    {
        // r = b;
        r.buffer_view().copy_from(b.buffer_view());

        // x == 0, so we don't need to do the following
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

    Float alpha, beta, rz, abs_rz0;

    // z = P * r (apply preconditioner)
    {
        SyncedTimer timer("Apply Preconditioner");
        apply_preconditioner(z, r, d_converged_false.view());
    }

    // p = z
    p = z;

    // init rz
    // rz = r^T * z
    rz = ctx().dot(r.cview(), z.cview());
    check_init_rz_nan_inf(rz);

    abs_rz0 = std::abs(rz);

    // check convergence
    if(accuracy_statisfied(r) && abs_rz0 == Float{0.0})
        return 0;

    for(k = 1; k < max_iter; ++k)
    {
        {
            SyncedTimer timer("SpMV");
            spmv(p.cview(), Ap.view());
        }

        // alpha = rz / p^T * Ap
        alpha = rz / ctx().dot(p.cview(), Ap.cview());

        // x = x + alpha * p
        // r = r - alpha * Ap
        update_xr(alpha, x, p.cview(), r.view(), Ap.cview());

        // z = P * r (apply preconditioner)
        {
            SyncedTimer timer("Apply Preconditioner");
            apply_preconditioner(z, r, d_converged_false.view());
        }

        // rz_new = r^T * z
        Float rz_new = ctx().dot(r.cview(), z.cview());
        check_iter_rz_nan_inf(rz_new, k);

        // check convergence
        if(accuracy_statisfied(r) && std::abs(rz_new) <= global_tol_rate * abs_rz0)
            break;

        // beta = rz_new / rz
        beta = rz_new / rz;

        // p = z + beta * p
        update_p(p.view(), z.cview(), beta);

        rz = rz_new;
    }

    return k;
}
}  // namespace uipc::backend::cuda
