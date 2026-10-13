#include "PCG.hh"

#include <MeshFEMCore/GlobalBenchmark.hh>

#include <cmath>
#include <stdexcept>
#include <string>

namespace MeshFEM {

PCG::Stats PCG::solve(const LinearOperator &A, const Preconditioner &M, const Eigen::VectorXd &b, Eigen::VectorXd &x) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("PCG.solve");
    const size_t n = A.size();
    if ((M.size() != n) || (size_t(b.size()) != n))
        throw std::runtime_error("PCG: operator, preconditioner, and right-hand side sizes differ");

    Stats st;
    x.setZero(n);
    Eigen::VectorXd r = b, z(n), p(n), q(n);
    const Real bnorm = r.norm();
    if (bnorm == 0.0) { st.converged = true; return st; }
    st.relResidual = 1.0;

    M.apply(r.data(), z.data());
    p = z;
    Real rz = r.dot(z);
    for (size_t it = 0; it < maxIters; ++it) {
        A.apply(p.data(), q.data());
        const Real pq = p.dot(q);
        if (!std::isfinite(pq)) throw std::runtime_error("PCG: non-finite value encountered");
        if (pq <= 0) {
            st.negativeCurvature = true;
            // Truncated Newton-CG: x stays a descent direction; on the first
            // iteration fall back to the preconditioned right-hand side.
            if (it == 0) x = z;
            break;
        }
        const Real alpha = rz / pq;
        x += alpha * p;
        r -= alpha * q;
        st.iterations = it + 1;
        st.relResidual = r.norm() / bnorm;
        if (st.relResidual <= relTol) { st.converged = true; break; }

        M.apply(r.data(), z.data());
        const Real rzNew = r.dot(z);
        if (!(rzNew > 0)) break; // M is SPD, so this only happens for r = 0 (or roundoff).
        p = z + (rzNew / rz) * p;
        rz = rzNew;
    }

    if (st.negativeCurvature && throwOnNegativeCurvature)
        throw std::runtime_error("PCG: negative curvature encountered");
    if (!st.converged && !st.negativeCurvature && throwOnNonConvergence)
        throw std::runtime_error("PCG: did not converge (relative residual " + std::to_string(st.relResidual) + ")");
    return st;
}

} // namespace MeshFEM
