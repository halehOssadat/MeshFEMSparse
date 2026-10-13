////////////////////////////////////////////////////////////////////////////////
// PCG.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Preconditioned conjugate gradient for a symmetric operator and a symmetric
// positive definite preconditioner, both given through the abstract
// interfaces in LinearOperator.hh.
//
// On negative (or zero) curvature p^T A p <= 0 the iteration stops and
// returns its current iterate (truncated Newton-CG behavior; on the first
// iteration the preconditioned right-hand side is returned) unless
// `throwOnNegativeCurvature` is set.
*///////////////////////////////////////////////////////////////////////////////
#ifndef ITERATIVE_PCG_HH
#define ITERATIVE_PCG_HH

#include "LinearOperator.hh"
#include "Preconditioner.hh"

#include <Eigen/Dense>

namespace MeshFEM {

struct MESHFEM_EXPORT PCG {
    struct Stats {
        size_t iterations = 0;
        Real relResidual = 0;
        bool converged = false;
        bool negativeCurvature = false;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Options
    ////////////////////////////////////////////////////////////////////////////
    Real   relTol   = 1e-6;  // Relative residual ||b - Ax|| / ||b|| at which PCG stops.
    size_t maxIters = 10000;
    bool throwOnNegativeCurvature = false;
    bool throwOnNonConvergence    = false;

    // Solve A x = b starting from x = 0.
    Stats solve(const LinearOperator &A, const Preconditioner &M, const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    Eigen::VectorXd solve(const LinearOperator &A, const Preconditioner &M, const Eigen::VectorXd &b) const {
        Eigen::VectorXd x;
        m_lastStats = solve(A, M, b, x);
        return x;
    }

    // Statistics of the most recent call to the vector-returning `solve`.
    const Stats &lastStats() const { return m_lastStats; }

private:
    mutable Stats m_lastStats;
};

} // namespace MeshFEM

#endif /* end of include guard: ITERATIVE_PCG_HH */
