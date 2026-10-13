////////////////////////////////////////////////////////////////////////////////
// LinearOperator.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// `LinearOperator` (y = A x), the operator interface used by the iterative
// solvers (see PCG.hh), and `SymmetricMatrixOperator`, which applies a
// symmetric sparse matrix (upper triangle storage) without copying it.
//
// Pinned variables are handled by replacing their rows and columns with the
// identity, so a system solved with a right-hand side that is zero on the
// pinned variables yields a solution that is zero there as well.
*///////////////////////////////////////////////////////////////////////////////
#ifndef ITERATIVE_LINEAROPERATOR_HH
#define ITERATIVE_LINEAROPERATOR_HH

#include <MeshFEMCore/Types.hh>
#include <MeshFEMSparse/SparseMatrices.hh>
#include <MeshFEMSparse/BlockCSCHessian.hh>

#include <vector>

namespace MeshFEM {

struct MESHFEM_EXPORT LinearOperator {
    virtual ~LinearOperator() = default;
    virtual size_t size() const = 0;
    // y = A x (x and y have length `size()` and do not alias).
    virtual void apply(const Real *x, Real *y) const = 0;
};

// y = (A + sigma I) x on the free variables and y = x on the pinned ones.
// The referenced matrix must outlive the operator; its values may change
// between applications (the operator stores no copy).
struct MESHFEM_EXPORT SymmetricMatrixOperator final : public LinearOperator {
    // Block Hessian: applied with `BlockCSCHessianBase::applyRawParallel`.
    SymmetricMatrixOperator(const BlockCSCHessianBase &A, const std::vector<size_t> &pinnedVars = {}, Real sigma = 0);
    // Scalar matrix: must hold the upper triangle (or be fully stored with a
    // symmetric `symmetry_mode`).
    SymmetricMatrixOperator(const SuiteSparseMatrix &A, const std::vector<size_t> &pinnedVars = {}, Real sigma = 0);

    size_t size() const override { return m_size; }
    void apply(const Real *x, Real *y) const override;

    Real sigma() const { return m_sigma; }
    void setSigma(Real sigma) { m_sigma = sigma; }

private:
    const BlockCSCHessianBase *m_block  = nullptr;
    const SuiteSparseMatrix   *m_scalar = nullptr;
    size_t m_size = 0;
    Real m_sigma = 0;
    std::vector<size_t> m_pinned; // sorted, unique
    mutable std::vector<Real> m_xFree;

    void m_setPinned(const std::vector<size_t> &pinnedVars);
};

} // namespace MeshFEM

#endif /* end of include guard: ITERATIVE_LINEAROPERATOR_HH */
