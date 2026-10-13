#include "LinearOperator.hh"

#include <algorithm>
#include <stdexcept>

namespace MeshFEM {

SymmetricMatrixOperator::SymmetricMatrixOperator(const BlockCSCHessianBase &A, const std::vector<size_t> &pinnedVars, Real sigma)
    : m_block(&A), m_size(A.numScalarCols()), m_sigma(sigma) {
    m_setPinned(pinnedVars);
}

SymmetricMatrixOperator::SymmetricMatrixOperator(const SuiteSparseMatrix &A, const std::vector<size_t> &pinnedVars, Real sigma)
    : m_scalar(&A), m_size(size_t(A.n)), m_sigma(sigma) {
    if (A.m != A.n) throw std::runtime_error("SymmetricMatrixOperator: matrix must be square");
    if (A.symmetry_mode == SuiteSparseMatrix::SymmetryMode::NONE)
        throw std::runtime_error("SymmetricMatrixOperator: matrix must be tagged symmetric (upper triangle storage)");
    m_setPinned(pinnedVars);
}

void SymmetricMatrixOperator::m_setPinned(const std::vector<size_t> &pinnedVars) {
    m_pinned = pinnedVars;
    std::sort(m_pinned.begin(), m_pinned.end());
    m_pinned.erase(std::unique(m_pinned.begin(), m_pinned.end()), m_pinned.end());
    if (!m_pinned.empty() && (m_pinned.back() >= m_size))
        throw std::runtime_error("SymmetricMatrixOperator: pinned variable index out of range");
}

void SymmetricMatrixOperator::apply(const Real *x, Real *y) const {
    const Real *xIn = x;
    if (!m_pinned.empty()) {
        m_xFree.assign(x, x + m_size);
        for (size_t i : m_pinned) m_xFree[i] = 0.0;
        xIn = m_xFree.data();
    }

    if (m_block) m_block->applyRawParallel(xIn, y);
    else         m_scalar->applyRaw(xIn, y);

    if (m_sigma != 0) {
        for (size_t i = 0; i < m_size; ++i) y[i] += m_sigma * xIn[i];
    }
    for (size_t i : m_pinned) y[i] = x[i];
}

} // namespace MeshFEM
