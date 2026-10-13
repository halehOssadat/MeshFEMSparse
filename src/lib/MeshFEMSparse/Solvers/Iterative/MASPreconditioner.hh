////////////////////////////////////////////////////////////////////////////////
// MASPreconditioner.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Connectivity-enhanced Multilevel Additive Schwarz (MAS) preconditioner of
// StiffGIPC (Huang et al., "StiffGIPC: Advancing GPU IPC for Stiff
// Affine-Deformable Simulation", ACM TOG 44(3), 2025, Section 4).
//
//  - analyzePattern: partition the node graph into subdomains of at most
//    `BankSize` nodes (METIS k-way when available, Sec. 4.2), pad every
//    subdomain to `BankSize` slots, and build the hierarchy of super nodes
//    (connected components inside each subdomain) for the coarser levels.
//  - compute: assemble the dense Galerkin subdomain matrices R_l A R_l^T of
//    every level from per-subdomain contribution lists (the CPU analogue of
//    the warp reduction in Algorithm 1) and invert them.
//  - apply: z = sum_l R_l^T M_l^{-1} R_l r.
//
// The values are read directly from the matrix passed to `compute` (block
// Hessians included, whatever their storage layout); no copy of A is kept.
//
// A "node" is a group of `nodeBlockSize` consecutive scalar variables
// (inferred as the GCD of the block sizes of a `BlockCSCHessian`; scalar
// matrices use single-variable nodes unless `nodeBlockSize` is set).
//
// Pinned variables have their rows and columns replaced by the identity
// (matching `SymmetricMatrixOperator`), so partially pinned nodes need no
// scalar fallback. `compute` throws if a subdomain matrix is not positive
// definite (a principal submatrix of a Galerkin product, hence a certificate
// that A + sigma I is indefinite).
*///////////////////////////////////////////////////////////////////////////////
#ifndef ITERATIVE_MASPRECONDITIONER_HH
#define ITERATIVE_MASPRECONDITIONER_HH

#include "Preconditioner.hh"

#include <MeshFEMSparse/SparseMatrices.hh>
#include <MeshFEMSparse/BlockCSCHessian.hh>

#include <cstdint>
#include <vector>

namespace MeshFEM {

struct MESHFEM_EXPORT MASPreconditioner final : public Preconditioner {
    static constexpr size_t BankSize = 16; // Nodes per subdomain (paper's N).

    enum class PartitionMethod {
        Auto,   // METIS when compiled in, otherwise Greedy.
        Metis,  // Throws if METIS is unavailable.
        Greedy  // Breadth-first graph growing; no external dependency.
    };

    ////////////////////////////////////////////////////////////////////////////
    // Options (take effect at the next `analyzePattern`)
    ////////////////////////////////////////////////////////////////////////////
    size_t maxLevels = 6;     // Including the fine level; 1 gives block Jacobi on subdomains.
    size_t nodeBlockSize = 0; // 0: infer from the matrix block structure.
    PartitionMethod partitionMethod = PartitionMethod::Auto;

    static bool metisAvailable();

    ////////////////////////////////////////////////////////////////////////////
    // Setup
    ////////////////////////////////////////////////////////////////////////////
    // Matrices hold the upper triangle of a symmetric matrix. `compute` must
    // be passed a matrix of the same kind (block/scalar) and sparsity pattern
    // as the preceding `analyzePattern`.
    void analyzePattern(const BlockCSCHessianBase &A, const std::vector<size_t> &pinnedVars = {});
    void analyzePattern(const SuiteSparseMatrix   &A, const std::vector<size_t> &pinnedVars = {});

    // Build the preconditioner for A + sigma I.
    void compute(const BlockCSCHessianBase &A, Real sigma = 0);
    void compute(const SuiteSparseMatrix   &A, Real sigma = 0);

    bool isComputed() const { return m_computed; }
    void clear();

    ////////////////////////////////////////////////////////////////////////////
    // Preconditioner interface
    ////////////////////////////////////////////////////////////////////////////
    size_t size() const override { return m_numScalarVars; }
    void apply(const Real *r, Real *z) const override;

    ////////////////////////////////////////////////////////////////////////////
    // Introspection
    ////////////////////////////////////////////////////////////////////////////
    size_t numLevels() const { return m_levels.size(); }
    size_t numLevelNodes(size_t level) const { return m_levels.at(level).numNodes; }
    size_t numSubdomains(size_t level) const { return m_levels.at(level).numSubdomains; }
    size_t nodeBlockSizeUsed() const { return m_d; }
    bool usedMetis() const { return m_usedMetis; }

private:
    using Index = int64_t;

    struct Level {
        size_t numNodes = 0;        // Active nodes (level 0: slots, including padding).
        size_t numSubdomains = 0;
        size_t subdomainOffset = 0; // Index of this level's first subdomain in the global list.
        size_t vecOffset = 0;       // Offset of this level's slots in m_levelR/m_levelZ (scalar entries).
        // For levels > 0: the node at this level that each node of the previous level merges into,
        // and the inverse map in CSR form.
        std::vector<Index> parentOfPrev;
        std::vector<Index> childPtr, children;
    };

    struct Contribution {
        Index entry;   // Node block entry (I, J), I <= J
        Index diagNode; // I if I == J, otherwise -1
        uint8_t a, b;  // Local node positions in the subdomain
    };

    // Problem structure
    bool   m_blockInput = false;
    size_t m_numScalarVars = 0;
    size_t m_d = 1;                // Scalar variables per node
    size_t m_numNodes = 0;
    size_t m_patternCols = 0, m_patternNNZ = 0, m_valuesSize = 0; // For checking `compute` input
    std::vector<char> m_isPinned;  // Per scalar variable

    // Location in the input matrix's `Ax` of each scalar entry of every node
    // block entry (d x d column-major per entry; -1 for structural zeros and
    // pinned rows/columns). Diagonal blocks have both triangles filled in.
    std::vector<Index> m_entryAx;

    // MAS hierarchy
    bool m_usedMetis = false;
    std::vector<Level> m_levels;
    std::vector<Index> m_slotNode;    // Level-0 slot -> node (-1 for padding)
    std::vector<Index> m_nodeSlot;    // Node -> level-0 slot (-1 if fully pinned)
    std::vector<Index> m_contribPtr;  // Per global subdomain
    std::vector<Contribution> m_contrib;
    std::vector<size_t> m_subdomainLevel;
    size_t m_numSubdomainsTotal = 0;
    size_t m_levelVecSize = 0;

    bool m_analyzed = false, m_computed = false;
    std::vector<float> m_inverses; // Dense (BankSize d)^2 inverse per subdomain

    mutable std::vector<Real> m_levelR, m_levelZ;

    size_t m_subdomainDim() const { return BankSize * m_d; }

    // `scalarPattern`: upper-triangle scalar sparsity pattern of A;
    // `axOffsets`: location in A.Ax of each scalar entry (nullptr: identity).
    void m_analyze(const SuiteSparseMatrix &scalarPattern, const Index *axOffsets, size_t d, const std::vector<size_t> &pinnedVars);
    void m_buildHierarchy(const std::vector<Index> &bAp, const std::vector<Index> &bAi, const std::vector<char> &nodeInMAS);
    void m_assembleAndInvert(const Real *Ax, Real sigma);
};

} // namespace MeshFEM

#endif /* end of include guard: ITERATIVE_MASPRECONDITIONER_HH */
