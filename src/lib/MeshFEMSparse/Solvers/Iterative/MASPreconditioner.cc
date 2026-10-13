#include "MASPreconditioner.hh"

#include <MeshFEMCore/GlobalBenchmark.hh>

#include <Eigen/Cholesky>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#if MESHFEM_WITH_SUITESPARSE_METIS
// METIS as bundled (with prefixed symbols) inside SuiteSparse >= 7's CHOLMOD.
// SuiteSparse builds it with 64-bit `idx_t`; we never pass `real_t` arrays.
extern "C" int SuiteSparse_metis_METIS_PartGraphKway(
        int64_t *nvtxs, int64_t *ncon, int64_t *xadj, int64_t *adjncy,
        int64_t *vwgt, int64_t *vsize, int64_t *adjwgt, int64_t *nparts,
        void *tpwgts, void *ubvec, int64_t *options, int64_t *objval, int64_t *part);
#endif

namespace MeshFEM {

namespace {

using Index = int64_t;

// Undirected graph in CSR form (both directions stored, no self loops).
struct Graph {
    Index numVertices = 0;
    std::vector<Index> xadj, adj;
};

// Renumber partition ids to be contiguous, dropping empty partitions.
Index compactPartitionIds(std::vector<Index> &part, Index numParts) {
    std::vector<Index> newId(numParts, -1);
    Index count = 0;
    for (Index &p : part) {
        if (newId[p] < 0) newId[p] = count++;
        p = newId[p];
    }
    return count;
}

#if MESHFEM_WITH_SUITESPARSE_METIS
// Sec. 4.2: M = ceil(V / (N - N_o)) partitions, increasing the slack N_o
// until no partition exceeds N.
bool metisPartition(const Graph &g, Index maxSize, std::vector<Index> &part, Index &numParts) {
    const Index nv = g.numVertices;
    if (g.adj.empty()) return false;

    std::vector<int64_t> xadj(g.xadj.begin(), g.xadj.end()), adj(g.adj.begin(), g.adj.end());
    std::vector<int64_t> result(nv);
    for (Index slack = 0; slack < maxSize; ++slack) {
        int64_t nvtxs = nv, ncon = 1, objval = 0;
        int64_t nparts = std::max<int64_t>(2, (nv + (maxSize - slack) - 1) / (maxSize - slack));
        int ret = SuiteSparse_metis_METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adj.data(),
                                                       nullptr, nullptr, nullptr, &nparts,
                                                       nullptr, nullptr, nullptr, &objval, result.data());
        if (ret != 1 /* METIS_OK */) return false;

        std::vector<Index> sizes(nparts, 0);
        bool valid = true;
        for (int64_t p : result) {
            if (p < 0 || p >= nparts) { valid = false; break; }
            ++sizes[p];
        }
        if (!valid) return false;
        if (*std::max_element(sizes.begin(), sizes.end()) <= maxSize) {
            part.assign(result.begin(), result.end());
            numParts = compactPartitionIds(part, nparts);
            return true;
        }
    }
    return false;
}
#endif

// Grow each partition breadth-first from the next unassigned vertex in a
// global BFS order, preferring the candidate with the most connections into
// the partition. Seeding in BFS order keeps consecutive partitions spatially
// close, which helps the coarse levels (built from consecutive subdomains).
void greedyPartition(const Graph &g, Index maxSize, std::vector<Index> &part, Index &numParts) {
    const Index nv = g.numVertices;
    std::vector<Index> order, rank(nv, -1);
    order.reserve(nv);
    for (Index s = 0; s < nv; ++s) {
        if (rank[s] >= 0) continue;
        rank[s] = Index(order.size());
        order.push_back(s);
        for (size_t head = order.size() - 1; head < order.size(); ++head) {
            Index v = order[head];
            for (Index k = g.xadj[v]; k < g.xadj[v + 1]; ++k) {
                Index u = g.adj[k];
                if (rank[u] < 0) { rank[u] = Index(order.size()); order.push_back(u); }
            }
        }
    }

    part.assign(nv, -1);
    numParts = 0;
    std::vector<Index> gain(nv, 0), candidates;
    for (Index seed : order) {
        if (part[seed] >= 0) continue;
        const Index p = numParts++;
        Index size = 0;
        auto add = [&](Index v) {
            part[v] = p;
            ++size;
            for (Index k = g.xadj[v]; k < g.xadj[v + 1]; ++k) {
                Index u = g.adj[k];
                if (part[u] >= 0) continue;
                if (gain[u] == 0) candidates.push_back(u);
                ++gain[u];
            }
        };
        add(seed);
        while (size < maxSize) {
            Index best = -1;
            for (Index c : candidates) {
                if (part[c] >= 0) continue;
                if ((best < 0) || (gain[c] > gain[best]) || ((gain[c] == gain[best]) && (rank[c] < rank[best])))
                    best = c;
            }
            if (best < 0) break;
            add(best);
        }
        for (Index c : candidates) gain[c] = 0;
        candidates.clear();
    }
}

struct DisjointSets {
    std::vector<Index> parent;
    explicit DisjointSets(size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), Index(0)); }
    Index find(Index a) {
        while (parent[a] != a) a = parent[a] = parent[parent[a]];
        return a;
    }
    void merge(Index a, Index b) {
        a = find(a); b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
};

} // namespace

bool MASPreconditioner::metisAvailable() {
#if MESHFEM_WITH_SUITESPARSE_METIS
    return true;
#else
    return false;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Pattern analysis
////////////////////////////////////////////////////////////////////////////////
void MASPreconditioner::analyzePattern(const BlockCSCHessianBase &A, const std::vector<size_t> &pinnedVars) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("MASPreconditioner.analyzePattern");
    const size_t d = nodeBlockSize ? nodeBlockSize : std::max<size_t>(1, A.blockSizeGCD());

    // The scalar pattern is only needed here; values are later read from
    // `A.Ax` through the offset table, whatever the block storage layout.
    const SuiteSparseMatrix scalarPattern = A.toScalar(/* sparsityOnly = */ true);
    const auto offsets = A.dataOffsetsForScalarCSCDataOffsets(scalarPattern);
    std::vector<Index> axOffsets(offsets.data(), offsets.data() + offsets.size());

    m_analyze(scalarPattern, axOffsets.empty() ? nullptr : axOffsets.data(), d, pinnedVars);
    m_blockInput  = true;
    m_patternCols = size_t(A.n);
    m_patternNNZ  = A.Ai.size();
    m_valuesSize  = A.scalarNNZ();
}

void MASPreconditioner::analyzePattern(const SuiteSparseMatrix &A, const std::vector<size_t> &pinnedVars) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("MASPreconditioner.analyzePattern");
    m_analyze(A, nullptr, nodeBlockSize ? nodeBlockSize : 1, pinnedVars);
    m_blockInput  = false;
    m_patternCols = size_t(A.n);
    m_patternNNZ  = size_t(A.Ap[A.n]);
    m_valuesSize  = m_patternNNZ;
}

void MASPreconditioner::m_analyze(const SuiteSparseMatrix &S, const Index *axOffsets, size_t d, const std::vector<size_t> &pinnedVars) {
    clear();
    if (S.m != S.n) throw std::runtime_error("MASPreconditioner: matrix must be square");
    const size_t n = size_t(S.n);
    if (n % d != 0) throw std::invalid_argument("MASPreconditioner: matrix size is not a multiple of the node block size");

    m_numScalarVars = n;
    m_d = d;
    m_numNodes = n / d;
    const Index N = Index(m_numNodes), dd = Index(d * d);

    m_isPinned.assign(n, 0);
    for (size_t v : pinnedVars) {
        if (v >= n) throw std::runtime_error("MASPreconditioner: pinned variable index out of range");
        m_isPinned[v] = 1;
    }
    std::vector<char> nodeInMAS(m_numNodes, 0); // Nodes with at least one free variable
    for (size_t i = 0; i < n; ++i)
        if (!m_isPinned[i]) nodeInMAS[i / d] = 1;

    // Node block upper-triangle pattern; every node gets a diagonal block.
    std::vector<Index> bAp(N + 1, 0), bAi, marker(N, -1);
    for (Index J = 0; J < N; ++J) {
        const size_t start = bAi.size();
        marker[J] = J;
        bAi.push_back(J);
        for (Index j = J * Index(d); j < (J + 1) * Index(d); ++j) {
            for (Index k = S.Ap[j]; k < S.Ap[j + 1]; ++k) {
                Index i = S.Ai[k];
                if (i > j) continue; // Only the upper triangle is referenced.
                Index I = i / Index(d);
                if (marker[I] != J) { marker[I] = J; bAi.push_back(I); }
            }
        }
        std::sort(bAi.begin() + start, bAi.end());
        bAp[J + 1] = Index(bAi.size());
    }

    // Location of each scalar entry of each node block in the input values.
    m_entryAx.assign(bAi.size() * dd, -1);
    for (Index j = 0; j < Index(n); ++j) {
        for (Index k = S.Ap[j]; k < S.Ap[j + 1]; ++k) {
            Index i = S.Ai[k];
            if ((i > j) || m_isPinned[i] || m_isPinned[j]) continue;
            Index I = i / Index(d), J = j / Index(d), li = i % Index(d), lj = j % Index(d);
            Index e = Index(std::lower_bound(bAi.begin() + bAp[J], bAi.begin() + bAp[J + 1], I) - bAi.begin());
            Index src = axOffsets ? axOffsets[k] : k;
            m_entryAx[e * dd + li + lj * Index(d)] = src;
            if ((I == J) && (li != lj)) m_entryAx[e * dd + lj + li * Index(d)] = src;
        }
    }

    m_buildHierarchy(bAp, bAi, nodeInMAS);

    m_subdomainLevel.assign(m_numSubdomainsTotal, 0);
    for (size_t l = 0; l < m_levels.size(); ++l)
        for (size_t s = 0; s < m_levels[l].numSubdomains; ++s)
            m_subdomainLevel[m_levels[l].subdomainOffset + s] = l;

    m_levelR.assign(m_levelVecSize, 0.0);
    m_levelZ.assign(m_levelVecSize, 0.0);
    m_analyzed = true;
}

void MASPreconditioner::m_buildHierarchy(const std::vector<Index> &bAp, const std::vector<Index> &bAi, const std::vector<char> &nodeInMAS) {
    const Index N = Index(m_numNodes);
    const Index Bank = Index(BankSize);

    // Graph on the nodes that participate in MAS.
    std::vector<Index> freeIdx(N, -1), freeNode;
    for (Index I = 0; I < N; ++I)
        if (nodeInMAS[I]) { freeIdx[I] = Index(freeNode.size()); freeNode.push_back(I); }
    const Index nFree = Index(freeNode.size());

    std::vector<std::pair<Index, Index>> edges; // f < g, each undirected edge once
    for (Index J = 0; J < N; ++J) {
        if (freeIdx[J] < 0) continue;
        for (Index e = bAp[J]; e < bAp[J + 1]; ++e) {
            Index I = bAi[e];
            if ((I != J) && (freeIdx[I] >= 0)) edges.emplace_back(freeIdx[I], freeIdx[J]);
        }
    }
    Graph g;
    g.numVertices = nFree;
    g.xadj.assign(nFree + 1, 0);
    for (const auto &ed : edges) { ++g.xadj[ed.first + 1]; ++g.xadj[ed.second + 1]; }
    std::partial_sum(g.xadj.begin(), g.xadj.end(), g.xadj.begin());
    g.adj.resize(g.xadj[nFree]);
    {
        std::vector<Index> fill(g.xadj.begin(), g.xadj.end() - 1);
        for (const auto &ed : edges) {
            g.adj[fill[ed.first]++] = ed.second;
            g.adj[fill[ed.second]++] = ed.first;
        }
    }

    // Partition (Sec. 4.2).
    std::vector<Index> part;
    Index numParts = 0;
    m_usedMetis = false;
    if (nFree <= Bank) {
        part.assign(nFree, 0);
        numParts = (nFree > 0) ? 1 : 0;
    }
    else {
        bool done = false;
        if (partitionMethod != PartitionMethod::Greedy) {
#if MESHFEM_WITH_SUITESPARSE_METIS
            done = m_usedMetis = metisPartition(g, Bank, part, numParts);
#endif
            if (!done && (partitionMethod == PartitionMethod::Metis))
                throw std::runtime_error(metisAvailable() ? "MASPreconditioner: METIS partitioning failed"
                                                          : "MASPreconditioner: compiled without METIS");
        }
        if (!done) greedyPartition(g, Bank, part, numParts);
    }

    // Level 0: each partition gets its own subdomain of `BankSize` slots, padded with -1.
    m_slotNode.assign(numParts * Bank, -1);
    m_nodeSlot.assign(N, -1);
    {
        std::vector<Index> fill(numParts, 0);
        for (Index f = 0; f < nFree; ++f) {
            Index p = part[f];
            if (fill[p] >= Bank) throw std::logic_error("MASPreconditioner: partition exceeds BankSize");
            Index slot = p * Bank + fill[p]++;
            m_slotNode[slot] = freeNode[f];
            m_nodeSlot[freeNode[f]] = slot;
        }
    }

    m_levels.clear();
    std::vector<std::vector<Index>> maps; // maps[l][f]: node at level l of free fine node f
    {
        Level L0;
        L0.numNodes = size_t(numParts * Bank);
        L0.numSubdomains = size_t(numParts);
        m_levels.push_back(std::move(L0));
        std::vector<Index> m0(nFree);
        for (Index f = 0; f < nFree; ++f) m0[f] = m_nodeSlot[freeNode[f]];
        maps.push_back(std::move(m0));
    }

    // Coarser levels: merge connected nodes inside each subdomain into super
    // nodes, numbered consecutively subdomain by subdomain.
    while (m_levels.size() < std::max<size_t>(maxLevels, 1)) {
        const Level &L = m_levels.back();
        if (L.numSubdomains <= 1) break;
        const bool isFine = (m_levels.size() == 1);
        const Index Lsize = Index(L.numSubdomains) * Bank;
        const auto &map = maps.back();
        auto active = [&](Index a) { return isFine ? (m_slotNode[a] >= 0) : (a < Index(L.numNodes)); };

        DisjointSets dsu(Lsize);
        for (const auto &ed : edges) {
            Index a = map[ed.first], b = map[ed.second];
            if (a / Bank == b / Bank) dsu.merge(a, b);
        }
        std::vector<Index> next(Lsize, -1), newId(Lsize, -1);
        Index count = 0;
        for (Index a = 0; a < Lsize; ++a) {
            if (!active(a)) continue;
            Index r = dsu.find(a);
            if (newId[r] < 0) newId[r] = count++;
            next[a] = newId[r];
        }
        const Index activeCount = isFine ? nFree : Index(L.numNodes);
        if (count == activeCount) break; // No aggregation possible.

        Level nl;
        nl.numNodes = size_t(count);
        nl.numSubdomains = size_t((count + Bank - 1) / Bank);
        nl.childPtr.assign(count + 1, 0);
        for (Index a = 0; a < Lsize; ++a)
            if (next[a] >= 0) ++nl.childPtr[next[a] + 1];
        std::partial_sum(nl.childPtr.begin(), nl.childPtr.end(), nl.childPtr.begin());
        nl.children.resize(nl.childPtr[count]);
        {
            std::vector<Index> fill(nl.childPtr.begin(), nl.childPtr.end() - 1);
            for (Index a = 0; a < Lsize; ++a)
                if (next[a] >= 0) nl.children[fill[next[a]]++] = a;
        }
        std::vector<Index> newMap(nFree);
        for (Index f = 0; f < nFree; ++f) newMap[f] = next[map[f]];
        nl.parentOfPrev = std::move(next);
        m_levels.push_back(std::move(nl));
        maps.push_back(std::move(newMap));
    }

    // Offsets of each level's subdomains and residual/solution slots.
    m_numSubdomainsTotal = 0;
    m_levelVecSize = 0;
    for (Level &L : m_levels) {
        L.subdomainOffset = m_numSubdomainsTotal;
        L.vecOffset = m_levelVecSize;
        m_numSubdomainsTotal += L.numSubdomains;
        m_levelVecSize += L.numSubdomains * BankSize * m_d;
    }

    // Galerkin contribution lists: block entry (I, J) contributes to the
    // subdomain matrix of level l whenever both nodes land in the same
    // level-l subdomain.
    m_contribPtr.assign(m_numSubdomainsTotal + 1, 0);
    auto visitContributions = [&](auto &&f) {
        for (Index J = 0; J < N; ++J) {
            if (freeIdx[J] < 0) continue;
            for (Index e = bAp[J]; e < bAp[J + 1]; ++e) {
                Index I = bAi[e];
                if (freeIdx[I] < 0) continue;
                for (size_t l = 0; l < m_levels.size(); ++l) {
                    Index a = maps[l][freeIdx[I]], b = maps[l][freeIdx[J]];
                    if (a / Bank != b / Bank) continue;
                    f(m_levels[l].subdomainOffset + size_t(a / Bank),
                      Contribution{e, (I == J) ? I : Index(-1), uint8_t(a % Bank), uint8_t(b % Bank)});
                }
            }
        }
    };
    visitContributions([&](size_t sub, const Contribution &) { ++m_contribPtr[sub + 1]; });
    std::partial_sum(m_contribPtr.begin(), m_contribPtr.end(), m_contribPtr.begin());
    m_contrib.resize(m_contribPtr.back());
    {
        std::vector<Index> fill(m_contribPtr.begin(), m_contribPtr.end() - 1);
        visitContributions([&](size_t sub, const Contribution &c) { m_contrib[fill[sub]++] = c; });
    }
}

////////////////////////////////////////////////////////////////////////////////
// Numeric phase
////////////////////////////////////////////////////////////////////////////////
void MASPreconditioner::compute(const BlockCSCHessianBase &A, Real sigma) {
    if (!m_analyzed || !m_blockInput || (size_t(A.n) != m_patternCols) || (A.Ai.size() != m_patternNNZ) || (A.Ax.size() != m_valuesSize))
        throw std::runtime_error("MASPreconditioner: matrix does not match the pattern passed to analyzePattern");
    m_assembleAndInvert(A.Ax.data(), sigma);
}

void MASPreconditioner::compute(const SuiteSparseMatrix &A, Real sigma) {
    if (!m_analyzed || m_blockInput || (size_t(A.n) != m_patternCols) || (size_t(A.Ap[A.n]) != m_patternNNZ) || (A.Ax.size() < m_valuesSize))
        throw std::runtime_error("MASPreconditioner: matrix does not match the pattern passed to analyzePattern");
    m_assembleAndInvert(A.Ax.data(), sigma);
}

// Assemble the dense Galerkin matrix of every subdomain from its
// contribution list (no atomics: each subdomain is owned by one task), pad
// inactive slots with the identity, and invert. Pinned variables contribute
// identity diagonal entries; free ones get the shift `sigma`.
void MASPreconditioner::m_assembleAndInvert(const Real *Ax, Real sigma) {
    BENCHMARK_SCOPED_TIMER_SECTION timer("MASPreconditioner.compute");
    const size_t d = m_d, dd = d * d, dim = m_subdomainDim();
    m_computed = false;
    m_inverses.assign(m_numSubdomainsTotal * dim * dim, 0.0f);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_numSubdomainsTotal), [&](const tbb::blocked_range<size_t> &r) {
        Eigen::MatrixXd M(dim, dim), inv(dim, dim), blk(d, d);
        for (size_t sub = r.begin(); sub < r.end(); ++sub) {
            const size_t l = m_subdomainLevel[sub];
            const Level &L = m_levels[l];
            const size_t s = sub - L.subdomainOffset;

            M.setZero();
            for (Index t = m_contribPtr[sub]; t < m_contribPtr[sub + 1]; ++t) {
                const Contribution &c = m_contrib[t];
                const Index *loc = m_entryAx.data() + c.entry * dd;
                for (size_t k = 0; k < dd; ++k) blk.data()[k] = (loc[k] >= 0) ? Ax[loc[k]] : 0.0;
                if (c.diagNode >= 0) {
                    for (size_t comp = 0; comp < d; ++comp)
                        blk(comp, comp) += m_isPinned[c.diagNode * d + comp] ? 1.0 : sigma;
                    M.block(c.a * d, c.b * d, d, d) += blk;
                }
                else {
                    M.block(c.a * d, c.b * d, d, d) += blk;
                    M.block(c.b * d, c.a * d, d, d) += blk.transpose();
                }
            }
            for (size_t a = 0; a < BankSize; ++a) {
                const size_t node = s * BankSize + a;
                const bool active = (l == 0) ? (m_slotNode[node] >= 0) : (node < L.numNodes);
                if (!active) M.block(a * d, a * d, d, d).setIdentity();
            }

            Eigen::LLT<Eigen::MatrixXd> llt(M);
            if (llt.info() != Eigen::Success)
                throw std::runtime_error("MASPreconditioner: matrix is not positive definite (level " + std::to_string(l) + " subdomain " + std::to_string(s) + ")");
            inv.setIdentity();
            llt.solveInPlace(inv);

            float *out = m_inverses.data() + sub * dim * dim;
            for (size_t j = 0; j < dim; ++j)
                for (size_t i = 0; i < dim; ++i)
                    out[i + j * dim] = float(0.5 * (inv(i, j) + inv(j, i)));
        }
    });

    m_computed = true;
}

////////////////////////////////////////////////////////////////////////////////
// Application
////////////////////////////////////////////////////////////////////////////////
// z = sum_l R_l^T M_l^{-1} R_l r on the free variables (R_l sums fine nodes
// into their level-l super nodes); z = r on the pinned variables.
void MASPreconditioner::apply(const Real *r, Real *z) const {
    if (!m_computed) throw std::runtime_error("MASPreconditioner: compute() must be called before apply()");
    const size_t d = m_d, dim = m_subdomainDim();
    Real *R = m_levelR.data();
    Real *Z = m_levelZ.data();
    std::fill(m_levelR.begin(), m_levelR.end(), 0.0);

    // Restriction
    const Level &L0 = m_levels[0];
    tbb::parallel_for(tbb::blocked_range<size_t>(0, L0.numSubdomains * BankSize), [&](const tbb::blocked_range<size_t> &rg) {
        for (size_t slot = rg.begin(); slot < rg.end(); ++slot) {
            const Index node = m_slotNode[slot];
            if (node < 0) continue;
            for (size_t c = 0; c < d; ++c) {
                const size_t i = node * d + c;
                R[slot * d + c] = m_isPinned[i] ? 0.0 : r[i];
            }
        }
    });
    for (size_t l = 1; l < m_levels.size(); ++l) {
        const Level &L = m_levels[l], &P = m_levels[l - 1];
        tbb::parallel_for(tbb::blocked_range<size_t>(0, L.numNodes), [&](const tbb::blocked_range<size_t> &rg) {
            for (size_t q = rg.begin(); q < rg.end(); ++q) {
                Real *out = R + L.vecOffset + q * d;
                for (Index t = L.childPtr[q]; t < L.childPtr[q + 1]; ++t) {
                    const Real *in = R + P.vecOffset + L.children[t] * d;
                    for (size_t c = 0; c < d; ++c) out[c] += in[c];
                }
            }
        });
    }

    // Local solves on all subdomains of all levels (additive, independent).
    // Level vectors are laid out subdomain by subdomain in global order.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_numSubdomainsTotal), [&](const tbb::blocked_range<size_t> &rg) {
        for (size_t sub = rg.begin(); sub < rg.end(); ++sub) {
            const Real *rs = R + sub * dim;
            Real *zs = Z + sub * dim;
            const float *inv = m_inverses.data() + sub * dim * dim;
            for (size_t i = 0; i < dim; ++i) zs[i] = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                const Real rj = rs[j];
                if (rj == 0.0) continue;
                const float *col = inv + j * dim;
                for (size_t i = 0; i < dim; ++i) zs[i] += Real(col[i]) * rj;
            }
        }
    });

    // Prolongation (injection): add each super node's correction to its children.
    for (size_t l = m_levels.size() - 1; l >= 1; --l) {
        const Level &L = m_levels[l], &P = m_levels[l - 1];
        tbb::parallel_for(tbb::blocked_range<size_t>(0, P.numSubdomains * BankSize), [&](const tbb::blocked_range<size_t> &rg) {
            for (size_t p = rg.begin(); p < rg.end(); ++p) {
                const Index q = L.parentOfPrev[p];
                if (q < 0) continue;
                const Real *in = Z + L.vecOffset + q * d;
                Real *out = Z + P.vecOffset + p * d;
                for (size_t c = 0; c < d; ++c) out[c] += in[c];
            }
        });
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_numNodes), [&](const tbb::blocked_range<size_t> &rg) {
        for (size_t I = rg.begin(); I < rg.end(); ++I) {
            const Index slot = m_nodeSlot[I];
            for (size_t c = 0; c < d; ++c) {
                const size_t i = I * d + c;
                z[i] = (m_isPinned[i] || (slot < 0)) ? r[i] : Z[slot * d + c];
            }
        }
    });
}

void MASPreconditioner::clear() {
    m_analyzed = false;
    m_computed = false;
    m_numScalarVars = 0;
    m_numNodes = 0;
    m_isPinned.clear();
    m_entryAx.clear();
    m_levels.clear();
    m_slotNode.clear();
    m_nodeSlot.clear();
    m_contribPtr.clear();
    m_contrib.clear();
    m_subdomainLevel.clear();
    m_numSubdomainsTotal = 0;
    m_levelVecSize = 0;
    m_inverses.clear();
    m_levelR.clear();
    m_levelZ.clear();
}

} // namespace MeshFEM
