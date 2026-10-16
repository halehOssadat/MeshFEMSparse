#include "block_csc_to_triplets.hh"
#include <MeshFEMCore/Parallelism.hh>
#include <memory>
#include <stdexcept>

namespace MeshFEM {

BlockTriplets3 blockCSCToTriplets(const BlockCSCHessianBase &H_in) {
    if (H_in.blockSizes() != std::vector<size_t>{3})
        throw std::runtime_error("blockCSCToTriplets: only uniform 3x3 blocks are supported");

    // Block k must occupy Ax[9k, 9k + 9).
    std::unique_ptr<BlockCSCHessianBase> H_contiguous;
    if (!H_in.hasContiguousBlocks()) H_contiguous = H_in.cloneWithContiguousBlocks();
    const BlockCSCHessianBase &H = H_contiguous ? *H_contiguous : H_in;

    const size_t numBlocks = H.Ai.size();
    if (size_t(H.Ax.size()) < 9 * numBlocks)
        throw std::runtime_error("blockCSCToTriplets: matrix has no values (sparsity-only?)");

    BlockTriplets3 result;
    result.rows = int(H.m);
    result.cols = int(H.n);
    result.row_indices.resize(numBlocks);
    result.col_indices.resize(numBlocks);
    result.values.resize(numBlocks);

    // Each block column bj writes only slots [Ap[bj], Ap[bj + 1]), so columns are independent.
    parallel_for_range(size_t(H.n), [&](size_t bj) {
        for (SuiteSparse_long k = H.Ap[bj]; k < H.Ap[bj + 1]; ++k) {
            result.row_indices[k] = int(H.Ai[k]);
            result.col_indices[k] = int(bj);
            Eigen::Matrix3d block = Eigen::Map<const Eigen::Matrix3d>(H.Ax.data() + 9 * k);
            if (SuiteSparse_long(H.Ai[k]) == SuiteSparse_long(bj)) {
                // The strict lower triangle of diagonal blocks may be uninitialized.
                block(1, 0) = block(0, 1);
                block(2, 0) = block(0, 2);
                block(2, 1) = block(1, 2);
            }
            result.values[k] = block;
        }
    });

    return result;
}

} // namespace MeshFEM
