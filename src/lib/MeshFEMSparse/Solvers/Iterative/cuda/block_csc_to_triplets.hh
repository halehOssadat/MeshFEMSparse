////////////////////////////////////////////////////////////////////////////////
// block_csc_to_triplets.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Convert a MeshFEM BlockCSCHessian (uniform 3x3 blocks) into host-side 3x3
// block triplets with the same fields as libuipc's
// cuda_tool::DeviceTripletMatrix<Float, 3>. On the GPU they are turned into
// libuipc's BCOO by MatrixConverter::ge2sym + convert.
*///////////////////////////////////////////////////////////////////////////////
#ifndef BLOCK_CSC_TO_TRIPLETS_HH
#define BLOCK_CSC_TO_TRIPLETS_HH

#include <MeshFEMSparse/BlockCSCHessian.hh>
#include <Eigen/Dense>
#include <vector>

namespace MeshFEM {

// Host-side 3x3 block triplets: same fields as libuipc's DeviceTripletMatrix<Float, 3>.
struct BlockTriplets3 {
    int rows = 0, cols = 0;                 // number of block rows/cols
    std::vector<int>             row_indices;
    std::vector<int>             col_indices;
    std::vector<Eigen::Matrix3d> values;    // column-major 3x3 blocks
};

// Upper-triangle blocks of `H` (as stored), with diagonal blocks made full (lower = upper^T).
BlockTriplets3 blockCSCToTriplets(const BlockCSCHessianBase &H);

} // namespace MeshFEM

#endif /* end of include guard: BLOCK_CSC_TO_TRIPLETS_HH */
