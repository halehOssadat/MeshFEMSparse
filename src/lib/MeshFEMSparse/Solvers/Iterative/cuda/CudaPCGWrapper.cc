#include "CudaPCGWrapper.hh"
#include "block_csc_to_triplets.hh"

namespace MeshFEM {

void CudaPCGWrapper::compute(const BlockCSCHessianBase &H) {
    BlockTriplets3 triplets = blockCSCToTriplets(H);
    m_buildLinearSystem(triplets.rows, triplets.cols, triplets.values.size(),
                        triplets.row_indices.data(), triplets.col_indices.data(),
                        triplets.values.data());
}

} // namespace MeshFEM
