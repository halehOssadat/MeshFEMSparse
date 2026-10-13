////////////////////////////////////////////////////////////////////////////////
// Preconditioner.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
// Preconditioner interface used by the iterative solvers (see PCG.hh).
*///////////////////////////////////////////////////////////////////////////////
#ifndef ITERATIVE_PRECONDITIONER_HH
#define ITERATIVE_PRECONDITIONER_HH

#include <MeshFEMCore/Types.hh>

#include <cstddef>

namespace MeshFEM {

struct MESHFEM_EXPORT Preconditioner {
    virtual ~Preconditioner() = default;
    virtual size_t size() const = 0;
    // z = M^{-1} r (r and z have length `size()` and do not alias).
    virtual void apply(const Real *r, Real *z) const = 0;
};

} // namespace MeshFEM

#endif /* end of include guard: ITERATIVE_PRECONDITIONER_HH */
