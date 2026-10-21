#if MESHFEM_WITH_CUDA_PCG
#include <MeshFEMSparse/Solvers/Iterative/cuda/CudaPCGWrapper.hh>
#include <MeshFEMSparse/BlockCSCHessian.hh>

// Include Catch after MeshFEM to avoid its BENCHMARK macro conflict.
#include <catch2/catch.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

using namespace MeshFEM;

namespace {

// Upper triangle of kron(L + mass * I, S) for the 6-neighbor graph Laplacian
// L of an n x n x n grid and a fixed SPD 3x3 block S.
SuiteSparseMatrix gridSystem(size_t n, double mass) {
    Eigen::Matrix3d S;
    S << 2.0, 0.5, 0.0,
         0.5, 2.0, 0.3,
         0.0, 0.3, 1.5;
    const size_t numNodes = n * n * n;
    auto id = [n](size_t x, size_t y, size_t z) { return x + n * (y + n * z); };

    std::vector<double> degree(numNodes, 0.0);
    std::vector<std::pair<size_t, size_t>> edges;
    for (size_t z = 0; z < n; ++z)
        for (size_t y = 0; y < n; ++y)
            for (size_t x = 0; x < n; ++x) {
                const size_t a = id(x, y, z);
                if (x + 1 < n) edges.emplace_back(a, id(x + 1, y, z));
                if (y + 1 < n) edges.emplace_back(a, id(x, y + 1, z));
                if (z + 1 < n) edges.emplace_back(a, id(x, y, z + 1));
            }
    for (const auto &e : edges) { degree[e.first] += 1; degree[e.second] += 1; }

    TripletMatrix<> triplets(3 * numNodes, 3 * numNodes);
    for (size_t a = 0; a < numNodes; ++a) {
        for (size_t c = 0; c < 3; ++c)
            for (size_t r = 0; r <= c; ++r)
                triplets.addNZ(3 * a + r, 3 * a + c, (degree[a] + mass) * S(r, c));
    }
    for (const auto &e : edges) {
        const size_t a = std::min(e.first, e.second), b = std::max(e.first, e.second);
        for (size_t c = 0; c < 3; ++c)
            for (size_t r = 0; r < 3; ++r)
                triplets.addNZ(3 * a + r, 3 * b + c, -S(r, c));
    }
    SuiteSparseMatrix A(triplets);
    A.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    return A;
}

Eigen::VectorXd expectedSolution(size_t n) {
    Eigen::VectorXd x(n);
    for (size_t i = 0; i < n; ++i) x[i] = std::sin(0.3 * double(i + 1)) + 0.25;
    return x;
}

}

TEST_CASE("CUDA PCG defaults match libuipc", "[cudapcg]") {
    REQUIRE(CudaPCGWrapper().solver() == "fused_pcg");  // libuipc config linear_system/solver

    const std::string solver = GENERATE(std::string("fused_pcg"), std::string("linear_pcg"));
    CAPTURE(solver);
    CudaPCGWrapper pcg(solver);
    REQUIRE(pcg.tolRate() == 1e-3);      // libuipc config linear_system/tol_rate
    REQUIRE(pcg.maxIterRatio() == 2.0);  // libuipc do_build
}

TEST_CASE("CUDA PCG solves SPD block systems", "[cudapcg]") {
    const std::string solver = GENERATE(std::string("fused_pcg"), std::string("linear_pcg"));
    CAPTURE(solver);
    const SuiteSparseMatrix scalarA = gridSystem(6, 1e-2);
    const Eigen::VectorXd expected = expectedSolution(scalarA.n);
    const Eigen::VectorXd b = scalarA.apply(expected);

    auto check = [&](const BlockCSCHessianBase &H) {
        CudaPCGWrapper pcg(solver);
        pcg.setTolRate(1e-16);           // |r.z| <= tolRate |r0.z0|
        pcg.compute(H);
        Eigen::VectorXd x = pcg.solve(b);
        CAPTURE(pcg.iterCount());
        REQUIRE(pcg.iterCount() > 0);
        REQUIRE(pcg.iterCount() < pcg.maxIterRatio() * scalarA.n);  // converged before max_iter
        REQUIRE((x - expected).norm() < 1e-6 * expected.norm());
    };

    SECTION("contiguous blocks") {
        auto H = BlockCSCHessianBase::fromScalar(scalarA, 3)->cloneWithContiguousBlocks();
        check(*H);
    }
    SECTION("noncontiguous blocks") {
        auto H = BlockCSCHessianBase::fromScalar(scalarA, 3)->cloneWithNoncontiguousBlocks();
        check(*H);
    }
}

TEST_CASE("CUDA PCG converges with the libuipc default tolerance", "[cudapcg]") {
    const std::string solver = GENERATE(std::string("fused_pcg"), std::string("linear_pcg"));
    CAPTURE(solver);
    const SuiteSparseMatrix scalarA = gridSystem(6, 1e-2);
    const Eigen::VectorXd b = scalarA.apply(expectedSolution(scalarA.n));
    auto H = BlockCSCHessianBase::fromScalar(scalarA, 3);

    CudaPCGWrapper pcg(solver);
    pcg.compute(*H);
    Eigen::VectorXd x = pcg.solve(b);
    CAPTURE(pcg.iterCount());
    REQUIRE(pcg.iterCount() < pcg.maxIterRatio() * scalarA.n);
    REQUIRE((scalarA.apply(x) - b).norm() < b.norm());
}

TEST_CASE("CUDA PCG rejects invalid input", "[cudapcg]") {
    const SuiteSparseMatrix scalarA = gridSystem(3, 1e-2);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(scalarA.n);
    REQUIRE_THROWS_AS(CudaPCGWrapper("cg"), std::runtime_error);             // unknown solver

    CudaPCGWrapper pcg;
    REQUIRE_THROWS_AS(pcg.solve(b), std::runtime_error);                     // before compute
    pcg.compute(*BlockCSCHessianBase::fromScalar(scalarA, 3));
    REQUIRE_THROWS_AS(pcg.solve(b.head(scalarA.n - 3)), std::runtime_error); // wrong size
    REQUIRE_THROWS_AS(pcg.compute(*BlockCSCHessianBase::fromScalar(scalarA, 1)),
                      std::runtime_error);                                   // not 3x3 blocks
}

#endif // MESHFEM_WITH_CUDA_PCG
