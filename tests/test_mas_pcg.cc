#include <MeshFEMSparse/Solvers/Iterative/MASPreconditioner.hh>
#include <MeshFEMSparse/Solvers/Iterative/PCG.hh>

// Include Catch after MeshFEM to avoid its BENCHMARK macro conflict.
#include <catch2/catch.hpp>

#include <cmath>

using namespace MeshFEM;

namespace {

// Upper triangle of kron(L + mass * I, S) for the 6-neighbor graph Laplacian
// L of an n x n x n grid and a fixed SPD 3x3 block S. Optionally flips the
// sign of one node's diagonal block to make the matrix indefinite.
SuiteSparseMatrix gridSystem(size_t n, double mass, long negativeNode = -1) {
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
        const double scale = (long(a) == negativeNode) ? -(degree[a] + mass) : (degree[a] + mass);
        for (size_t c = 0; c < 3; ++c)
            for (size_t r = 0; r <= c; ++r)
                triplets.addNZ(3 * a + r, 3 * a + c, scale * S(r, c));
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

Eigen::VectorXd expectedSolution(size_t n, const std::vector<size_t> &pins) {
    Eigen::VectorXd x(n);
    for (size_t i = 0; i < n; ++i) x[i] = std::sin(0.3 * double(i + 1)) + 0.25;
    for (size_t i : pins) x[i] = 0.0;
    return x;
}

// Right-hand side of (A + sigma I) x = b on the free variables, zero on the pinned ones.
Eigen::VectorXd rhs(const SuiteSparseMatrix &scalarA, const Eigen::VectorXd &x, double sigma, const std::vector<size_t> &pins) {
    Eigen::VectorXd b = scalarA.apply(x) + sigma * x;
    for (size_t i : pins) b[i] = 0.0;
    return b;
}

template<class Matrix>
void checkSolves(MASPreconditioner &mas, const Matrix &A, const SuiteSparseMatrix &scalarA, const std::vector<size_t> &pins) {
    const Eigen::VectorXd expected = expectedSolution(scalarA.n, pins);
    PCG pcg;
    pcg.relTol = 1e-11;
    mas.analyzePattern(A, pins);
    for (double sigma : {0.0, 0.5, 0.0}) {
        CAPTURE(sigma);
        mas.compute(A, sigma);
        REQUIRE(mas.isComputed());
        SymmetricMatrixOperator op(A, pins, sigma);
        Eigen::VectorXd x;
        const PCG::Stats stats = pcg.solve(op, mas, rhs(scalarA, expected, sigma, pins), x);
        REQUIRE(stats.converged);
        REQUIRE((x - expected).norm() < 1e-8 * expected.norm());
    }
}

} // namespace

TEST_CASE("MAS-preconditioned PCG solves SPD block systems", "[maspcg]") {
    const size_t n = 7;
    SuiteSparseMatrix scalarA = gridSystem(n, 1e-2);
    const auto pins = GENERATE(std::vector<size_t>{}, std::vector<size_t>{0, 1, 2}, std::vector<size_t>{4, 100, 101});
    const auto partition = GENERATE(MASPreconditioner::PartitionMethod::Auto, MASPreconditioner::PartitionMethod::Greedy);
    CAPTURE(pins, int(partition));

    MASPreconditioner mas;
    mas.partitionMethod = partition;

    SECTION("block matrix, contiguous blocks") {
        auto A = BlockCSCHessianBase::fromScalar(scalarA, 3)->cloneWithContiguousBlocks();
        REQUIRE(A->hasContiguousBlocks());
        checkSolves(mas, *A, scalarA, pins);
        REQUIRE(mas.nodeBlockSizeUsed() == 3);
    }
    SECTION("block matrix, noncontiguous blocks") {
        auto A = BlockCSCHessianBase::fromScalar(scalarA, 3)->cloneWithNoncontiguousBlocks();
        REQUIRE(!A->hasContiguousBlocks());
        checkSolves(mas, *A, scalarA, pins);
        REQUIRE(mas.nodeBlockSizeUsed() == 3);
    }
    SECTION("scalar matrix with node block size") {
        mas.nodeBlockSize = 3;
        checkSolves(mas, scalarA, scalarA, pins);
        REQUIRE(mas.nodeBlockSizeUsed() == 3);
    }
    SECTION("scalar matrix with scalar nodes") {
        checkSolves(mas, scalarA, scalarA, pins);
        REQUIRE(mas.nodeBlockSizeUsed() == 1);
    }

    REQUIRE(mas.numLevels() >= 2);
    const size_t numNodes = n * n * n * 3 / mas.nodeBlockSizeUsed();
    REQUIRE(mas.numSubdomains(0) * MASPreconditioner::BankSize >= numNodes - pins.size());
    for (size_t l = 1; l < mas.numLevels(); ++l)
        REQUIRE(mas.numLevelNodes(l) < mas.numLevelNodes(l - 1));
    if (partition == MASPreconditioner::PartitionMethod::Greedy) REQUIRE(!mas.usedMetis());
    else                                                         REQUIRE(mas.usedMetis() == MASPreconditioner::metisAvailable());
}

TEST_CASE("SymmetricMatrixOperator and MAS treat pinned variables as identity", "[maspcg]") {
    SuiteSparseMatrix scalarA = gridSystem(4, 1e-1);
    auto A = BlockCSCHessianBase::fromScalar(scalarA, 3);
    const std::vector<size_t> pins{1, 5, 30};
    const double sigma = 0.25;
    const size_t n = scalarA.n;

    Eigen::VectorXd x(n), w(n);
    for (size_t i = 0; i < n; ++i) { x[i] = std::cos(0.7 * double(i)); w[i] = std::sin(1.3 * double(i) + 0.2); }

    // Operator: (A + sigma I) on the free variables, identity on the pinned ones.
    Eigen::VectorXd xFree = x;
    for (size_t i : pins) xFree[i] = 0.0;
    Eigen::VectorXd expected = scalarA.apply(xFree) + sigma * xFree;
    for (size_t i : pins) expected[i] = x[i];
    Eigen::VectorXd yBlock(n), yScalar(n);
    SymmetricMatrixOperator(*A, pins, sigma).apply(x.data(), yBlock.data());
    SymmetricMatrixOperator(scalarA, pins, sigma).apply(x.data(), yScalar.data());
    REQUIRE((yBlock  - expected).norm() < 1e-12 * expected.norm());
    REQUIRE((yScalar - expected).norm() < 1e-12 * expected.norm());

    // Preconditioner: symmetric, and the identity on the pinned variables.
    MASPreconditioner mas;
    mas.analyzePattern(*A, pins);
    REQUIRE_THROWS_AS(mas.apply(x.data(), yBlock.data()), std::runtime_error);
    mas.compute(*A, sigma);
    Eigen::VectorXd Mx(n), Mw(n);
    mas.apply(x.data(), Mx.data());
    mas.apply(w.data(), Mw.data());
    REQUIRE(std::abs(w.dot(Mx) - x.dot(Mw)) < 1e-5 * std::abs(w.dot(Mx)));
    for (size_t i : pins) REQUIRE(Mx[i] == x[i]);
}

TEST_CASE("MAS multilevel hierarchy accelerates convergence", "[maspcg]") {
    SuiteSparseMatrix A = gridSystem(10, 1e-4);
    const Eigen::VectorXd b = A.apply(expectedSolution(A.n, {}));
    SymmetricMatrixOperator op(A);

    auto iterations = [&](size_t maxLevels) {
        MASPreconditioner mas;
        mas.nodeBlockSize = 3;
        mas.maxLevels = maxLevels;
        mas.analyzePattern(A);
        mas.compute(A);
        PCG pcg;
        pcg.relTol = 1e-8;
        pcg.solve(op, mas, b);
        REQUIRE(pcg.lastStats().converged);
        return pcg.lastStats().iterations;
    };
    const size_t singleLevel = iterations(1), multiLevel = iterations(6);
    CAPTURE(singleLevel, multiLevel);
    REQUIRE(multiLevel < singleLevel);
}

TEST_CASE("MAS detects indefinite matrices", "[maspcg]") {
    SuiteSparseMatrix A = gridSystem(5, 1e-2, /* negativeNode = */ 17);
    MASPreconditioner mas;
    mas.nodeBlockSize = 3;
    mas.analyzePattern(A);
    REQUIRE_THROWS_AS(mas.compute(A), std::runtime_error);
    REQUIRE(!mas.isComputed());
    // A sufficiently large shift restores definiteness (as in Newton's shift loop).
    mas.compute(A, 100.0);
    REQUIRE(mas.isComputed());
}

TEST_CASE("MAS rejects matrices that do not match the analyzed pattern", "[maspcg]") {
    SuiteSparseMatrix scalarA = gridSystem(3, 1e-1);
    auto A = BlockCSCHessianBase::fromScalar(scalarA, 3);
    MASPreconditioner mas;

    REQUIRE_THROWS_AS(mas.compute(scalarA), std::runtime_error); // not analyzed
    mas.analyzePattern(scalarA);
    REQUIRE_THROWS_AS(mas.compute(*A), std::runtime_error);      // block vs. scalar
    REQUIRE_THROWS_AS(mas.compute(gridSystem(4, 1e-1)), std::runtime_error);
    mas.compute(scalarA);
    REQUIRE(mas.isComputed());

    PCG pcg;
    Eigen::VectorXd x, wrongSize(5);
    REQUIRE_THROWS_AS(pcg.solve(SymmetricMatrixOperator(scalarA), mas, wrongSize, x), std::runtime_error);
}
