// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on deal.II.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/ida_lagging_policy.h>

#ifndef IMMERSX_USE_PETSC_LA
#  include <deal.II/base/index_set.h>
#  include <deal.II/base/mpi.h>

#  include <deal.II/lac/trilinos_solver.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_sparsity_pattern.h>

#  include <chrono>
#  include <cmath>
#  include <filesystem>
#  include <fstream>
#  include <memory>
#  include <sstream>
#  include <string>
#  include <thread>
#  include <vector>

#  include "test_paths.h"

namespace
{
  using namespace ImmersX;
  using Vector = ImmersXLA::MPI::Vector;
  using Policy = IDALaggingPolicy<Vector>;

  // J(t, y, alpha) = K + (alpha + 0.1 t + 0.01 ||y||) I with a nonsymmetric
  // tridiagonal K, distributed over all ranks; rows i >= n / 2 are scaled
  // by row_scale, like the rows of a field whose equations are much smaller
  // than its unknowns. The factorization acts on a
  // copy, so it stays fixed while J changes, as a lagged seed must. The
  // optional delays make the policy's timing rules fire deterministically.
  // The error weights mimic IDA's, 1 / (rtol |y| + atol): IDA's current
  // ones at the state of the last step, times weight_scale (a tolerance
  // ramp), and the final ones at a given state.
  struct ToySystem
  {
    static constexpr unsigned int n = 24;

    ToySystem()
      : owned(dealii::Utilities::MPI::create_evenly_distributed_partitioning(
          MPI_COMM_WORLD,
          n))
    {
      dealii::TrilinosWrappers::SparsityPattern pattern(owned,
                                                        owned,
                                                        MPI_COMM_WORLD,
                                                        3);
      for (const auto i : owned)
        {
          pattern.add(i, i);
          if (i > 0)
            pattern.add(i, i - 1);
          if (i + 1 < n)
            pattern.add(i, i + 1);
        }
      pattern.compress();
      jacobian.reinit(pattern);
      factored.reinit(pattern);
      state = vector();
    }

    void
    build(const double t, const Vector &y, const double alpha)
    {
      const double shift = alpha + .1 * t + .01 * y.l2_norm();
      for (const auto i : owned)
        {
          const double s = i >= n / 2 ? row_scale : 1.;
          jacobian.set(i, i, s * (2. + shift));
          if (i > 0)
            jacobian.set(i, i - 1, -s);
          if (i + 1 < n)
            jacobian.set(i, i + 1, -.5 * s);
        }
      jacobian.compress(dealii::VectorOperation::insert);
    }

    Vector
    vector(const double scale = 1.) const
    {
      Vector v(owned, MPI_COMM_WORLD);
      for (const auto i : owned)
        v(i) = scale * (1. + std::sin(1. + i));
      v.compress(dealii::VectorOperation::insert);
      return v;
    }

    void
    weights(Vector &w) const
    {
      final_weights(state, w);
      w *= weight_scale;
    }

    void
    final_weights(const Vector &y, Vector &w) const
    {
      w.reinit(owned, MPI_COMM_WORLD);
      for (const auto i : owned)
        w(i) = 1. / (1.e-3 * std::abs(y(i)) + 1.e-4);
      w.compress(dealii::VectorOperation::insert);
    }

    // ||d .* w||_2 / sqrt(N) with the final weights at y.
    double
    wrms(const Vector &d, const Vector &y) const
    {
      Vector w;
      final_weights(y, w);
      Vector scaled(d);
      scaled.scale(w);
      return scaled.l2_norm() / std::sqrt(static_cast<double>(n));
    }

    // ||J z - r||_2 with the Jacobian assembled last.
    double
    residual_norm(const Vector &z, const Vector &r) const
    {
      Vector Jz(r);
      jacobian.vmult(Jz, z);
      Jz -= r;
      return Jz.l2_norm();
    }

    Policy::Operations
    operations()
    {
      Policy::Operations ops;
      ops.build_jacobian = [this](const double  t,
                                  const Vector &y,
                                  const Vector &,
                                  const double alpha) { build(t, y, alpha); };
      ops.factorize      = [this]() {
        std::this_thread::sleep_for(factorize_delay);
        factored.copy_from(jacobian);
        direct = std::make_unique<dealii::TrilinosWrappers::SolverDirect>();
        direct->initialize(factored);
      };
      ops.apply_factorization = [this](Vector &dst, const Vector &src) {
        direct->solve(dst, src);
      };
      ops.direct_solve   = ops.apply_factorization;
      ops.apply_jacobian = [this](Vector &dst, const Vector &src) {
        std::this_thread::sleep_for(jacobian_delay);
        jacobian.vmult(dst, src);
      };
      ops.error_weights = [this](Vector &w) { weights(w); };
      ops.state_weights = [this](const Vector &y, Vector &w) {
        final_weights(y, w);
      };
      return ops;
    }

    dealii::IndexSet                                        owned;
    dealii::TrilinosWrappers::SparseMatrix                  jacobian, factored;
    std::unique_ptr<dealii::TrilinosWrappers::SolverDirect> direct;
    std::chrono::milliseconds factorize_delay{0}, jacobian_delay{0};
    Vector                    state;
    double                    weight_scale = 1.;
    double                    row_scale    = 1.;
  };

  // T_F >= 10 ms and every Jacobian product >= 50 ms: R1 keeps the first
  // factorization (predicted cost T_a < T_F), and the first lagged solve
  // fires R2 (E_s >= T_F).
  void
  make_lagged_solves_expensive(ToySystem &system)
  {
    system.factorize_delay = std::chrono::milliseconds(10);
    system.jacobian_delay  = std::chrono::milliseconds(50);
  }

  // Mode ida: GMRES stops at ||b - Jx||_2 <= the tolerance passed, so the
  // scenario tests below can prescribe the accuracy of every solve.
  IDALaggingParameters
  parameters(const std::string &policy)
  {
    IDALaggingParameters result;
    result.policy               = policy;
    result.gmres_tolerance_mode = "ida";
    return result;
  }

  // One IDA step: a residual at t, an optional Jacobian setup, and solves.
  void
  step(Policy            &policy,
       ToySystem         &system,
       const double       t,
       const double       alpha,
       const bool         setup,
       const unsigned int n_solves,
       const double       relative_tolerance,
       const double       max_relative_error)
  {
    const auto y = system.vector(1. + t);
    const auto r = system.vector(-1.);
    system.state = y;
    policy.residual(t, y, y);
    if (setup)
      policy.setup_jacobian(t, y, y, alpha);
    for (unsigned int k = 0; k < n_solves; ++k)
      {
        Vector z(r);
        policy.solve(r, z, relative_tolerance * r.l2_norm());
        EXPECT_LE(system.residual_norm(z, r), max_relative_error * r.l2_norm());
      }
  }

  std::string
  log_prefix(const std::string &name)
  {
    const auto directory = TestPaths::output_path("ida_lagging_policy_01");
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::filesystem::create_directories(directory);
    return (directory / (name + "_np" +
                         std::to_string(dealii::Utilities::MPI::n_mpi_processes(
                           MPI_COMM_WORLD))))
      .string();
  }

  // Data rows of a log, split into whitespace-separated fields.
  std::vector<std::vector<std::string>>
  rows(const std::string &path)
  {
    std::ifstream file(path);
    EXPECT_TRUE(file) << path;
    std::vector<std::vector<std::string>> result;
    for (std::string line; std::getline(file, line);)
      if (!line.empty() && line[0] != '#')
        {
          std::istringstream       fields(line);
          std::vector<std::string> row;
          for (std::string field; fields >> field;)
            row.push_back(field);
          result.push_back(row);
        }
    return result;
  }
} // namespace

TEST(IDALaggingPolicy, BOTH_RejectsOffAndMissingOperations)
{
  ToySystem system;
  EXPECT_THROW(
    Policy(parameters("off"), {}, MPI_COMM_WORLD, system.operations()),
    dealii::ExceptionBase);
  auto ops         = system.operations();
  ops.direct_solve = {};
  EXPECT_THROW(Policy(parameters("direct"), {}, MPI_COMM_WORLD, ops),
               dealii::ExceptionBase);
}

TEST(IDALaggingPolicy, BOTH_DirectFactorizesEveryJacobian)
{
  ToySystem system;
  Policy policy(parameters("direct"), {}, MPI_COMM_WORLD, system.operations());
  policy.start();
  for (unsigned int s = 0; s < 5; ++s)
    step(policy, system, .1 * s, 1. + s, true, 2, 1.e-10, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.jacobian_evaluations, 5u);
  EXPECT_EQ(totals.factorizations, 5u);
  EXPECT_EQ(totals.direct_solves, 10u);
  EXPECT_EQ(totals.lagged_solves, 0u);
}

TEST(IDALaggingPolicy, BOTH_LagAlwaysKeepsTheFirstFactorization)
{
  ToySystem system;
  Policy    policy(parameters("lag_always"),
                   {},
                MPI_COMM_WORLD,
                system.operations());
  policy.start();
  for (unsigned int s = 0; s < 5; ++s)
    step(policy, system, .1 * s, 1. + .05 * s, true, 2, 1.e-10, 1.e-9);
  policy.finish();

  // The first segment is direct; every later solve is a converged GMRES
  // solve on the current Jacobian with the first factorization.
  const auto &totals = policy.totals();
  EXPECT_EQ(totals.jacobian_evaluations, 5u);
  EXPECT_EQ(totals.factorizations, 1u);
  EXPECT_EQ(totals.direct_solves, 2u);
  EXPECT_EQ(totals.lagged_solves, 8u);
  EXPECT_EQ(totals.fired_R3, 0u);
  EXPECT_GT(totals.gmres_iterations, totals.lagged_solves);
}

TEST(IDALaggingPolicy, BOTH_CapForcesRefactorizationAndDirectSolve)
{
  ToySystem system;
  auto      knobs              = parameters("lag_always");
  knobs.absolute_iteration_cap = 1;
  Policy policy(knobs, {}, MPI_COMM_WORLD, system.operations());
  policy.start();
  step(policy, system, 0., 1., true, 1, 1.e-12, 1.e-10);
  // A large alpha jump: one GMRES iteration cannot reach the tolerance.
  step(policy, system, .1, 50., true, 2, 1.e-12, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.factorizations, 2u);
  EXPECT_EQ(totals.fired_R3, 1u);
  EXPECT_EQ(totals.lagged_solves, 1u);
  // The abandoned solve is redone directly, and so is the next one.
  EXPECT_EQ(totals.direct_solves, 3u);
}

TEST(IDALaggingPolicy, BOTH_ExternalFactorizationStartsANewEpisode)
{
  ToySystem system;
  Policy    policy(parameters("lag_always"),
                   {},
                MPI_COMM_WORLD,
                system.operations());
  policy.start();
  step(policy, system, 0., 1., true, 1, 1.e-12, 1.e-10);
  step(policy, system, .1, 1.05, true, 1, 1.e-12, 1.e-10);
  // The caller factorized another matrix: lag_always must not keep using it
  // as the seed, so the next Jacobian is factorized and solved directly.
  policy.reset_factorization();
  step(policy, system, .2, 1.1, true, 1, 1.e-12, 1.e-10);
  step(policy, system, .3, 1.15, true, 1, 1.e-12, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.factorizations, 2u);
  EXPECT_EQ(totals.direct_solves, 2u);
  EXPECT_EQ(totals.lagged_solves, 2u);
}

TEST(IDALaggingPolicy, BOTH_RebuildEverySolveFollowsTheLatestIterate)
{
  ToySystem system;
  auto      knobs                       = parameters("direct");
  knobs.operator_rebuild_every_n_solves = 1;
  Policy policy(knobs, {}, MPI_COMM_WORLD, system.operations());
  policy.start();
  // IDA calls setup_jacobian once; every later Newton iterate rebuilds J at
  // the state of the latest residual. The solve check uses that J.
  step(policy, system, .1, 2., true, 1, 1.e-12, 1.e-10);
  for (unsigned int k = 1; k < 4; ++k)
    step(policy, system, .1 + k * .01, 2., false, 1, 1.e-12, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.jacobian_evaluations, 4u);
  EXPECT_EQ(totals.factorizations, 4u);
  EXPECT_EQ(totals.direct_solves, 4u);
}

TEST(IDALaggingPolicy, BOTH_DeferredR2FactorizesTheNextJacobian)
{
  ToySystem system;
  make_lagged_solves_expensive(system);
  auto knobs                            = parameters("adaptive");
  knobs.operator_rebuild_every_n_solves = 1;
  Policy policy(knobs, {}, MPI_COMM_WORLD, system.operations());
  policy.start();
  step(policy, system, 0., 1., true, 1, 1.e-12, 1.e-10);
  // Standard Newton: every solve has its own Jacobian. R2 fires after the
  // lagged solve and factorizes the Jacobian of the next solve.
  step(policy, system, .01, 1., false, 1, 1.e-12, 1.e-10);
  step(policy, system, .02, 1., false, 1, 1.e-12, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.fired_R2, 1u);
  EXPECT_EQ(totals.deferred_new_jacobian, 1u);
  EXPECT_EQ(totals.deferred_same_jacobian, 0u);
  EXPECT_EQ(totals.factorizations, 2u);
  EXPECT_EQ(totals.lagged_solves, 1u);
  EXPECT_EQ(totals.direct_solves, 2u);
}

TEST(IDALaggingPolicy, BOTH_DeferredR2WithinASegment)
{
  ToySystem system;
  make_lagged_solves_expensive(system);
  Policy policy(parameters("adaptive"),
                {},
                MPI_COMM_WORLD,
                system.operations());
  policy.start();
  step(policy, system, 0., 1., true, 1, 1.e-12, 1.e-10);
  // Modified Newton: the second solve of the segment reuses J_s, which is
  // factorized before that solve.
  step(policy, system, .01, 1.01, true, 2, 1.e-12, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.fired_R2, 1u);
  EXPECT_EQ(totals.deferred_new_jacobian, 0u);
  EXPECT_EQ(totals.deferred_same_jacobian, 1u);
  EXPECT_EQ(totals.factorizations, 2u);
  EXPECT_EQ(totals.lagged_solves, 1u);
  EXPECT_EQ(totals.direct_solves, 2u);
}

TEST(IDALaggingPolicy, BOTH_DeferredR2IsDroppedAtTheEnd)
{
  ToySystem system;
  make_lagged_solves_expensive(system);
  Policy policy(parameters("adaptive"),
                {},
                MPI_COMM_WORLD,
                system.operations());
  policy.start();
  step(policy, system, 0., 1., true, 1, 1.e-12, 1.e-10);
  step(policy, system, .01, 1.01, true, 1, 1.e-12, 1.e-10);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.fired_R2, 1u);
  EXPECT_EQ(totals.deferred_dropped, 1u);
  EXPECT_EQ(totals.factorizations, 1u);
  EXPECT_EQ(totals.lagged_solves, 1u);
}

TEST(IDALaggingPolicy, BOTH_R1RefactorizesBeforeTheNextSolve)
{
  ToySystem system;
  system.factorize_delay = std::chrono::milliseconds(200);
  system.jacobian_delay  = std::chrono::milliseconds(30);
  Policy policy(parameters("adaptive"),
                {},
                MPI_COMM_WORLD,
                system.operations());
  policy.start();
  // 20 direct solves: the episode average A = T_F / 20 stays below the cost
  // of one GMRES iteration (>= 30 ms), but above T_a, so R1 keeps the
  // factorization at the next Jacobian.
  step(policy, system, 0., 1., true, 20, 1.e-12, 1.e-10);
  // The first lagged solve needs a few iterations, below the R2 budget
  // T_F >= 200 ms. Before the second solve, its predicted cost (at least one
  // iteration) exceeds A, so R1 refactorizes J_s and that solve is direct.
  step(policy, system, .01, 1.01, true, 2, 1.e-4, 1.e-3);
  policy.finish();

  const auto &totals = policy.totals();
  EXPECT_EQ(totals.keep_R1, 1u);
  EXPECT_EQ(totals.refact_R1_solve, 1u);
  EXPECT_EQ(totals.fired_R2, 0u);
  EXPECT_EQ(totals.factorizations, 2u);
  EXPECT_EQ(totals.lagged_solves, 1u);
  EXPECT_EQ(totals.direct_solves, 21u);
}

// ida_forcing: from IDA's tolerance tol = nrmfac eps_L / wbar, the policy
// recovers eps_L and asks GMRES for ||b - Jz||_2 <= eta ||b||_2 with
// eta = sqrt(N) eps_L / ||W M^-1 b||_2. The result must not depend on
// ls_norm_factor.
TEST(IDALaggingPolicy, BOTH_ForcingTermFollowsIDAsLinearTolerance)
{
  const double       eps_L = 1.e-3;
  const unsigned int N     = ToySystem::n;
  unsigned int       iterations[2];
  for (const unsigned int variant : {0u, 1u})
    {
      ToySystem    system;
      const double nrmfac        = variant == 0 ? 1. : std::sqrt(double(N));
      auto         knobs         = parameters("lag_always");
      knobs.gmres_tolerance_mode = "ida_forcing";
      Policy::IDATolerance tolerance;
      tolerance.ls_norm_factor            = variant == 0 ? 1. : 0.;
      tolerance.expected_linear_tolerance = eps_L;
      Policy policy(knobs, tolerance, MPI_COMM_WORLD, system.operations());
      policy.start();
      for (unsigned int s = 0; s < 4; ++s)
        {
          const double t = .1 * s;
          const auto   y = system.vector(1. + t);
          const auto   r = system.vector(-1.);
          system.state   = y;
          policy.residual(t, y, y);
          policy.setup_jacobian(t, y, y, 1. + .05 * s);

          Vector w;
          system.weights(w);
          const double wbar = w.l2_norm() / std::sqrt(double(N));
          Vector       z(r);
          // The factorization before the solve is the one GMRES uses.
          Vector Mr(r);
          if (s > 0)
            system.direct->solve(Mr, r);
          policy.solve(r, z, nrmfac * eps_L / wbar);
          if (s > 0)
            {
              Mr.scale(w);
              const double eta = std::sqrt(double(N)) * eps_L / Mr.l2_norm();
              ASSERT_LT(eta, 1.);
              EXPECT_LE(system.residual_norm(z, r),
                        eta * r.l2_norm() * (1. + 1.e-6));
            }
        }
      policy.finish();
      const auto &totals = policy.totals();
      EXPECT_EQ(totals.factorizations, 1u);
      EXPECT_EQ(totals.lagged_solves, 3u);
      EXPECT_NEAR(totals.eps_ratio_min, 1., 1.e-12);
      EXPECT_NEAR(totals.eps_ratio_max, 1., 1.e-12);
      iterations[variant] = totals.gmres_iterations;
    }
  EXPECT_GT(iterations[0], 0u);
  EXPECT_EQ(iterations[0], iterations[1]);
}

// ida_left meets IDA's linear test exactly, ||W M^{-1}(Jz - r)||_WRMS <=
// eps_L with the factorization GMRES used, also when half of the rows are
// 1e6 times smaller than the corrections they produce. The 2-norm of the
// residual, which ida_forcing reduces, hardly sees those rows.
TEST(IDALaggingPolicy, BOTH_LeftModeMeetsIDAsLinearTest)
{
  const double       eps_L = 1.e-3;
  const unsigned int N     = ToySystem::n;
  ToySystem          system;
  system.row_scale           = 1.e-6;
  auto knobs                 = parameters("lag_always");
  knobs.gmres_tolerance_mode = "ida_left";
  Policy::IDATolerance tolerance;
  tolerance.ls_norm_factor            = 1.;
  tolerance.expected_linear_tolerance = eps_L;
  Policy policy(knobs, tolerance, MPI_COMM_WORLD, system.operations());
  policy.start();
  for (unsigned int s = 0; s < 4; ++s)
    {
      const double t = .1 * s;
      const auto   y = system.vector(1. + t);
      auto         r = system.vector(-1.);
      for (const auto i : system.owned)
        if (i >= N / 2)
          r(i) *= system.row_scale;
      r.compress(dealii::VectorOperation::insert);
      system.state = y;
      policy.residual(t, y, y);
      policy.setup_jacobian(t, y, y, 1. + .05 * s);

      Vector w;
      system.weights(w);
      const double wbar = w.l2_norm() / std::sqrt(double(N));
      Vector       z(r);
      policy.solve(r, z, eps_L / wbar);
      if (s > 0)
        {
          Vector d(r), Md(r);
          system.jacobian.vmult(d, z);
          d -= r;
          system.direct->solve(Md, d);
          Md.scale(w);
          EXPECT_LE(Md.l2_norm() / std::sqrt(double(N)), eps_L * (1. + 1.e-6));
        }
    }
  policy.finish();
  const auto &totals = policy.totals();
  EXPECT_EQ(totals.factorizations, 1u);
  EXPECT_EQ(totals.lagged_solves, 3u);
  EXPECT_GT(totals.gmres_iterations, 0u);
  EXPECT_NEAR(totals.eps_ratio_min, 1., 1.e-12);
}

// ida_forcing and ida_left with eta >= 1, i.e. IDA's test already holds at
// x = 0: a current factorization still returns M^{-1} b = J^{-1} b; a lagged
// one returns 0 or M^{-1} b, as "No-iteration correction" selects, without
// GMRES iterations.
TEST(IDALaggingPolicy, BOTH_ForcingTermAboveOneSkipsGMRES)
{
  for (const std::string mode : {"ida_forcing", "ida_left"})
    for (const std::string correction : {"zero", "preconditioned_rhs"})
      {
        SCOPED_TRACE(mode + ", " + correction);
        ToySystem system;
        auto      knobs               = parameters("lag_always");
        knobs.gmres_tolerance_mode    = mode;
        knobs.no_iteration_correction = correction;
        Policy policy(knobs, {}, MPI_COMM_WORLD, system.operations());
        policy.start();
        const auto r = system.vector(-1.);
        for (unsigned int s = 0; s < 4; ++s)
          {
            const auto y = system.vector(1. + .1 * s);
            system.state = y;
            policy.residual(.1 * s, y, y);
            policy.setup_jacobian(.1 * s, y, y, 1. + .05 * s);
            // At s = 0 the second solve has a factorization (M = J) when it
            // computes M^{-1} b.
            for (unsigned int k = 0; k < (s == 0 ? 2u : 1u); ++k)
              {
                Vector z(r);
                policy.solve(r, z, 1.e8);
                if (s > 0 && correction == "zero")
                  EXPECT_EQ(z.l2_norm(), 0.);
                else
                  {
                    // The toy system holds a factorization only after the
                    // policy's first one; it is M, current at s = 0.
                    Vector Mr(r);
                    system.direct->solve(Mr, r);
                    z -= Mr;
                    EXPECT_LE(z.l2_norm(), 1.e-14 * Mr.l2_norm());
                  }
              }
          }
        policy.finish();
        EXPECT_EQ(policy.totals().factorizations, 1u);
        EXPECT_EQ(policy.totals().direct_solves, 2u);
        EXPECT_EQ(policy.totals().lagged_solves, 3u);
        EXPECT_EQ(policy.totals().gmres_iterations, 0u);
      }
}

// Each state drift is measured from its own seed with that seed's final
// weights. After an external factorization the carried factor comes from an
// earlier seed: the prediction then uses Sigma_l in the old seed's weights,
// and logs the same drift in the new seed's weights as a diagnostic.
TEST(IDALaggingPolicy, BOTH_DriftUsesEachSeedsOwnWeights)
{
  ToySystem system;
  auto      knobs            = parameters("lag_always");
  knobs.gmres_tolerance_mode = "fixed";
  knobs.log_prefix           = log_prefix("drift_seed_weights");
  Policy policy(knobs, {}, MPI_COMM_WORLD, system.operations());
  policy.start();
  // Constant alpha: theta_D = 0, so the whole pooled factor is the state
  // part. Segment 1 seeds at y(0); segment 2 lags and carries its factor;
  // segment 3 seeds again at y(0.2) after the external factorization.
  for (unsigned int s = 0; s < 5; ++s)
    {
      if (s == 2)
        policy.reset_factorization();
      // IDA's current weights change (a ramp); the drift norm must not.
      system.weight_scale = s >= 3 ? 10. : 1.;
      step(policy, system, .1 * s, 1., true, 1, 1., 1.e-6);
    }
  policy.finish();

  // Independent oracle: the toy states are y(t) = vector(1 + t).
  const auto y     = [&](const double t) { return system.vector(1. + t); };
  const auto drift = [&](const double t, const double t_seed) {
    Vector d = y(t);
    d -= y(t_seed);
    return d;
  };
  const double sigma_1     = system.wrms(drift(.1, 0.), y(0.));
  const double sigma_3     = system.wrms(drift(.3, .2), y(.2));
  const double sigma_4     = system.wrms(drift(.4, .2), y(.2));
  const double sigma_1_cur = system.wrms(drift(.1, 0.), y(.2));
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
    return;
  const auto segments = rows(knobs.log_prefix + "_segments.log");
  ASSERT_EQ(segments.size(), 5u);
  enum Column
  {
    sigma          = 4,
    decision       = 6,
    theta_pred     = 23,
    sigma_l        = 24,
    sigma_l_cur    = 25,
    theta_pred_cur = 26,
    cross          = 27
  };
  const auto value = [&](const unsigned int row, const Column column) {
    return std::stod(segments[row][column]);
  };
  EXPECT_EQ(segments[2][decision], "first");
  EXPECT_NEAR(value(1, sigma), sigma_1, 1.e-3 * sigma_1);
  EXPECT_NEAR(value(3, sigma), sigma_3, 1.e-3 * sigma_3);
  EXPECT_NEAR(value(4, sigma), sigma_4, 1.e-3 * sigma_4);

  // Segment 4: factor from segment 2, measured from the first seed.
  EXPECT_EQ(segments[3][cross], "1");
  EXPECT_NEAR(value(3, sigma_l), sigma_1, 1.e-3 * sigma_1);
  EXPECT_NEAR(value(3, sigma_l_cur), sigma_1_cur, 1.e-3 * sigma_1_cur);
  EXPECT_GT(std::abs(sigma_1_cur / sigma_1 - 1.), 1.e-2);
  EXPECT_NEAR(value(3, theta_pred) / value(3, theta_pred_cur),
              sigma_1_cur / sigma_1,
              2.e-3 * sigma_1_cur / sigma_1);

  // Segment 5: factor from segment 4, same seed: both versions agree.
  EXPECT_EQ(segments[4][cross], "0");
  EXPECT_NEAR(value(4, sigma_l), sigma_3, 1.e-3 * sigma_3);
  EXPECT_NEAR(value(4, sigma_l_cur), sigma_3, 1.e-3 * sigma_3);
  EXPECT_NEAR(value(4, theta_pred),
              value(4, theta_pred_cur),
              1.e-3 * value(4, theta_pred));
}

TEST(IDALaggingPolicy, BOTH_AdaptiveWritesOneRowPerSolveAndSegment)
{
  ToySystem system;
  auto      knobs  = parameters("adaptive");
  knobs.log_prefix = log_prefix("adaptive");
  Policy policy(knobs, {}, MPI_COMM_WORLD, system.operations());
  policy.start("IDA: test header");
  for (unsigned int s = 0; s < 4; ++s)
    step(policy, system, .1 * s, 1. + .05 * s, true, 3, 1.e-10, 1.e-9);
  policy.finish("test statistics");

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      const std::string prefix = knobs.log_prefix;
      EXPECT_EQ(rows(prefix + "_solves.log").size(), 12u);
      EXPECT_EQ(rows(prefix + "_segments.log").size(), 4u);
      EXPECT_EQ(rows(prefix + "_episodes.log").size(),
                policy.totals().factorizations);
      std::ifstream     summary(prefix + "_summary.log");
      const std::string text((std::istreambuf_iterator<char>(summary)),
                             std::istreambuf_iterator<char>());
      EXPECT_NE(text.find("IDA: test header"), std::string::npos);
      EXPECT_NE(text.find("Policy                          = adaptive"),
                std::string::npos);
      EXPECT_NE(text.find("factorizations        "), std::string::npos);
      EXPECT_NE(text.find("test statistics"), std::string::npos);
    }
}
#endif
