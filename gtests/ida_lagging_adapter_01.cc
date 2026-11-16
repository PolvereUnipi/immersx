// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/config.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <gtest/gtest.h>
#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/sundials_ida_adapter.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "test_paths.h"

#if defined(DEAL_II_WITH_SUNDIALS) && defined(DEAL_II_WITH_MUMPS) && \
  !defined(IMMERSX_USE_PETSC_LA)

namespace
{
  using namespace ImmersX;
  using FieldVector  = ImmersXLA::MPI::Vector;
  using GlobalVector = ImmersXLA::MPI::BlockVector;
  using Adapter      = IDAAdapter<FieldVector, GlobalVector>;

  struct Run
  {
    Adapter::Statistics statistics;
    std::vector<double> solution;
    double              max_error = 0.;
    std::string         summary;
  };

  // y' + K y = 0 with K = 2 I - P, P a cyclic shift across all ranks. Every
  // row of K sums to one, so y(0) = 1 decays as exp(-t). IDA's step size
  // changes alpha from setup to setup, which is the drift a lagged
  // factorization has to absorb.
  Run
  run(const IDALaggingParameters &lagging)
  {
    TimeIntervalParameters time;
    IDAParameters          parameters;
    time.initial_time                            = 0.;
    time.final_time                              = 1.;
    time.output_time_interval                    = .5;
    parameters.initial_step_size                 = 1.e-3;
    parameters.maximum_order                     = 2;
    parameters.absolute_tolerance                = 1.e-8;
    parameters.relative_tolerance                = 1.e-6;
    parameters.ignore_algebraic_terms_for_errors = false;
    parameters.lagging                           = lagging;

    const auto communicator = MPI_COMM_WORLD;
    const auto size =
      10 * dealii::Utilities::MPI::n_mpi_processes(communicator);
    const auto owned =
      dealii::Utilities::MPI::create_evenly_distributed_partitioning(
        communicator, size);
    Adapter                        adapter(time, parameters, communicator);
    dealii::DynamicSparsityPattern sparsity(size, size, owned);
    for (const auto i : owned)
      {
        sparsity.add(i, i);
        sparsity.add(i, (i + 1) % size);
      }
    auto identity  = std::make_shared<ImmersXLA::MPI::SparseMatrix>();
    auto stiffness = std::make_shared<ImmersXLA::MPI::SparseMatrix>();
    identity->reinit(owned, owned, sparsity, communicator);
    stiffness->reinit(owned, owned, sparsity, communicator);
    for (const auto i : owned)
      {
        identity->set(i, i, 1.);
        stiffness->set(i, i, 2.);
        stiffness->set(i, (i + 1) % size, -1.);
      }
    identity->compress(dealii::VectorOperation::insert);
    stiffness->compress(dealii::VectorOperation::insert);
    const auto field = adapter.add(
      [owned, identity, stiffness](auto &builder) {
        using Model        = typename std::decay_t<decltype(builder)>::Model;
        const auto value   = builder.differential_field("value", owned);
        const auto mass    = builder.matrix_operator(*identity);
        const auto spatial = builder.matrix_operator(*stiffness);
        builder.term(value, "coupled decay")
          .residual([value, stiffness](const auto &context) {
            const auto               &state     = context.state(value);
            const auto               &state_dot = context.derivative(value);
            typename Model::Operation result;
            result.reinit_vector = [state](FieldVector &vector, bool) {
              vector.reinit(state);
            };
            result.apply = [state, state_dot, stiffness](FieldVector &vector) {
              stiffness->vmult(vector, state);
              vector += state_dot;
            };
            result.apply_add =
              [state, state_dot, stiffness](FieldVector &vector) {
                FieldVector contribution;
                contribution.reinit(state);
                stiffness->vmult(contribution, state);
                contribution += state_dot;
                vector += contribution;
              };
            return result;
          })
          .state(value, spatial)
          .derivative(value, mass);
        return value;
      },
      "coupled decay");
    adapter.set_linear_solver(Adapter::LinearSolver::mumps);
    adapter.set_stop_at_output_times(true);
    auto state = adapter.make_state();
    auto dot   = adapter.make_state();
    state      = 1.;
    dot        = -1.;
    EXPECT_GT(adapter.solve(state, dot), 0u);

    Run result;
    result.statistics     = adapter.statistics();
    const auto  &solution = adapter.field(state, field.fields());
    const double expected = std::exp(-time.final_time);
    for (const auto i : owned)
      {
        result.solution.push_back(solution[i]);
        result.max_error =
          std::max(result.max_error, std::abs(solution[i] - expected));
      }
    result.max_error =
      dealii::Utilities::MPI::max(result.max_error, communicator);
    if (!lagging.log_prefix.empty() &&
        dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        std::ifstream file(lagging.log_prefix + "_summary.log");
        EXPECT_TRUE(file) << lagging.log_prefix;
        result.summary.assign(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
      }
    return result;
  }

  IDALaggingParameters
  policy(const std::string &name, const unsigned int rebuild = 0)
  {
    IDALaggingParameters result;
    result.policy                          = name;
    result.operator_rebuild_every_n_solves = rebuild;
    const auto directory = TestPaths::output_path("ida_lagging_adapter_01");
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::filesystem::create_directories(directory);
    result.log_prefix =
      (directory / (name + "_rebuild" + std::to_string(rebuild) + "_np" +
                    std::to_string(
                      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))))
        .string();
    return result;
  }

  // The number after a key of the summary, e.g. "lagged_solves  12".
  double
  summary_value(const std::string &summary, const std::string &key)
  {
    std::istringstream lines(summary);
    for (std::string line; std::getline(lines, line);)
      if (line.rfind(key, 0) == 0)
        return std::stod(line.substr(key.size()));
    ADD_FAILURE() << "no " << key << " in the summary";
    return 0.;
  }

  // "eps_L / expected      min a, max b": a and b.
  std::pair<double, double>
  eps_ratio(const std::string &summary)
  {
    const std::string key = "eps_L / expected      min ";
    const auto        at  = summary.find(key);
    if (at == std::string::npos)
      {
        ADD_FAILURE() << "no eps_L check in the summary";
        return {0., 0.};
      }
    std::istringstream values(summary.substr(at + key.size()));
    double             min = 0., max = 0.;
    std::string        comma, word;
    values >> min >> comma >> word >> max;
    return {min, max};
  }
} // namespace

// The policy direct factorizes every Jacobian at its first solve and solves
// directly: IDA must take exactly the path of the plain direct solver.
TEST(IDALaggingAdapter, BOTH_DirectPolicyReproducesThePlainDirectSolver)
{
  const auto off    = run(IDALaggingParameters());
  const auto direct = run(policy("direct"));
  EXPECT_EQ(off.statistics.steps, direct.statistics.steps);
  EXPECT_EQ(off.statistics.nonlinear_iterations,
            direct.statistics.nonlinear_iterations);
  EXPECT_EQ(off.statistics.jacobian_setups, direct.statistics.jacobian_setups);
  EXPECT_EQ(off.statistics.linear_solves, direct.statistics.linear_solves);
  ASSERT_EQ(off.solution.size(), direct.solution.size());
  for (std::size_t i = 0; i < off.solution.size(); ++i)
    EXPECT_NEAR(off.solution[i], direct.solution[i], 1.e-13);
  EXPECT_LT(off.max_error, 1.e-4);
}

// Lagged GMRES solves on the matrix-free Jacobian, preconditioned by an older
// MUMPS factorization, under modified and standard Newton. The forcing term
// recovers IDA's linear tolerance eplifac * epcon from the tolerance IDA
// passes.
TEST(IDALaggingAdapter, BOTH_LaggedSolvesFollowIDAsTolerance)
{
  const unsigned int ranks =
    dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const bool root =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  // Inexact lagged solves must not cost accuracy against the direct solver.
  const auto reference = run(IDALaggingParameters());
  for (const std::string name : {"lag_always", "adaptive"})
    for (const unsigned int rebuild : {0u, 1u})
      {
        SCOPED_TRACE(name + ", rebuild " + std::to_string(rebuild) + ", " +
                     std::to_string(ranks) + " rank(s)");
        const auto result = run(policy(name, rebuild));
        EXPECT_LT(result.max_error, 10. * reference.max_error + 1.e-7);
        if (!root)
          continue;
        const auto [min, max] = eps_ratio(result.summary);
        EXPECT_NEAR(min, 1., 1.e-3);
        EXPECT_NEAR(max, 1., 1.e-3);
        const double evaluations =
          summary_value(result.summary, "jacobian_evaluations");
        const double factorizations =
          summary_value(result.summary, "factorizations");
        const double lagged = summary_value(result.summary, "lagged_solves");
        EXPECT_GT(evaluations, 1.);
        if (name == "lag_always")
          {
            EXPECT_GT(lagged, 0.);
            EXPECT_LT(factorizations, evaluations);
          }
      }
}

#endif
