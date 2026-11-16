// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------
#ifndef immersx_ida_lagging_parameters_h
#define immersx_ida_lagging_parameters_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <string>

namespace ImmersX
{
  /** Controls for reusing a direct factorization as a GMRES preconditioner
   * across IDA Jacobian setups. Names and defaults follow MetricFlowX's
   * blood_flow_IDA_lagging driver. Differences: `off` keeps IDAAdapter's
   * plain direct solve, the prediction always uses the drift model with the
   * pooled factor, and an empty log prefix writes no log files.
   */
  struct IDALaggingParameters
  {
    std::string  policy                          = "off";
    unsigned int operator_rebuild_every_n_solves = 0;
    std::string  gmres_tolerance_mode            = "ida_forcing";
    double       gmres_fixed_relative_tolerance  = 1.e-8;
    unsigned int absolute_iteration_cap          = 20;
    std::string  probe_rule                      = "alpha_check";
    unsigned int probe_interval_min              = 1;
    unsigned int probe_interval_max              = 8;
    double       moving_average_weight           = .2;
    std::string  log_prefix;

    bool
    enabled() const
    {
      return policy != "off";
    }

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Lagging policy");
      prm.add_parameter(
        "Policy",
        policy,
        "off: factorize every Jacobian setup, without the policy | direct: "
        "the same through the policy, with its logs | adaptive: cost-model "
        "rules R1-R4 | lag_always: keep the factorization until GMRES reaches "
        "the iteration cap",
        dealii::Patterns::Selection("off|direct|adaptive|lag_always"));
      prm.add_parameter(
        "Operator rebuild every n solves",
        operator_rebuild_every_n_solves,
        "0: IDA decides the Jacobian setups (modified Newton) | 1: rebuild "
        "the Jacobian before every linear solve (standard Newton) | n: "
        "rebuild every n solves. Each rebuild assembles the linearized terms.");
      prm.add_parameter(
        "GMRES tolerance mode",
        gmres_tolerance_mode,
        "ida_forcing: ||b-Jx||_2 <= eta ||b||_2 with IDA's forcing term eta = "
        "sqrt(N) eps_L / ||W M^-1 b||_2, eps_L = eplifac * epsNewt recovered "
        "from the tolerance IDA passes | ida: ||b-Jx||_2 <= the tolerance IDA "
        "passes (over-solves) | fixed: ||b-Jx||_2 <= tau ||b||_2",
        dealii::Patterns::Selection("ida_forcing|ida|fixed"));
      prm.add_parameter("GMRES fixed relative tolerance",
                        gmres_fixed_relative_tolerance,
                        "tau of the fixed mode.",
                        dealii::Patterns::Double(0.));
      prm.add_parameter(
        "Absolute iteration cap",
        absolute_iteration_cap,
        "I_abs: maximum GMRES iterations of every lagged solve.",
        dealii::Patterns::Integer(1));
      prm.add_parameter(
        "Probe rule",
        probe_rule,
        "always: probe (R4) whenever the backoff allows | alpha_check: probe "
        "only if R1 would keep the factorization with the alpha part of the "
        "predicted factor alone",
        dealii::Patterns::Selection("always|alpha_check"));
      prm.add_parameter("Probe interval min",
                        probe_interval_min,
                        "b_min of R4: refactorizations before the first probe.",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("Probe interval max",
                        probe_interval_max,
                        "b_max of R4: largest probe interval reached by "
                        "doubling.",
                        dealii::Patterns::Integer(1));
      prm.add_parameter("Moving average weight",
                        moving_average_weight,
                        "Weight of a new sample in the factorization, direct "
                        "solve and GMRES iteration time averages.",
                        dealii::Patterns::Double(0., 1.));
      prm.add_parameter(
        "Log prefix",
        log_prefix,
        "Write <prefix>_{solves,segments,episodes,summary}.log; empty writes "
        "no logs.");
      prm.leave_subsection();
    }

    /** Check that the entries are consistent with each other. */
    void
    validate() const
    {
      AssertThrow(probe_interval_min <= probe_interval_max,
                  dealii::ExcMessage(
                    "Lagging policy: Probe interval min must not exceed "
                    "Probe interval max."));
      AssertThrow(moving_average_weight > 0.,
                  dealii::ExcMessage(
                    "Lagging policy: Moving average weight must be positive."));
      AssertThrow(gmres_tolerance_mode != "fixed" ||
                    gmres_fixed_relative_tolerance > 0.,
                  dealii::ExcMessage(
                    "Lagging policy: the fixed GMRES tolerance mode needs a "
                    "positive relative tolerance."));
    }
  };
} // namespace ImmersX
#endif
