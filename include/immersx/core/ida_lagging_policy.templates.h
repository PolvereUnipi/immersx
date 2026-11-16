// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------
#ifndef immersx_ida_lagging_policy_templates_h
#define immersx_ida_lagging_policy_templates_h

#include <deal.II/base/mpi.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <immersx/core/ida_lagging_policy.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace ImmersX
{
  namespace internal
  {
    // Adapters from the policy's callbacks to the matrix and preconditioner
    // interface of deal.II's solvers.
    template <typename VectorType>
    struct LaggingOperator
    {
      const std::function<void(VectorType &, const VectorType &)> &apply;
      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        apply(dst, src);
      }
    };
  } // namespace internal



  template <typename VectorType>
  IDALaggingPolicy<VectorType>::IDALaggingPolicy(
    const IDALaggingParameters &lagging_parameters,
    const IDATolerance         &ida_tolerance,
    const MPI_Comm              mpi_communicator,
    Operations                  callbacks)
    : parameters(lagging_parameters)
    , tolerance(ida_tolerance)
    , communicator(mpi_communicator)
    , operations(std::move(callbacks))
    , root(dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , pcout(std::cout, root)
  {
    AssertThrow(parameters.enabled(),
                dealii::ExcMessage("IDALaggingPolicy needs a policy other "
                                   "than off."));
    parameters.validate();
    AssertThrow(operations.build_jacobian && operations.factorize &&
                  operations.apply_factorization && operations.direct_solve &&
                  operations.apply_jacobian && operations.error_weights,
                dealii::ExcMessage(
                  "IDALaggingPolicy needs every operation callback."));
  }



  template <typename VectorType>
  template <typename F>
  double
  IDALaggingPolicy<VectorType>::timed(F &&f) const
  {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return dealii::Utilities::MPI::max(
      std::chrono::duration<double>(t1 - t0).count(), communicator);
  }



  // Weighted RMS norm ||d .* w||_2 / sqrt(N).
  template <typename VectorType>
  double
  IDALaggingPolicy<VectorType>::wrms(const VectorType &d, const VectorType &w)
  {
    VectorType tmp(d);
    tmp.scale(w);
    return tmp.l2_norm() / std::sqrt(static_cast<double>(d.size()));
  }



  template <typename VectorType>
  double
  IDALaggingPolicy<VectorType>::T_a() const
  {
    return cost.T_a.value;
  }



  // tau_it starts at T_a (optimistic) until a lagged solve is measured.
  template <typename VectorType>
  double
  IDALaggingPolicy<VectorType>::tau_it() const
  {
    return cost.tau_it.init ? cost.tau_it.value : cost.T_a.value;
  }



  template <typename VectorType>
  double
  IDALaggingPolicy<VectorType>::episode_average() const
  {
    return (cost.T_F.value + ep.E) / std::max(1u, ep.nu);
  }



  // Iterations needed to reduce the residual by eta at the factor theta: 1
  // without a factor (optimistic start), infinite if the factor shows
  // stagnation. Not capped: a count beyond the cap means the solve would be
  // abandoned by (R3).
  template <typename VectorType>
  double
  IDALaggingPolicy<VectorType>::predict_m(const double eta, const double theta)
  {
    if (eta >= 1.)
      return 0.;
    if (!(theta > 0.))
      return 1.;
    if (theta >= 1.)
      return std::numeric_limits<double>::infinity();
    return std::ceil(std::log(eta) / std::log(theta));
  }



  // GMRES rate of a spectrum filling the disk with diameter [1, r]: the
  // alpha drift alone puts the spectrum of J_s M^{-1} there.
  template <typename VectorType>
  double
  IDALaggingPolicy<VectorType>::theta_disk(const double r)
  {
    if (std::isinf(r))
      return 1.;
    return std::abs(1. - r) / (1. + r);
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::start(const std::string &header)
  {
    seg  = Segment();
    ep   = Episode();
    ns   = NewtonState();
    seed = Seed();
    cost = Costs();
    tot  = Totals();

    theta_last      = std::numeric_limits<double>::quiet_NaN();
    theta_pool      = std::numeric_limits<double>::quiet_NaN();
    r_last          = 1.;
    sigma_last_seed = 0.;
    // displacement_last is read only once theta_pool is set again.
    probe_c = 0;
    probe_b = parameters.probe_interval_min;
    pending_refactorization.clear();

    pcout << "IDALaggingPolicy: policy " << parameters.policy << std::endl;
    open_logs(header);
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::residual(const double      t,
                                         const VectorType &y,
                                         const VectorType &ydot)
  {
    ns.t                = t;
    ns.y                = y;
    ns.ydot             = ydot;
    ns.operator_current = false;
    if (std::isnan(ns.last_t) || t != ns.last_t)
      {
        ns.last_t    = t;
        ns.newton_it = 0;
        ++ns.step;
      }
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::setup_jacobian(const double      t,
                                               const VectorType &y,
                                               const VectorType &ydot,
                                               const double      alpha)
  {
    ns.alpha = alpha;
    build_operator(t, y, ydot, alpha);
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::build_operator(const double      t,
                                               const VectorType &y,
                                               const VectorType &ydot,
                                               const double      alpha)
  {
    tot.time_jacobian +=
      timed([&]() { operations.build_jacobian(t, y, ydot, alpha); });
    ++tot.jacobian_evaluations;
    ns.operator_current     = true;
    ns.new_operator         = true;
    ns.solves_since_rebuild = 0;
    ns.op_alpha             = alpha;
    ns.op_y                 = y;
    seed.current            = false;
  }



  // Factorize the current operator; it becomes the seed of a new episode.
  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::factorize(const std::string &reason)
  {
    if (seed.valid)
      close_episode(reason);

    const double dt = timed([&]() { operations.factorize(); });
    cost.T_F.add(dt, parameters.moving_average_weight);
    tot.time_factorize += dt;
    ++tot.factorizations;

    seed.valid   = true;
    seed.current = true;
    seed.alpha   = ns.op_alpha;
    seed.y       = ns.op_y;
    operations.error_weights(seed.w);

    ep         = Episode();
    ep.id      = tot.factorizations;
    ep.t_start = ns.t;
    ep.T_F     = dt;

    seg.E_s = 0.; // the decision on this segment is settled
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::close_episode(const std::string &reason)
  {
    if (root)
      log_episodes << std::setw(6) << ep.id << std::setw(14) << ep.t_start
                   << std::setw(14) << ep.T_F << std::setw(8) << ep.nu
                   << std::setw(8) << ep.n_lagged_segments << std::setw(14)
                   << ep.E << std::setw(14) << episode_average() << "  "
                   << reason << '\n';
  }



  // Decision at a Jacobian evaluation: (R1) and (R4).
  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::begin_segment(const double eta)
  {
    close_segment();

    const unsigned int id = seg.id + 1;
    seg                   = Segment();
    seg.id                = id;
    seg.t                 = ns.t;
    seg.alpha             = ns.op_alpha;
    ns.new_operator       = false;

    if (!seed.valid)
      {
        seg.decision = "first";
        factorize("-");
        return;
      }

    // Drift of the new operator with respect to the seed, in IDA's current
    // weights (and in the seed's, as the driver measures it).
    VectorType w;
    operations.error_weights(w);
    seg.displacement = ns.op_y;
    seg.displacement -= seed.y;
    seg.dalpha     = std::abs(seg.alpha - seed.alpha);
    seg.sigma      = wrms(seg.displacement, w);
    seg.sigma_seed = wrms(seg.displacement, seed.w);
    seg.euclid =
      seg.displacement.l2_norm() / std::max(seed.y.l2_norm(), 1e-300);

    // Predicted factor: the alpha part of this segment plus the state part
    // of the pooled factor of the last segment with lagged iterations,
    // scaled by the state drift. Both displacements are measured in the
    // same (current) weights, so the scaling of the norm cancels.
    seg.r       = seed.alpha != 0. ?
                    seg.alpha / seed.alpha :
                    (seg.alpha == 0. ? 1. : std::numeric_limits<double>::infinity());
    seg.theta_D = theta_disk(seg.r);
    if (theta_pool > 0.)
      {
        const double state_part = std::max(theta_pool - theta_disk(r_last), 0.);
        const double sigma_last = wrms(displacement_last, w);
        seg.theta_y =
          sigma_last > 0. ? state_part * seg.sigma / sigma_last : state_part;
        seg.theta_y_seed = sigma_last_seed > 0. ?
                             state_part * seg.sigma_seed / sigma_last_seed :
                             state_part;
      }
    seg.theta_pred = seg.theta_D + seg.theta_y;
    seg.m_hat      = predict_m(eta, seg.theta_pred);
    seg.E_hat      = seg.m_hat * tau_it();
    seg.A          = episode_average();
    seg.c          = probe_c;
    seg.b          = probe_b;

    bool keep = false;
    if (!pending_refactorization.empty())
      {
        seg.decision = "refact_R2";
        ++tot.deferred_new_jacobian;
      }
    else if (parameters.policy == "direct")
      seg.decision = "direct";
    else if (parameters.policy == "lag_always")
      {
        seg.decision = "lag_always";
        keep         = true;
      }
    else if (seg.E_hat < seg.A) // (R1)
      {
        seg.decision = "keep_R1";
        keep         = true;
        probe_c      = 0;
        probe_b      = parameters.probe_interval_min;
        ++tot.keep_R1;
      }
    else if (probe_c >= probe_b &&
             (parameters.probe_rule == "always" ||
              predict_m(eta, seg.theta_D) * tau_it() < seg.A)) // (R4)
      {
        seg.decision = "probe_R4";
        keep         = true;
        probe_c      = 0;
        probe_b      = std::min(2 * probe_b, parameters.probe_interval_max);
        ++tot.probes;
      }
    else
      {
        // With alpha_check, a due probe is skipped when the alpha drift
        // alone already makes lagging too expensive.
        seg.decision = probe_c >= probe_b ? "refact_noprobe" : "refact_R1";
        if (probe_c >= probe_b)
          ++tot.probes_skipped;
        ++probe_c;
        ++tot.refact_R1_eval;
      }

    if (keep)
      {
        seg.lagged = true;
        ++ep.n_lagged_segments;
      }
    else if (!pending_refactorization.empty())
      factorize(pending_refactorization);
    else
      factorize(parameters.policy == "direct" ? "direct" : "R1@eval");
    pending_refactorization.clear();
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::close_segment()
  {
    if (seg.id == 0)
      return;

    if (seg.refactorized && parameters.policy == "adaptive")
      {
        ++tot.bound_checks;
        if (seg.E_s >= seg.bound)
          ++tot.bound_violations;
      }

    if (root)
      {
        const double mean_m =
          seg.n_lagged > 0 ? double(seg.sum_m) / seg.n_lagged : 0.;
        log_segments << std::setw(6) << seg.id << std::setw(14) << seg.t
                     << std::setw(14) << seg.alpha << std::setw(14)
                     << seg.dalpha << std::setw(14) << seg.sigma
                     << std::setw(14) << seg.euclid << "  " << std::left
                     << std::setw(14) << seg.decision << std::right
                     << std::setw(10) << seg.m_hat << std::setw(14) << seg.E_hat
                     << std::setw(14) << seg.A << std::setw(4) << seg.c
                     << std::setw(4) << seg.b << std::setw(6) << seg.n_solves
                     << std::setw(6) << seg.n_lagged << std::setw(6)
                     << seg.first_m << std::setw(14) << seg.first_theta
                     << std::setw(14) << mean_m << std::setw(14) << seg.E_s
                     << std::setw(14) << (seg.refactorized ? seg.bound : 0.)
                     << "  " << std::left << std::setw(10) << seg.ended_by
                     << std::right << std::setw(14) << seg.r << std::setw(14)
                     << seg.theta_D << std::setw(14) << seg.theta_y
                     << std::setw(14) << seg.theta_pred << std::setw(14)
                     << seg.sigma_seed << std::setw(14) << seg.theta_y_seed
                     << '\n';
      }
  }



  // (R3): remaining segment budget in iterations, at most I_abs.
  template <typename VectorType>
  unsigned int
  IDALaggingPolicy<VectorType>::iteration_cap() const
  {
    if (parameters.policy != "adaptive")
      return parameters.absolute_iteration_cap;
    const double budget = (cost.T_F.value - seg.E_s) / tau_it();
    const double cap    = std::max(1., std::ceil(budget));
    return static_cast<unsigned int>(
      std::min(cap, static_cast<double>(parameters.absolute_iteration_cap)));
  }



  template <typename VectorType>
  typename IDALaggingPolicy<VectorType>::GmresSystem
  IDALaggingPolicy<VectorType>::gmres_system(const VectorType &b,
                                             const double      ida_tolerance)
  {
    GmresSystem sys;
    sys.r0 = b.l2_norm();

    // IDA hands a solver without scaling vectors
    // tol = nrmfac * eps_L / wbar, wbar the RMS of its error weights, so that
    // its test ||P^{-1}(b - Jx)||_WRMS <= eps_L reads as a 2-norm test.
    // Recover eps_L from it.
    VectorType w;
    operations.error_weights(w);
    const double N    = static_cast<double>(b.size());
    const double wbar = w.l2_norm() / std::sqrt(N);
    const double nrmfac =
      tolerance.ls_norm_factor > 0. ? tolerance.ls_norm_factor : std::sqrt(N);
    sys.eps_L = ida_tolerance * wbar / nrmfac;
    if (tolerance.expected_linear_tolerance > 0.)
      {
        const double ratio = sys.eps_L / tolerance.expected_linear_tolerance;
        tot.eps_ratio_min  = std::min(tot.eps_ratio_min, ratio);
        tot.eps_ratio_max  = std::max(tot.eps_ratio_max, ratio);
      }

    if (parameters.gmres_tolerance_mode == "ida_forcing")
      {
        // IDA's test at x = 0 asks ||W M^{-1} b||_2 to drop below
        // sqrt(N) eps_L; that relative reduction is imposed on the true
        // residual of right-preconditioned GMRES. Before the first
        // factorization there is no M and the solve is direct anyway.
        sys.target = sys.r0;
        if (seed.valid && parameters.policy != "direct")
          {
            sys.Mb = b;
            sys.t_pre =
              timed([&]() { operations.apply_factorization(sys.Mb, b); });
            sys.has_Mb = true;
            VectorType WMb(sys.Mb);
            WMb.scale(w);
            const double pr0       = WMb.l2_norm();
            const double threshold = std::sqrt(N) * sys.eps_L;
            if (pr0 > threshold)
              sys.target = threshold / pr0 * sys.r0;
          }
      }
    else if (parameters.gmres_tolerance_mode == "ida")
      sys.target = ida_tolerance;
    else if (parameters.gmres_tolerance_mode == "fixed")
      sys.target = parameters.gmres_fixed_relative_tolerance * sys.r0;
    else
      AssertThrow(false, dealii::ExcInternalError());
    return sys;
  }



  // Right-preconditioned GMRES without restart, x0 = 0.
  template <typename VectorType>
  typename IDALaggingPolicy<VectorType>::GmresResult
  IDALaggingPolicy<VectorType>::gmres(const VectorType  &b,
                                      const GmresSystem &sys,
                                      VectorType        &x,
                                      const unsigned int cap)
  {
    GmresResult res;
    res.target = sys.target;

    // ida_forcing with eta >= 1: no iteration; the correction is M^{-1} b,
    // as IDA does when its Krylov solver needs no iteration.
    if (sys.has_Mb && sys.r0 <= sys.target)
      {
        x      = sys.Mb;
        res.r0 = res.rm = sys.r0;
        return res;
      }

    dealii::SolverControl control(cap, sys.target, false, false);
    typename dealii::SolverGMRES<VectorType>::AdditionalData data(
      parameters.absolute_iteration_cap + 1, /*right_preconditioning=*/true);
    dealii::SolverGMRES<VectorType>             solver(control, data);
    const internal::LaggingOperator<VectorType> J{operations.apply_jacobian};
    const internal::LaggingOperator<VectorType> M{
      operations.apply_factorization};
    x = 0.;
    try
      {
        solver.solve(J, x, b, M);
      }
    catch (const dealii::SolverControl::NoConvergence &)
      {
        res.converged = false;
      }

    res.m  = control.last_step();
    res.r0 = control.initial_value();
    res.rm = control.last_value();
    return res;
  }



  // One IDA linear solve.
  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::solve(const VectorType &r,
                                      VectorType       &z,
                                      const double      ida_tolerance)
  {
    ++ns.newton_it;

    // Standard Newton: rebuild the operator at the current iterate.
    if (!ns.operator_current &&
        parameters.operator_rebuild_every_n_solves > 0 &&
        ++ns.solves_since_rebuild >= parameters.operator_rebuild_every_n_solves)
      build_operator(ns.t, ns.y, ns.ydot, ns.alpha);

    std::string  kind = "direct", action = "-";
    GmresResult  g;
    unsigned int cap     = 0;
    double       t_solve = 0., e = 0., theta = 0.;
    double       m_hat = std::numeric_limits<double>::quiet_NaN();
    double       E_hat = m_hat;

    // The forcing term of this solve is known before it is decided on.
    const GmresSystem sys = gmres_system(r, ida_tolerance);

    if (ns.new_operator)
      {
        begin_segment(sys.eta()); // decision point (i)
        m_hat = seg.m_hat;
        E_hat = seg.E_hat;
      }
    else if (!pending_refactorization.empty())
      {
        // (R2) fired after the previous solve, on the same Jacobian.
        const double E_s_star = seg.E_s;
        factorize(pending_refactorization);
        seg.E_s = E_s_star; // keep E_s* for the segment log
        pending_refactorization.clear();
        ++tot.deferred_same_jacobian;
      }
    else if (!seed.current)
      {
        // Decision point (ii), (R1) on the same J_s: the factor of the last
        // lagged solve still applies, the forcing term is this solve's.
        m_hat = predict_m(sys.eta(), theta_last);
        E_hat = m_hat * tau_it();
        if (parameters.policy == "adaptive" && E_hat >= episode_average())
          {
            seg.refactorized      = true;
            seg.ended_by          = "R1@solve";
            seg.bound             = cost.T_F.value + tau_it();
            action                = "refact_R1@solve";
            const double E_s_star = seg.E_s;
            factorize("R1@solve");
            seg.E_s = E_s_star;
            ++tot.refact_R1_solve;
          }
      }

    ++seg.n_solves;
    ++ep.nu;

    if (seed.current)
      {
        t_solve = timed([&]() { operations.direct_solve(z, r); });
        cost.T_a.add(t_solve, parameters.moving_average_weight);
        tot.time_direct += t_solve;
        ++tot.direct_solves;
        // ida_forcing: M^{-1} b with the old factorization was spent on a
        // decision that refactorized; count it as GMRES overhead.
        tot.time_gmres += sys.t_pre;
      }
    else
      {
        kind = "lagged";
        cap  = iteration_cap();
        // ida_forcing: the application M^{-1} b made for the forcing term is
        // part of the cost of the lagged solve.
        t_solve = timed([&]() { g = gmres(r, sys, z, cap); }) + sys.t_pre;
        e       = std::max(0., t_solve - T_a());
        theta   = g.m > 0 && g.r0 > 0. ?
                    std::pow(g.rm / g.r0, 1. / static_cast<double>(g.m)) :
                    0.;
        tot.time_gmres += t_solve;
        tot.gmres_iterations += g.m;
        ++tot.lagged_solves;
        ++seg.n_lagged;
        seg.sum_m += g.m;
        seg.E_s += e;
        ep.E += e;

        if (g.m > 0)
          cost.tau_it.add(e / g.m, parameters.moving_average_weight);
        // A solve without iterations measures no factor: keep the last one.
        if (g.m > 0)
          {
            theta_last = theta;
            if (!seg.carried)
              {
                // This segment's drift becomes the reference of the next
                // prediction.
                seg.carried       = true;
                r_last            = seg.r;
                sigma_last_seed   = seg.sigma_seed;
                displacement_last = seg.displacement;
              }
          }
        if (g.m > 0 && g.r0 > 0.)
          {
            seg.pool_log += std::log(std::max(g.rm / g.r0, 1e-300));
            seg.pool_m += g.m;
            theta_pool = std::exp(seg.pool_log / seg.pool_m);
          }

        // Prediction errors, with m_hat clipped to the cap: no solve can
        // show a larger count.
        const double m_hat_seen = std::min(m_hat, static_cast<double>(cap));
        if (seg.n_lagged == 1) // first lagged solve: regression target
          {
            seg.first_m     = g.m;
            seg.first_theta = theta;
            ++tot.predictions;
            tot.pred_err_sum += g.m - m_hat_seen;
            tot.pred_abs_err_sum += std::abs(g.m - m_hat_seen);
          }
        else if (!std::isnan(m_hat))
          {
            ++tot.predictions_in_segment;
            tot.pred_seg_err_sum += g.m - m_hat_seen;
            tot.pred_seg_abs_err_sum += std::abs(g.m - m_hat_seen);
          }
        if (ep.prev_m > 0)
          {
            ++tot.m_pairs;
            if (g.m < ep.prev_m)
              ++tot.m_decreases;
          }
        ep.prev_m = g.m;

        // (R3) cap reached: refactorize J_s and redo this solve. (R2) after
        // a converged solve: factorize before the next solve. (R1) is
        // decided at the next solve, which knows its forcing term.
        std::string reason;
        if (!g.converged)
          {
            kind   = "abandoned";
            reason = parameters.policy == "adaptive" ? "R3" : "cap";
            ++tot.fired_R3;
          }
        else if (parameters.policy == "adaptive" && seg.E_s >= cost.T_F.value)
          {
            reason = "R2";
            ++tot.fired_R2;
          }

        if (!reason.empty())
          {
            seg.refactorized = true;
            seg.ended_by     = reason;
            seg.bound        = cost.T_F.value + tau_it();
            if (g.converged)
              {
                action                  = "defer_" + reason;
                pending_refactorization = reason;
              }
            else
              {
                action                = "refact_" + reason;
                const double E_s_star = seg.E_s;
                factorize(reason);
                seg.E_s = E_s_star;
                const double dt =
                  timed([&]() { operations.direct_solve(z, r); });
                cost.T_a.add(dt, parameters.moving_average_weight);
                tot.time_direct += dt;
                ++tot.direct_solves;
                ++ep.nu;
              }
          }
      }

    // Collective norms first, then rank 0 writes.
    const double r_norm = r.l2_norm();
    if (root)
      log_solves << std::setw(8) << ns.step << std::setw(14) << ns.t
                 << std::setw(5) << ns.newton_it << std::setw(7) << seg.id
                 << std::setw(7) << ep.id << "  " << std::left << std::setw(10)
                 << kind << std::right << std::setw(5) << g.m << std::setw(5)
                 << cap << std::setw(14) << theta << std::setw(14)
                 << ida_tolerance << std::setw(14) << g.target << std::setw(14)
                 << r_norm << std::setw(14) << t_solve << std::setw(14) << e
                 << std::setw(14) << ep.E << std::setw(14) << seg.E_s
                 << std::setw(6) << ep.nu << std::setw(14) << episode_average()
                 << std::setw(14) << (sys.r0 > 0. ? sys.eta() : 0.)
                 << std::setw(14) << m_hat << std::setw(14) << E_hat
                 << std::setw(14) << sys.eps_L << "  " << action << '\n';
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::finish(const std::string &statistics)
  {
    if (!pending_refactorization.empty())
      ++tot.deferred_dropped;
    close_segment();
    close_episode("end");
    write_summary(statistics);
    if (root)
      {
        log_solves.close();
        log_segments.close();
        log_episodes.close();
        log_summary.close();
      }
  }



  template <typename VectorType>
  const typename IDALaggingPolicy<VectorType>::Totals &
  IDALaggingPolicy<VectorType>::totals() const
  {
    return tot;
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::write_header(std::ostream      &out,
                                             const std::string &what,
                                             const std::string &header) const
  {
    const auto &p = parameters;
    out
      << "# =================================================================\n"
      << "# " << what << "\n"
      << "# =================================================================\n"
      << "# MPI ranks           "
      << dealii::Utilities::MPI::n_mpi_processes(communicator) << "\n"
      << "# ls_norm_factor      " << tolerance.ls_norm_factor
      << " (<= 0: sqrt(N))\n"
      << "# expected eps_L      " << tolerance.expected_linear_tolerance
      << " (0: unknown)\n"
      << "#\n"
      << "# Lagging policy:\n"
      << "#   Policy                          = " << p.policy << "\n"
      << "#   Operator rebuild every n solves = "
      << p.operator_rebuild_every_n_solves << "\n"
      << "#   GMRES tolerance mode            = " << p.gmres_tolerance_mode
      << "\n"
      << "#   GMRES fixed relative tolerance  = "
      << p.gmres_fixed_relative_tolerance << "\n"
      << "#   Absolute iteration cap          = " << p.absolute_iteration_cap
      << "\n"
      << "#   Probe rule                      = " << p.probe_rule << "\n"
      << "#   Probe interval min              = " << p.probe_interval_min
      << "\n"
      << "#   Probe interval max              = " << p.probe_interval_max
      << "\n"
      << "#   Moving average weight           = " << p.moving_average_weight
      << "\n";
    std::istringstream lines(header);
    for (std::string line; std::getline(lines, line);)
      out << "#   | " << line << '\n';
    out << "#\n";
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::open_logs(const std::string &header)
  {
    if (!root || parameters.log_prefix.empty())
      return;
    const auto open = [&](std::ofstream &f, const std::string &name) {
      const std::string file = parameters.log_prefix + "_" + name + ".log";
      f.open(file, std::ios::trunc);
      AssertThrow(f, dealii::ExcMessage("Cannot open the log " + file));
      f << std::scientific << std::setprecision(4);
    };
    open(log_solves, "solves");
    open(log_segments, "segments");
    open(log_episodes, "episodes");
    open(log_summary, "summary");

    write_header(log_solves,
                 "Lagging policy: one row per linear solve",
                 header);
    log_solves
      << "# kind      direct | lagged | abandoned (GMRES hit the cap)\n"
      << "# m, cap    GMRES iterations and cap I_max (R3)\n"
      << "# theta     (||r_m|| / ||r_0||)^(1/m), residual of the solved system\n"
      << "# ida_tol   tolerance passed by IDA; target = tolerance used by GMRES\n"
      << "# e         extra cost: solve time - T_a\n"
      << "# E, E_s    extra cost of the episode / of the segment so far\n"
      << "# nu, A     solves of the episode, A = (T_F + E) / nu\n"
      << "# eta       forcing term target / ||r_0|| of the solved system\n"
      << "# m_hat     predicted count of this solve; E_hat = m_hat * tau_it,\n"
      << "#           compared with A by (R1)\n"
      << "# eps_L     IDA's linear tolerance recovered from ida_tol\n"
      << "# action    refact_<rule>: refactorized before this solve (R1@solve)\n"
      << "#           or after it (R3, cap); defer_R2: factorize before the\n"
      << "#           next solve\n"
      << "#" << std::setw(7) << "step" << std::setw(14) << "t" << std::setw(5)
      << "nwt" << std::setw(7) << "seg" << std::setw(7) << "ep"
      << "  " << std::left << std::setw(10) << "kind" << std::right
      << std::setw(5) << "m" << std::setw(5) << "cap" << std::setw(14)
      << "theta" << std::setw(14) << "ida_tol" << std::setw(14) << "target"
      << std::setw(14) << "||r||" << std::setw(14) << "t_solve" << std::setw(14)
      << "e" << std::setw(14) << "E" << std::setw(14) << "E_s" << std::setw(6)
      << "nu" << std::setw(14) << "A" << std::setw(14) << "eta" << std::setw(14)
      << "m_hat" << std::setw(14) << "E_hat" << std::setw(14) << "eps_L"
      << "  action\n";

    write_header(log_segments,
                 "Lagging policy: one row per segment (Jacobian evaluation)",
                 header);
    log_segments
      << "# dalpha    |alpha_s - alpha_seed|\n"
      << "# sigma     ||y_s - y_seed||_WRMS with IDA's current weights\n"
      << "# euclid    ||y_s - y_seed||_2 / ||y_seed||_2\n"
      << "# decision  first | direct | lag_always | keep_R1 | probe_R4 |\n"
      << "#           refact_R1 | refact_R2 (deferred R2 of the previous\n"
      << "#           segment) | refact_noprobe (R4 due, skipped by the\n"
      << "#           alpha check)\n"
      << "# m_hat, E_hat, A, c, b   values used by (R1) and (R4)\n"
      << "# n, n_lag  solves and lagged solves of the segment\n"
      << "# m1, theta1  first lagged solve: count and factor\n"
      << "# E_s*      extra cost of the lagged solves; bound = T_F + tau_it if\n"
      << "#           a rule fired inside the segment (E_s* < bound)\n"
      << "# r         alpha_s / alpha_seed; theta_D = |1-r|/(1+r)\n"
      << "# theta_y   state part: pooled factor minus its alpha part, times\n"
      << "#           sigma / sigma_l, both with the current weights\n"
      << "# theta_pred = theta_D + theta_y\n"
      << "# sigma_seed, theta_y_seed  the same with the seed's weights (the\n"
      << "#           driver's version, logged only)\n"
      << "#" << std::setw(5) << "seg" << std::setw(14) << "t" << std::setw(14)
      << "alpha" << std::setw(14) << "dalpha" << std::setw(14) << "sigma"
      << std::setw(14) << "euclid"
      << "  " << std::left << std::setw(14) << "decision" << std::right
      << std::setw(10) << "m_hat" << std::setw(14) << "E_hat" << std::setw(14)
      << "A" << std::setw(4) << "c" << std::setw(4) << "b" << std::setw(6)
      << "n" << std::setw(6) << "n_lag" << std::setw(6) << "m1" << std::setw(14)
      << "theta1" << std::setw(14) << "mean_m" << std::setw(14) << "E_s*"
      << std::setw(14) << "bound"
      << "  " << std::left << std::setw(10) << "ended_by" << std::right
      << std::setw(14) << "r" << std::setw(14) << "theta_D" << std::setw(14)
      << "theta_y" << std::setw(14) << "theta_pred" << std::setw(14)
      << "sigma_seed" << std::setw(14) << "theta_y_seed" << '\n';

    write_header(log_episodes,
                 "Lagging policy: one row per episode (factorization)",
                 header);
    log_episodes << "# T_F       time of this factorization\n"
                 << "# nu        solves served, n_lag  lagged segments\n"
                 << "# E, A      total extra cost, final (T_F_avg + E) / nu\n"
                 << "# ended_by  rule that triggered the next factorization\n"
                 << "#" << std::setw(5) << "ep" << std::setw(14) << "t_start"
                 << std::setw(14) << "T_F" << std::setw(8) << "nu"
                 << std::setw(8) << "n_lag" << std::setw(14) << "E"
                 << std::setw(14) << "A"
                 << "  ended_by\n";

    write_header(log_summary, "Lagging policy: summary", header);
  }



  template <typename VectorType>
  void
  IDALaggingPolicy<VectorType>::write_summary(const std::string &statistics)
  {
    std::ostringstream s;
    s << std::scientific << std::setprecision(4);
    const double time_linear =
      tot.time_factorize + tot.time_direct + tot.time_gmres;
    const auto ratio = [](const double a, const double b) {
      return b > 0. ? a / b : 0.;
    };

    s << "\n# --- Lagging policy summary, policy " << parameters.policy
      << " ---\n"
      << "jacobian_evaluations  " << tot.jacobian_evaluations << '\n'
      << "factorizations        " << tot.factorizations << '\n'
      << "direct_solves         " << tot.direct_solves << '\n'
      << "lagged_solves         " << tot.lagged_solves << '\n'
      << "gmres_iterations      " << tot.gmres_iterations << '\n'
      << "mean_m_lagged         "
      << ratio(tot.gmres_iterations, tot.lagged_solves) << '\n'
      << "decisions: keep_R1 " << tot.keep_R1 << ", refact_R1@eval "
      << tot.refact_R1_eval << ", refact_R1@solve " << tot.refact_R1_solve
      << ", R2 " << tot.fired_R2 << ", R3 " << tot.fired_R3 << ", probes "
      << tot.probes << " (skipped " << tot.probes_skipped << ")\n"
      << "deferred refactorizations: on new J " << tot.deferred_new_jacobian
      << ", on same J " << tot.deferred_same_jacobian << ", dropped "
      << tot.deferred_dropped << '\n'
      << "\n# times [s]\n"
      << "time_jacobian         " << tot.time_jacobian << '\n'
      << "time_factorize        " << tot.time_factorize << '\n'
      << "time_direct           " << tot.time_direct << '\n'
      << "time_gmres            " << tot.time_gmres << '\n'
      << "time_linear_total     " << time_linear << '\n'
      << "T_F_avg               " << cost.T_F.value << '\n'
      << "T_a_avg               " << cost.T_a.value << '\n'
      << "tau_it_avg            " << tau_it() << '\n'
      << "rho = T_F / T_a       " << ratio(cost.T_F.value, cost.T_a.value)
      << '\n'
      << "\n# checks\n"
      << "eps_L / expected      ";
    if (tolerance.expected_linear_tolerance > 0. && tot.eps_ratio_max > 0.)
      s << "min " << tot.eps_ratio_min << ", max " << tot.eps_ratio_max << '\n';
    else
      s << "not checked\n";
    s << "loss_bound_violations " << tot.bound_violations << " / "
      << tot.bound_checks << "  (E_s* >= T_F + tau_it)\n"
      << "m_decreases_in_episode " << tot.m_decreases << " / " << tot.m_pairs
      << "  (consecutive lagged solves with fewer iterations)\n"
      << "prediction_error      mean "
      << ratio(tot.pred_err_sum, tot.predictions) << ", mean abs "
      << ratio(tot.pred_abs_err_sum, tot.predictions) << "  over "
      << tot.predictions << " segments (m1 - m_hat)\n"
      << "prediction_error_in_segment mean "
      << ratio(tot.pred_seg_err_sum, tot.predictions_in_segment)
      << ", mean abs "
      << ratio(tot.pred_seg_abs_err_sum, tot.predictions_in_segment)
      << "  over " << tot.predictions_in_segment
      << " later lagged solves (m - m_hat)\n";

    pcout << s.str();
    if (root && log_summary.is_open())
      log_summary << s.str() << statistics;
  }
} // namespace ImmersX
#endif
