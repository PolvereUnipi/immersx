// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on deal.II.
//
// ---------------------------------------------------------------------
#ifndef immersx_ida_lagging_policy_h
#define immersx_ida_lagging_policy_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi_stub.h>

#include <immersx/algebra/linear_algebra.h>
#include <immersx/core/ida_lagging_parameters.h>

#include <fstream>
#include <functional>
#include <limits>
#include <string>

namespace ImmersX
{
  /** Decides, for every linear solve requested by IDA, whether to solve
   * directly with a factorization of the current Jacobian or with GMRES
   * preconditioned by an older factorization. It implements the policies
   * direct, adaptive (rules R1-R4) and lag_always of the cost model in
   * MetricFlowX's latex/lagging_cost_model.tex (Algorithm "Online
   * refactorization policy"), and writes the logs of MetricFlowX's
   * blood_flow_IDA_lagging driver with a few extra columns.
   *
   * The policy owns the decisions, the cost averages and the logs. The caller
   * owns the Jacobian, its factorization and IDA, and exposes them through
   * Operations. All Operations and all member functions are collective on the
   * communicator. Timings are maximized over ranks, so every rank takes the
   * same decisions.
   *
   * As in the driver, each state drift Sigma = ||y - y_seed||_WRMS is
   * measured with the weights of its own seed at the final tolerances, so
   * Sigma_s / Sigma_l compares drifts relative to their linearization points
   * even when the carried factor comes from an earlier seed.
   *
   * Difference from the driver: the forcing term uses IDA's own error weights
   * and recovers IDA's linear tolerance eps_L from the tolerance IDA passes
   * and the ls_norm_factor, so it does not depend on that factor.
   */
  template <typename VectorType>
  class IDALaggingPolicy
  {
  public:
    using Apply = std::function<void(VectorType &, const VectorType &)>;

    struct Operations
    {
      /** Build J = dF/dy + alpha dF/dydot at (t, y, ydot) for apply_jacobian,
       * without factorizing it.
       */
      std::function<void(const double,
                         const VectorType &,
                         const VectorType &,
                         const double)>
        build_jacobian;
      /** Factorize the Jacobian built last. Its time is the cost T_F. */
      std::function<void()> factorize;
      /** dst = M^{-1} src with the current factorization. */
      Apply apply_factorization;
      /** Solve J dst = src with a factorization of the current Jacobian,
       * including the caller's acceptance checks. Its time is T_a.
       */
      Apply direct_solve;
      /** dst = J src with the Jacobian built last. */
      Apply apply_jacobian;
      /** Reinitialize the argument and fill it with IDA's current error
       * weights. Used by the forcing term.
       */
      std::function<void(VectorType &)> error_weights;
      /** Reinitialize w and fill it with IDA's error weights at the state y
       * with the final tolerances, without a tolerance ramp:
       * w_i = 1 / (rtol |y_i| + atol_i). Used for the seed's drift norm.
       */
      std::function<void(const VectorType &y, VectorType &w)> state_weights;
    };

    /** How IDA converts its linear tolerance before passing it. */
    struct IDATolerance
    {
      /** IDA's ls_norm_factor; zero or negative means sqrt(N), as in
       * SUNDIALS.
       */
      double ls_norm_factor = 0.;
      /** eps_L = eplifac * epsNewt that IDA should be using, logged next to
       * the recovered value as a check; zero if unknown.
       */
      double expected_linear_tolerance = 0.;
    };

    /** Run totals, also written to the summary log. */
    struct Totals
    {
      unsigned int jacobian_evaluations = 0, factorizations = 0;
      unsigned int direct_solves = 0, lagged_solves = 0, gmres_iterations = 0;
      unsigned int keep_R1 = 0, refact_R1_eval = 0, refact_R1_solve = 0;
      unsigned int fired_R2 = 0, fired_R3 = 0, probes = 0, probes_skipped = 0;
      unsigned int deferred_new_jacobian = 0, deferred_same_jacobian = 0;
      unsigned int deferred_dropped = 0;
      unsigned int bound_violations = 0, bound_checks = 0;
      unsigned int m_decreases = 0, m_pairs = 0;
      unsigned int predictions  = 0;
      double       pred_err_sum = 0., pred_abs_err_sum = 0.;
      unsigned int predictions_in_segment = 0;
      double       pred_seg_err_sum = 0., pred_seg_abs_err_sum = 0.;
      double       time_factorize = 0., time_direct = 0., time_gmres = 0.;
      double       time_jacobian = 0.;
      /** Recovered eps_L divided by the expected one, over all solves. */
      double eps_ratio_min = std::numeric_limits<double>::infinity();
      double eps_ratio_max = 0.;
    };

    /** The policy must not be off: the caller then solves directly. */
    IDALaggingPolicy(const IDALaggingParameters &lagging_parameters,
                     const IDATolerance         &ida_tolerance,
                     const MPI_Comm              mpi_communicator,
                     Operations                  callbacks);

    /** Reset all state and open the logs. The header text, e.g. the IDA
     * controls, is copied into every log after the policy knobs.
     */
    void
    start(const std::string &header = "");

    /** IDA's residual callback: record the Newton iterate. */
    void
    residual(const double t, const VectorType &y, const VectorType &ydot);

    /** IDA's setup_jacobian callback. */
    void
    setup_jacobian(const double      t,
                   const VectorType &y,
                   const VectorType &ydot,
                   const double      alpha);

    /** IDA's solve_with_jacobian callback: solve J z = r. */
    void
    solve(const VectorType &r, VectorType &z, const double tolerance);

    /** The caller replaced the factorization outside the policy, e.g. for an
     * initial-condition correction. Close the episode; the next Jacobian
     * evaluation through the policy factorizes again.
     */
    void
    reset_factorization();

    /** Close the open segment and episode, and write the summary. The text is
     * appended to the summary, e.g. the IDA statistics.
     */
    void
    finish(const std::string &statistics = "");

    const Totals &
    totals() const;

  private:
    struct Ema
    {
      double value = 0.;
      bool   init  = false;
      void
      add(const double x, const double weight)
      {
        value = init ? (1. - weight) * value + weight * x : x;
        init  = true;
      }
    };

    // One segment = the solves between two Jacobian evaluations.
    struct Segment
    {
      unsigned int id      = 0;
      double       t       = 0.;
      double       alpha   = 0.;
      double       dalpha  = 0.; // |alpha_s - alpha_seed|
      double       sigma   = 0.; // ||y_s - y_seed||_WRMS, seed's weights
      double       euclid  = 0.; // ||y_s - y_seed||_2 / ||y_seed||_2
      unsigned int seed_id = 0;  // seed the drift is measured from
      std::string  decision;
      double       m_hat  = 0.; // predicted iterations of a lagged solve
      double       E_hat  = 0.; // predicted extra cost of a lagged solve
      double       A      = 0.; // episode average cost per solve at decision
      unsigned int c      = 0;  // (R4) state at decision
      unsigned int b      = 0;
      bool         lagged = false; // segment started with a kept factorization

      unsigned int n_solves = 0, n_lagged = 0, sum_m = 0;
      unsigned int first_m      = 0;
      double       first_theta  = std::numeric_limits<double>::quiet_NaN();
      double       E_s          = 0.;    // extra cost paid in this segment
      bool         refactorized = false; // R1@solve, R2 or R3 fired
      std::string  ended_by     = "-";
      double       bound        = 0.; // T_F + tau_it when refactorized
      double       r            = 1.; // alpha_s / alpha_seed
      double       theta_D      = 0.; // alpha part of the predicted factor
      double       theta_y      = 0.; // state part of the predicted factor
      double       theta_pred   = std::numeric_limits<double>::quiet_NaN();
      // Diagnostics, not used by the decision: Sigma_l of the carried factor
      // in its own seed's weights (used) and in this seed's weights, the
      // prediction with the latter, and whether the factor comes from an
      // earlier seed.
      double sigma_l        = std::numeric_limits<double>::quiet_NaN();
      double sigma_l_cur    = std::numeric_limits<double>::quiet_NaN();
      double theta_pred_cur = std::numeric_limits<double>::quiet_NaN();
      bool   cross_seed     = false;
      // Pool of the lagged solves with m >= 1: sum of log(rm/r0), sum of m.
      double       pool_log = 0.;
      unsigned int pool_m   = 0;
      // y_s - y_seed; kept as the carried displacement once this segment
      // measures a factor.
      VectorType displacement;
      bool       carried = false;
    };

    // One episode = the solves served by one factorization.
    struct Episode
    {
      unsigned int id      = 0;
      double       t_start = 0., T_F = 0.;
      unsigned int nu = 0, n_lagged_segments = 0;
      double       E      = 0.;
      unsigned int prev_m = 0; // last lagged count, for the monotonicity check
    };

    // Latest Newton iterate and the point where the operator was built.
    struct NewtonState
    {
      double       t = 0., alpha = 0.;
      VectorType   y, ydot;
      double       last_t = std::numeric_limits<double>::quiet_NaN();
      unsigned int step = 0, newton_it = 0, solves_since_rebuild = 0;
      bool         operator_current = false; // built at the latest iterate
      bool         new_operator     = false; // changed since the last solve
      double       op_alpha         = 0.;
      VectorType   op_y;
    };

    struct Seed
    {
      bool         valid   = false;
      bool         current = false; // factorization of the current operator
      unsigned int id      = 0;     // = episode id
      double       alpha   = 0.;
      // Seed state and its IDA weights at the final tolerances (no ramp):
      // Sigma is the drift relative to the linearization point.
      VectorType y, w;
    };

    struct Costs
    {
      Ema T_F, T_a, tau_it;
    };

    // The system GMRES solves for a right-hand side b: tolerance and initial
    // residual (x0 = 0), and IDA's linear tolerance recovered from tol.
    struct GmresSystem
    {
      double target = 0., r0 = 0.;
      double eps_L = 0.;
      // ida_forcing, ida_left: M^{-1} b with the current factorization and
      // the time spent on it. If no iteration is needed (eta >= 1) it is the
      // correction, as in IDA.
      VectorType Mb;
      bool       has_Mb = false;
      double     t_pre  = 0.;
      // ida_left: GMRES runs on IDA's own test. Unknown y = W x, operator
      // W M^{-1} J W^{-1}, right-hand side W M^{-1} b; r0 and target are
      // 2-norms of that system's residual. w holds IDA's weights.
      bool       left = false;
      VectorType w, w_inv, WMb;
      // Forcing term: the relative reduction GMRES must achieve.
      double
      eta() const
      {
        return r0 > target ? target / r0 : 1.;
      }
    };

    struct GmresResult
    {
      unsigned int m  = 0;
      double       r0 = 0., rm = 0., target = 0.;
      bool         converged = true;
    };

    template <typename F>
    double
    timed(F &&f) const;

    static double
    wrms(const VectorType &d, const VectorType &w);

    double
    T_a() const;

    double
    tau_it() const;

    double
    episode_average() const;

    static double
    predict_m(const double eta, const double theta);

    static double
    theta_disk(const double r);

    void
    build_operator(const double      t,
                   const VectorType &y,
                   const VectorType &ydot,
                   const double      alpha);

    void
    factorize(const std::string &reason);

    void
    begin_segment(const double eta);

    void
    close_segment();

    void
    close_episode(const std::string &reason);

    unsigned int
    iteration_cap() const;

    GmresSystem
    gmres_system(const VectorType &b, const double ida_tolerance);

    GmresResult
    gmres(const VectorType  &b,
          const GmresSystem &system,
          VectorType        &x,
          const unsigned int cap);

    void
    open_logs(const std::string &header);

    void
    write_header(std::ostream      &out,
                 const std::string &what,
                 const std::string &header) const;

    void
    write_summary(const std::string &statistics);

    const IDALaggingParameters parameters;
    const IDATolerance         tolerance;
    const MPI_Comm             communicator;
    const Operations           operations;
    const bool                 root;
    dealii::ConditionalOStream pcout;

    Segment     seg;
    Episode     ep;
    NewtonState ns;
    Seed        seed;
    Costs       cost;
    Totals      tot;
    // GMRES factor of the last lagged solve with m >= 1; NaN before it.
    double theta_last = std::numeric_limits<double>::quiet_NaN();
    // Pooled factor of the last segment with lagged iterations, the drift at
    // which it was measured (r, Sigma in its seed's weights) and that seed.
    // The displacement vector is kept for the diagnostic sigma_l_cur.
    double       theta_pool = std::numeric_limits<double>::quiet_NaN();
    double       r_last = 1., sigma_last = 0.;
    unsigned int seed_last = 0;
    VectorType   displacement_last;
    unsigned int probe_c = 0, probe_b = 1;

    // (R2) fired after a converged lagged solve. The factorization waits for
    // the next solve, so that a Jacobian built in between is the one
    // factorized.
    std::string pending_refactorization;

    std::ofstream log_solves, log_segments, log_episodes, log_summary;
  };

  extern template class IDALaggingPolicy<ImmersXLA::MPI::Vector>;
  extern template class IDALaggingPolicy<ImmersXLA::MPI::BlockVector>;
} // namespace ImmersX
#endif
