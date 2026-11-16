PR: #N

- Add an optional lagged-factorization policy for IDA's direct linear solves. It keeps an older factorization as the GMRES preconditioner across Jacobian setups and decides when to refactorize from measured factorization, backsolve and GMRES costs, following MetricFlowX's cost model.
