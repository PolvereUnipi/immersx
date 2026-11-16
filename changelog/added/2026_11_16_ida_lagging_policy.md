PR: #N

- Add an optional lagged-factorization policy for IDA's MUMPS solves, set in `IDA parameters/Linear solver/Lagging policy`. It keeps an older factorization as the GMRES preconditioner of the matrix-free Jacobian across Jacobian setups and decides when to refactorize from measured factorization, backsolve and GMRES costs, following MetricFlowX's cost model. The default `Policy = off` keeps the direct solver unchanged.
