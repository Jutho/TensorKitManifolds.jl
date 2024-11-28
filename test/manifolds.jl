using Manifolds
using TensorKit, TensorKitManifolds
using ManifoldsBase
using Manopt

V = ComplexSpace(2) ← ComplexSpace(2)

qs = [rand(V) for _ in 1:4]
M = Euclidean(V)

f(M, p) = 1 / (2 * length(qs)) * sum(distance(M, p, q) for q in qs)
grad_f(M, p) = -1 / length(qs) * sum(Manifolds.log(M, p, q) for q in qs)

m2 = gradient_descent(M, f, grad_f, qs[1];
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.11f | "), "\n", :Stop],
    stopping_criterion = StopAfterIteration(6)
  )
