# TensorKitIsometries [![Build Status](https://github.com/Jutho/TensorKitIsometries.jl/workflows/CI/badge.svg)](https://github.com/Jutho/TensorKitIsometries.jl/actions) [![Coverage](https://codecov.io/gh/Jutho/TensorKitIsometries.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Jutho/TensorKitIsometries.jl)

There are three manifolds: Grassmann, Stiefel and Unitary, corresponding to submodules of TensorKitManifolds, whose names are exported.

They all have a function `Δ = project(!)(X,W)` (e.g. `Grassmann.project(!)` etc) to project an arbitrary tensor `X` onto the tangent space of `W`, which is assumed to be isometric/unitary (not checked). The exclamation mark denotes that `X` is destroyed in the process. The result `Δ` is of a specific type, the corresponding `TensorMap` object can be obtained via an argumentless `getindex`, i.e. `Δ[]` returns the corresponding `TensorMap`. However, you typically don't need those. The base point `W` is also stored in `Δ` and can be returned using `base(Δ)`.

The objects `Δ` returned by `project(!)` behave as vectors: they have scalar multiplication, addition, left and right in-place multiplication with scalars using `lmul!` and `rmul!`, `axpy!` and `axpby!` as well as `dot` and `norm`. When combining two tangent vectors using addition or inner product, they need to have the same `base`.

Furthermore, there are the routines required for OptimKit.jl, which also directly work with the objects returned by `project(!)`:
* `W′, Δ′ = retract(W, Δ, α)`: retract `W` in the direction of `Δ` with step length `α`, return both the retracted isometry `W′` as well as the local tangent `Δ′`
* `Θ′ = transport(!)(Θ, W, Δ, α, W′)`: transport tangent vector `Θ` along the retraction of `W` in the direction of `Δ` with step length `α`, which ends at `W′`. The result is a the transported vector `Θ′` with `base(Θ′) == W′`. The method with exclamation mark destroys `Θ` in the process.

When multiple methods are avaible, they are specified using a keyword argument to the above methods, or explicitly as
`Stiefel.inner_euclidean`, `Stiefel.inner_canonical`, `Stiefel.project_euclidean(!)`, `Stiefel.project_canonical(!)`, `Stiefel.retract_exp`, `Stiefel.transport_exp(!)`, `Stiefel.retract_cayley`, `Stiefel.transport_cayley(!)`, `Unitary.transport_parallel(!)`, `Unitary.transport_stiefel(!)`.
