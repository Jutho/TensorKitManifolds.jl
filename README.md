# TensorKitManifolds [![Build Status](https://github.com/Jutho/TensorKitManifolds.jl/workflows/CI/badge.svg)](https://github.com/Jutho/TensorKitManifolds.jl/actions) [![Coverage](https://codecov.io/gh/Jutho/TensorKitManifolds.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Jutho/TensorKitManifolds.jl)

There are three manifolds: Grassmann, Stiefel and Unitary, corresponding to submodules of TensorKitManifolds, whose names are exported.

They all have a function `Δ = project(!)(X,W)` (e.g. `Grassmann.project(!)` etc) to project an arbitrary tensor `X` onto the tangent space of `W`, which is assumed to be isometric/unitary (not checked). The exclamation mark denotes that `X` is destroyed in the process. The result `Δ` is of a specific type, the corresponding `TensorMap` object can be obtained via an argumentless `getindex`, i.e. `Δ[]` returns the corresponding `TensorMap`. However, you typically don't need those. The base point `W` is also stored in `Δ` and can be returned using `W = base(Δ)`. Hence, `Δ` should be assumed to be a point `(W, Δ[])` on the tangent bundle of the manifold.

The objects `Δ` returned by `project(!)` also satisfy the behaviour of vector: they have scalar multiplication, addition, left and right in-place multiplication with scalars using `lmul!` and `rmul!`, `axpy!` and `axpby!` as well as complex euclidean inner product `dot` and corresponding `norm`. When combining two tangent vectors using addition or inner product, they need to have the same `base`.

Furthermore, there are the routines required for OptimKit.jl, which also directly work with the objects returned by `project(!)`:
* `W′, Δ′ = retract(W, Δ, α)`: retract `W` in the direction of `Δ` with step length `α`, return both the retracted isometry `W′` as well as the local tangent `Δ′`
* `inner(W, Δ₁, Δ₂)`: inner product between tangent vectors at the point `W`. Note that `W` is already encoded in `base(Δ₁)` and `base(Δ₂)`, but this is the required interface for the inner product of OptimKit.jl. `inner(W, Δ₁, Δ₂; metric = :euclidean) = real(dot(Δ₁,Δ₂))` but other metrics might also be available.
* `Θ′ = transport(!)(Θ, W, Δ, α, W′)`: transport tangent vector `Θ` along the retraction of `W` in the direction of `Δ` with step length `α`, which ends at `W′`. The result is a the transported vector `Θ′` with `base(Θ′) == W′`. The method with exclamation mark destroys `Θ` in the process.

When multiple methods are avaible, they are specified using a keyword argument to the above methods, or explicitly as
`Stiefel.inner_euclidean`, `Stiefel.inner_canonical`, `Stiefel.project_euclidean(!)`, `Stiefel.project_canonical(!)`, `Stiefel.retract_exp`, `Stiefel.transport_exp(!)`, `Stiefel.retract_cayley`, `Stiefel.transport_cayley(!)`, `Unitary.transport_parallel(!)`, `Unitary.transport_stiefel(!)`.
