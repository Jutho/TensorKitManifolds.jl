module Stiefel

# isometric W
# tangent vectors Δ = W*A + Z where Z = W⟂*B = Q*R = U*S*V
# with thus A' = -A, W'*Z = 0, W'*U = 0

using TensorKit
using TensorKit: similarstoragetype, fusiontreetype, StaticLength, SectorDict
using ..TensorKitManifolds: projecthermitian!, projectantihermitian!,
                            projectisometric!, projectcomplement!, PolarNewton,
                            _stiefelexp, _stiefellog, eleps
import ..TensorKitManifolds: base, checkbase,
                                inner, retract, invretract, transport, transport!

# special type to store tangent vectors using A and Z = Q*R,
mutable struct StiefelTangent{T<:AbstractTensorMap, TA<:AbstractTensorMap}
    W::T
    A::TA
    Z::T
end
function Base.copy(Δ::StiefelTangent)
    Δ′ = StiefelTangent(Δ.W, copy(Δ.A), copy(Δ.Z))
    return Δ′
end

Base.getindex(Δ::StiefelTangent) = Δ.W * Δ.A + Δ.Z
base(Δ::StiefelTangent) = Δ.W
checkbase(Δ₁::StiefelTangent, Δ₂::StiefelTangent) = Δ₁.W == Δ₂.W ? Δ₁.W :
    throw(ArgumentError("tangent vectors with different base points"))

# Basic vector space behaviour
Base.:+(Δ₁::StiefelTangent, Δ₂::StiefelTangent) =
    StiefelTangent(checkbase(Δ₁,Δ₂), Δ₁.A + Δ₂.A, Δ₁.Z + Δ₂.Z)
Base.:-(Δ₁::StiefelTangent, Δ₂::StiefelTangent) =
    StiefelTangent(checkbase(Δ₁,Δ₂), Δ₁.A - Δ₂.A, Δ₁.Z - Δ₂.Z)
Base.:-(Δ::StiefelTangent) = (-1)*Δ

Base.:*(Δ::StiefelTangent, α::Real) = rmul!(copy(Δ), α)
Base.:*(α::Real, Δ::StiefelTangent) = lmul!(α, copy(Δ))
Base.:/(Δ::StiefelTangent, α::Real) = rmul!(copy(Δ), inv(α))
Base.:\(α::Real, Δ::StiefelTangent) = lmul!(inv(α), copy(Δ))

function TensorKit.rmul!(Δ::StiefelTangent, α::Real)
    rmul!(Δ.A, α)
    rmul!(Δ.Z, α)
    return Δ
end
function TensorKit.lmul!(α::Real, Δ::StiefelTangent)
    lmul!(α, Δ.A)
    lmul!(α, Δ.Z)
    return Δ
end
function TensorKit.axpy!(α::Real, Δx::StiefelTangent, Δy::StiefelTangent)
    checkbase(Δx, Δy)
    axpy!(α, Δx.A, Δy.A)
    axpy!(α, Δx.Z, Δy.Z)
    return Δy
end
function TensorKit.axpby!(α::Real, Δx::StiefelTangent, β::Real, Δy::StiefelTangent)
    checkbase(Δx, Δy)
    axpby!(α, Δx.A, β, Δy.A)
    axpby!(α, Δx.Z, β, Δy.Z)
    return Δy
end

function TensorKit.dot(Δ₁::StiefelTangent, Δ₂::StiefelTangent)
    checkbase(Δ₁, Δ₂)
    dot(Δ₁.A, Δ₂.A) + dot(Δ₁.Z, Δ₂.Z)
end
TensorKit.norm(Δ::StiefelTangent, p::Real = 2) =
    norm((norm(Δ.A, p), norm(Δ.Z, p)), p)

# tangent space methods
function inner(W::AbstractTensorMap, Δ₁::StiefelTangent, Δ₂::StiefelTangent;
                metric = :euclidean)
    if metric == :euclidean
        return inner_euclidean(W, Δ₁, Δ₂)
    elseif metric == :canonical
        return inner_canonical(W, Δ₁, Δ₂)
    else
        throw(ArgumentError("unknown metric: $metric"))
    end
end
function project!(X::AbstractTensorMap, W::AbstractTensorMap; metric = :euclidean)
    if metric == :euclidean
        return project_euclidean!(X, W)
    elseif metric == :canonical
        return project_canonical!(X, W)
    else
        throw(ArgumentError("unknown metric: `metric = $metric`"))
    end
end
project(X::AbstractTensorMap, W::AbstractTensorMap; metric = :euclidean) =
    project!(copy(X), W; metric = metric)

function retract(W::AbstractTensorMap, Δ::StiefelTangent, α::Real; alg = :exp)
    if alg == :exp
        return retract_exp(W, Δ, α)
    elseif alg == :cayley
        return retract_cayley(W, Δ, α)
    else
        throw(ArgumentError("unknown algorithm: `alg = $alg`"))
    end
end
function invretract(Wold::AbstractTensorMap, Wnew::AbstractTensorMap; alg = :exp)
    if alg == :exp
        return invretract_exp(Wold, Wnew)
    elseif alg == :cayley
        return invretract_cayley(Wold, Wnew)
    else
        throw(ArgumentError("unknown algorithm: `alg = $alg`"))
    end
end
function transport!(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent, α::Real, W′;
                    alg = :exp)
    if alg == :exp
        return transport_exp!(Θ, W, Δ, α, W′)
    elseif alg == :cayley
        return transport_cayley!(Θ, W, Δ, α, W′)
    else
        throw(ArgumentError("unknown algorithm: `alg = $alg`"))
    end
end
function transport(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent, α::Real, W′;
                    alg = :exp)
    return transport!(copy(Θ), W, Δ, α, W′; alg = alg)
end

# euclidean metric
function inner_euclidean(W::AbstractTensorMap, Δ₁::StiefelTangent, Δ₂::StiefelTangent)
    Δ₁ === Δ₂ ? norm(Δ₁)^2 : real(dot(Δ₁,Δ₂))
end
function project_euclidean!(X::AbstractTensorMap, W::AbstractTensorMap)
    P = W'*X
    Z = mul!(X, W, P, -1, 1)
    A = projectantihermitian!(P)
    Z = projectcomplement!(Z, W)
    return StiefelTangent(W, A, Z)
end
project_euclidean(X, W) = project_euclidean!(copy(X), W)

# canonical metric
function inner_canonical(W::AbstractTensorMap, Δ₁::StiefelTangent, Δ₂::StiefelTangent)
    if Δ₁ === Δ₂
        return (norm(Δ₁.A)^2)/2 + norm(Δ₁.Z)^2
    else
        return real(dot(Δ₁.A, Δ₂.A)/2 + dot(Δ₁.Z, Δ₂.Z))
    end
end
function project_canonical!(X::AbstractTensorMap, W::AbstractTensorMap)
    P = W'*X
    Z = mul!(X, W, P, -1, 1)
    A = rmul!(projectantihermitian!(P), 2)
    Z = projectcomplement!(Z, W)
    return StiefelTangent(W, A, Z)
end
project_canonical(X, W) = project_canonical!(copy(X), W)

# geodesic retraction for canonical metric using exponential
function stiefelexp(W::AbstractTensorMap,
                    A::AbstractTensorMap,
                    Z::AbstractTensorMap,
                    α::Real)
    S = spacetype(W)
    G = sectortype(W)
    Wdata′ = TensorKit.SectorDict{G, storagetype(W)}()
    Qdata = TensorKit.SectorDict{G, storagetype(W)}()
    Qdata′ = TensorKit.SectorDict{G, storagetype(W)}()
    Rdata′ = TensorKit.SectorDict{G, storagetype(W)}()
    dims = TensorKit.SectorDict{G, Int}()
    for c in blocksectors(W)
        w′, q, q′, r′ = _stiefelexp(block(W, c), block(A, c), block(Z, c), α)
        Wdata′[c] = w′
        Qdata[c] = q
        Qdata′[c] = q′
        Rdata′[c] = r′
        dims[c] = size(q, 2)
    end
    V = S(dims)
    W′ = TensorMap(Wdata′, space(W))
    Q = TensorMap(Qdata, codomain(W)←V)
    Q′ = TensorMap(Qdata′, codomain(W)←V)
    R′ = TensorMap(Rdata′, V←domain(W))
    return W′, Q, Q′, R′
end

# can be computed efficiently: O(np^2) + O(p^3)
function retract_exp(W::AbstractTensorMap, Δ::StiefelTangent, α::Real)
    W == base(Δ) || throw(ArgumentError("not a valid tangent vector at base point"))
    W′, Q, Q′, R′ = stiefelexp(W, Δ.A, Δ.Z, α)
    A′ = Δ.A
    Z′ = projectcomplement!(Q′*R′, W′) # to ensure orthogonality
    return W′, StiefelTangent(W′, A′, Z′)
end
function invretract_exp(Wold::AbstractTensorMap, Wnew::AbstractTensorMap;
                            tol = eleps(Wold)^(2/3))
    space(Wold) == space(Wnew) || throw(SectorMismatch())

    S = spacetype(Wold)
    G = sectortype(Wold)
    Adata = TensorKit.SectorDict{G, storagetype(Wold)}()
    Zdata = TensorKit.SectorDict{G, storagetype(Wold)}()
    for c in blocksectors(Wold)
        a, q, r = _stiefellog(block(Wold, c), block(Wnew, c); tol = tol)
        Adata[c] = a
        Zdata[c] = q*r
    end
    A = TensorMap(Adata, domain(Wold)←domain(Wold))
    Z = TensorMap(Zdata, space(Wold))
    return StiefelTangent(Wold, A, Z)
end

# vector transport compatible with above `retract`: also differentiated retraction
# isometric for both euclidean and canonical metric
# not parallel transport for either metric as the corresponding connection has torsion
# can be computed efficiently: O(np^2) + O(p^3)
function transport_exp!(Θ::StiefelTangent, W::AbstractTensorMap,
                        Δ::StiefelTangent, α::Real, W′)
    W == checkbase(Δ,Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    W′, Q, Q′, R′ = stiefelexp(W, Δ.A, Δ.Z, α)
    A = Θ.A
    Z = Θ.Z
    QZ = Q'*Θ.Z
    A′ = Θ.A
    Z′ = projectcomplement!(mul!(Z, (Q′-Q), QZ, 1, 1), W′)
    return StiefelTangent(W′, A′, Z′)
end
transport_exp(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent, α::Real, W′) =
    transport_exp!(copy(Θ), W, Δ, α, W′)

# Cayley retraction, slightly more efficient than above?
# can be computed efficiently: O(np^2) + O(p^3)
function retract_cayley(W::AbstractTensorMap, Δ::StiefelTangent, α::Real)
    W == base(Δ) || throw(ArgumentError("not a valid tangent vector at base point"))
    A, Z = Δ.A, Δ.Z
    ZdZ = Z'*Z
    X = axpy!(α^2/4, ZdZ, axpy!(-α/2, A, one(A)))
    iX = inv(X)
    W′ = projectisometric!((2*W+α*Z)*iX - W; alg = PolarNewton())
    A′ = projectantihermitian!((A - (α/2)*ZdZ)*iX)
    Z′ = (Z-α*(W+α/2*Z)*(iX*ZdZ))
    Z′ = projectcomplement!(Z′*projecthermitian!(iX), W′)
    return W′, StiefelTangent(W′, A′, Z′)
end
function invretract_cayley(Wnew::AbstractTensorMap, Wold::AbstractTensorMap)
    space(Wnew) == space(Wold) || throw(SpaceMismatch())

    P = Wold'*Wnew
    iX = rmul!(axpy!(1,P,one(P)), 1/2)
    X = inv(iX)
    Z = projectcomplement!(Wnew - Wold*P, Wold)*X
    A = projectantihermitian!(rmul!(axpy!(-1, X, mul!(one(X), Z', Z, 1/4, 1)), 2))
    return StiefelTangent(Wold, A, Z)
end

# vector transport compatible with above `retract_caley`, but not differentiated retraction
# isometric for both euclidean and canonical metric
# can be computed efficiently: O(np^2) + O(p^3)
function transport_cayley!(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent,
                            α::Real, W′)
    W == checkbase(Δ,Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    A, Z = Δ.A, Δ.Z
    X = axpy!(α^2/4, Z'*Z, axpy!(-α/2, A, one(A)))
    A′ = Θ.A
    ZdZ = Z'*Θ.Z
    Z′ = projectcomplement!(axpy!(-α, (W+(α/2)*Z)*(X\ZdZ), Θ.Z), W′)
    return StiefelTangent(W′, A′, Z′)
end
transport_cayley(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent, α::Real, W′) =
    transport_cayley!(copy(Θ), W, Δ, α, W′)

end
