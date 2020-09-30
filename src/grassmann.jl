module Grassmann

# isometric W (representative of equivalence class)
# tangent vectors Δ = Z where Z = W⟂*B = Q*R = U*S*V
# with thus W'*Z = 0, W'*U = 0

using TensorKit
using TensorKit: similarstoragetype, SectorDict
using ..TensorKitManifolds: projecthermitian!, projectantihermitian!,
                            projectisometric!, projectcomplement!, PolarNewton
import ..TensorKitManifolds: base, checkbase, inner, retract, transport, transport!

# special type to store tangent vectors using Z
# add SVD of Z = U*S*V upon first creation
mutable struct GrassmannTangent{T<:AbstractTensorMap,
                                TU<:AbstractTensorMap,
                                TS<:AbstractTensorMap,
                                TV<:AbstractTensorMap}
    W::T
    Z::T
    U::Union{Nothing,TU}
    S::Union{Nothing,TS}
    V::Union{Nothing,TV}
    function GrassmannTangent(W::AbstractTensorMap{S,N₁,N₂},
                                Z::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
        T = typeof(W)
        TT = promote_type(float(eltype(W)), eltype(Z))
        G = sectortype(W)
        M = similarstoragetype(W, TT)
        Mr = similarstoragetype(W, real(TT))
        TU = tensormaptype(S, N₁, 1, M)
        TS = isreal(sectortype(S)) ? tensormaptype(S, 1, 1, Mr) : tensormaptype(S, 1, 1, M)
        TV = tensormaptype(S, 1, N₂, M)
        return new{T,TU,TS,TV}(W, Z, nothing, nothing, nothing)
    end
end
function Base.copy(Δ::GrassmannTangent)
    Δ′ = GrassmannTangent(Δ.W, copy(Δ.Z))
    if Base.getfield(Δ, :U) !== nothing
        Base.setfield!(Δ′, :U, copy(Base.getfield(Δ, :U)))
        Base.setfield!(Δ′, :S, copy(Base.getfield(Δ, :S)))
        Base.setfield!(Δ′, :V, copy(Base.getfield(Δ, :V)))
    end
    return Δ′
end

Base.getindex(Δ::GrassmannTangent) = Δ.Z
base(Δ::GrassmannTangent) = Δ.W
checkbase(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) = Δ₁.W == Δ₂.W ? Δ₁.W :
    throw(ArgumentError("tangent vectors with different base points"))

function Base.getproperty(Δ::GrassmannTangent, sym::Symbol)
    if sym ∈ (:W, :Z)
        return Base.getfield(Δ, sym)
    elseif sym ∈ (:U, :S, :V)
        v = Base.getfield(Δ, sym)
        v !== nothing && return v
        U, S, V, = tsvd(Δ.Z)
        Base.setfield!(Δ, :U, U)
        Base.setfield!(Δ, :S, S)
        Base.setfield!(Δ, :V, V)
        v = Base.getfield(Δ, sym)
        @assert v !== nothing
        return v
    else
        error("type GrassmannTangent has no field $sym")
    end
end
function Base.setproperty!(Δ::GrassmannTangent, sym::Symbol, v)
    error("type GrassmannTangent does not allow to change its fields")
end

# Basic vector space behaviour
Base.:+(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) =
    GrassmannTangent(checkbase(Δ₁,Δ₂), Δ₁.Z + Δ₂.Z)
Base.:-(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) =
    GrassmannTangent(checkbase(Δ₁,Δ₂), Δ₁.Z - Δ₂.Z)
Base.:-(Δ::GrassmannTangent) = (-1)*Δ

Base.:*(Δ::GrassmannTangent, α::Number) = rmul!(copy(Δ), α)
Base.:*(α::Number, Δ::GrassmannTangent) = lmul!(α, copy(Δ))
Base.:/(Δ::GrassmannTangent, α::Number) = rmul!(copy(Δ), inv(α))
Base.:\(α::Number, Δ::GrassmannTangent) = lmul!(inv(α), copy(Δ))

Base.zero(Δ::GrassmannTangent) = GrassmannTangent(Δ.W, zero(Δ.Z))

function TensorKit.rmul!(Δ::GrassmannTangent, α::Number)
    rmul!(Δ.Z, α)
    if Base.getfield(Δ, :S) !== nothing
        if sign(α) != 1
            rmul!(Δ.V, sign(α))
        end
        rmul!(Δ.S, abs(α))
    end
    return Δ
end
function TensorKit.lmul!(α::Number, Δ::GrassmannTangent)
    lmul!(α, Δ.Z)
    if Base.getfield(Δ, :S) !== nothing
        if sign(α) != 1
            lmul!(sign(α), Δ.U)
        end
        lmul!(abs(α), Δ.S)
    end
    return Δ
end
function TensorKit.axpy!(α::Number, Δx::GrassmannTangent, Δy::GrassmannTangent)
    checkbase(Δx, Δy)
    axpy!(α, Δx.Z, Δy.Z)
    Base.setfield!(Δy, :U, nothing)
    Base.setfield!(Δy, :S, nothing)
    Base.setfield!(Δy, :V, nothing)
    return Δy
end
function TensorKit.axpby!(α::Number, Δx::GrassmannTangent, β::Number, Δy::GrassmannTangent)
    checkbase(Δx, Δy)
    axpby!(α, Δx.Z, β, Δy.Z)
    Base.setfield!(Δy, :U, nothing)
    Base.setfield!(Δy, :S, nothing)
    Base.setfield!(Δy, :V, nothing)
    return Δy
end

function TensorKit.dot(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent)
    checkbase(Δ₁, Δ₂)
    return dot(Δ₁.Z, Δ₂.Z)
end
TensorKit.norm(Δ::GrassmannTangent, p::Real = 2) = norm(Δ.Z, p)

# tangent space methods
function project!(X::AbstractTensorMap, W::AbstractTensorMap; metric = :euclidean)
    @assert metric == :euclidean
    P = W'*X
    Z = mul!(X, W, P, -1, 1)
    Z = projectcomplement!(Z, W)
    return GrassmannTangent(W, Z)
end

"""
    Grassmann.project(X::AbstractTensorMap, W::AbstractTensorMap; metric = :euclidean)

Project X onto the Grassmann tangent space at the base point `W`, which is assumed to be
isometric, i.e. `W'*W ≈ id(domain(W))`. The resulting tensor `Z` in the tangent space of
`W` is given by `Z = X - W * (W'*X)` and satisfies `W'*Z = 0`.
"""
project(X::AbstractTensorMap, W::AbstractTensorMap; metric = :euclidean) =
    project!(copy(X), W; metric=metric)

function inner(W::AbstractTensorMap, Δ₁::GrassmannTangent, Δ₂::GrassmannTangent;
               metric = :euclidean)
    @assert metric == :euclidean
    Δ₁ === Δ₂ ? norm(Δ₁)^2 : real(dot(Δ₁, Δ₂))
end

"""
    retract(W::AbstractTensorMap, Δ::GrassmannTangent, α; alg = nothing)

Retract isometry `W == base(Δ)` within the Grassmann manifold using tangent vector `Δ.Z`.
If the singular value decomposition of `Z` is given by `U * S * V`, then the resulting
isometry is

`W′ = W * V' * cos(α*S) * V + U * sin(α * S) * V`

while the local tangent vector along the retraction curve is

`Z′ = - W * V' * sin(α*S) * S * V + U * cos(α * S) * S * V'`.
"""
function retract(W::AbstractTensorMap, Δ::GrassmannTangent, α; alg = nothing)
    W == base(Δ) || throw(ArgumentError("not a valid tangent vector at base point"))
    U, S, V = Δ.U, Δ.S, Δ.V
    WVd = W*V'
    sSV, cSV = _sincosSV(α, S, V) # sin(S)*V, cos(S)*V
    W′ = projectisometric!(WVd*cSV + U*sSV)
    sSSV = _lmul!(S, sSV) # sin(S)*S*V
    cSSV = _lmul!(S, cSV) # cos(S)*S*V
    Z′ = projectcomplement!(-WVd*sSSV + U*cSSV, W′)
    return W′, GrassmannTangent(W′, Z′)
end

"""
    Grassmann.invretract(Wold::AbstractTensorMap, Wnew::AbstractTensorMap; alg = nothing)

Return the Grassmann tangent `Z` and unitary `Y` such that `retract(Wold, Z, 1) * Y ≈ Wnew`.

This is done by solving the equation `Wold * V' * cos(S) * V + U * sin(S) * V = Wnew * Y'`
for the isometries `U`, `V`, and `Y`, and the diagonal matrix `S`, and returning
`Z = U * S * V` and `Y`.
"""
function invretract(Wold::AbstractTensorMap, Wnew::AbstractTensorMap; alg = nothing)
    space(Wold) == space(Wnew) || throw(SectorMismatch())
    WodWn = Wold' * Wnew # V' * cos(S) * V * Y
    Wneworth = Wnew - Wold * WodWn
    Vd, cS, VY = tsvd!(WodWn)
    Scmplx = acos(cS)
    # acos always returns a complex TensorMap. We cast back to real if possible.
    S = eltype(WodWn) <: Real && isreal(sectortype(Scmplx)) ? real(Scmplx) : Scmplx
    UsS = Wneworth * VY' # U * sin(S) # should be in polar decomposition form
    U = projectisometric!(UsS; alg = Polar())
    Y = Vd*VY
    V = Vd'
    Z = Grassmann.GrassmannTangent(Wold, U * S * V)
    return Z, Y
end

"""
    relativegauge(W::AbstractTensorMap, V::AbstractTensorMap)

Return the unitary Y such that V*Y and W are "in the same Grassmann gauge" (technical term
from fibre bundles: in the same section), such that they can be related by a Grassmann
retraction.
"""
function relativegauge(W::AbstractTensorMap, V::AbstractTensorMap)
    space(W) == space(V) || throw(SectorMismatch())
    return projectisometric!(V'*W; alg = Polar())
end

function transport!(Θ::GrassmannTangent, W::AbstractTensorMap, Δ::GrassmannTangent, α, W′;
                    alg = nothing)
    W == checkbase(Δ,Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    U, S, V = Δ.U, Δ.S, Δ.V
    WVd = W*V'
    UdΘ = U'*Θ.Z
    sSUdθ, cSUdθ = _sincosSV(α, S, UdΘ) # sin(S)*U'*Θ, cos(S)*U'*Θ
    cSm1UdΘ = axpy!(-1, UdΘ, cSUdθ) # (cos(S)-1)*U'*Θ
    Z′ = axpy!(true, U*cSm1UdΘ - WVd*sSUdθ, Θ.Z)
    Z′ = projectcomplement!(Z′, W′)
    return GrassmannTangent(W′, Z′)
end
function transport(Θ::GrassmannTangent, W::AbstractTensorMap, Δ::GrassmannTangent, α, W′;
                   alg = nothing)
    return transport!(copy(Θ), W, Δ, α, W′; alg = alg)
end

# auxiliary methods: unsafe, no checking
# compute sin(α*S)*V and cos(α*S)*V, where S is assumed diagonal
function _sincosSV(α::Real, S::AbstractTensorMap, V::AbstractTensorMap)
    # S is assumed diagonal
    cSV = similar(V)
    sSV = similar(V)
    @inbounds for (c,bS) in blocks(S)
        bcSV = block(cSV,c)
        bsSV = block(sSV,c)
        bV = block(V,c)
        Threads.@threads for j = 1:size(bV,2)
            @simd for i = 1:size(bV, 1)
                sS, cS = sincos(α*bS[i,i])
                # TODO: we are computing sin and cos above within the loop over j, while it is independent; moving it out the loop requires extra storage though.
                bsSV[i,j] = sS*bV[i,j]
                bcSV[i,j] = cS*bV[i,j]
            end
        end
    end
    return sSV, cSV
end
# multiply V with diagonal S in place, S is assumed diagonal
function _lmul!(S::AbstractTensorMap, V::AbstractTensorMap)
    @inbounds for (c,bS) in blocks(S)
        bV = block(V,c)
        Threads.@threads for j = 1:size(bV,2)
            @simd for i = 1:size(bV, 1)
                bV[i,j] *= bS[i,i]
            end
        end
    end
    return V
end

end
