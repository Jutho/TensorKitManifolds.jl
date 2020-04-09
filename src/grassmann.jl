module Grassmann

# isometric W (representative of equivalence class)
# tangent vectors Δ = Z where Z = W⟂*B = Q*R = U*S*V
# with thus W'*Z = 0, W'*U = 0

using TensorKit
import TensorKit: similarstoragetype, fusiontreetype, StaticLength, SectorDict
import ..TensorKitManifolds: base, checkbase, projectantihermitian!

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
        if G === Trivial
            TU = TensorMap{S,N₁,1,G,M,Nothing,Nothing}
            TS = TensorMap{S,1,1,G,Mr,Nothing,Nothing}
            TV = TensorMap{S,1,N₂,G,M,Nothing,Nothing}
            return new{T,TU,TS,TV}(W, Z, nothing, nothing, nothing)
        else
            F = fusiontreetype(G, StaticLength(1))
            F1 = fusiontreetype(G, StaticLength(N₁))
            F2 = fusiontreetype(G, StaticLength(N₂))
            D = SectorDict{G,M}
            Dr = SectorDict{G,Mr}
            TU = TensorMap{S,N₁,1,G,D,F1,F}
            TS = TensorMap{S,1,1,G,Dr,F,F}
            TV = TensorMap{S,1,N₂,G,D,F,F2}
            return new{T,TU,TS,TV}(W, Z, nothing, nothing, nothing)
        end
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
checkbase(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) = Δ₁.W === Δ₂.W ? Δ₁.W :
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

function TensorKit.rmul!(Δ::GrassmannTangent, α::Number)
    rmul!(Δ.Z, α)
    if Base.getfield(Δ, :S) !== nothing
        rmul!(Δ.S, α)
    end
    return Δ
end
function TensorKit.lmul!(α::Number, Δ::GrassmannTangent)
    lmul!(α, Δ.Z)
    if Base.getfield(Δ, :S) !== nothing
        lmul!(α, Δ.S)
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
function project!(X::AbstractTensorMap, W::AbstractTensorMap)
    P = W'*X
    Δ = mul!(X, W, P, -1, 1)
    return GrassmannTangent(W, Δ)
end
project(X, W) = project!(copy(X), W)

function inner(W::AbstractTensorMap, Δ₁::GrassmannTangent, Δ₂::GrassmannTangent)
    Δ₁ === Δ₂ ? norm(Δ₁)^2 : real(dot(Δ₁, Δ₂))
end

function retract(W::AbstractTensorMap, Δ::GrassmannTangent, α)
    W === base(Δ) || throw(ArgumentError("not a valid tangent vector at base point"))
    U, S, V = Δ.U, Δ.S, Δ.V
    WVd = W*V'
    cS = cos(α*S)
    sS = sin(α*S)
    cSV = cS*V
    sSV = sS*V
    SV = S*V
    cSSV = cS*SV
    sSSV = sS*SV
    # W′, = leftorth!(WVd*cSV + U*sSV; alg = QRpos()) # additional QRpos for stability
    W′ = WVd*cSV + U*sSV # no additional QRpos since this changes the domain of W′
    return W′, GrassmannTangent(W′, -WVd*sSSV + U*cSSV)
end

function transport!(Θ::GrassmannTangent, W::AbstractTensorMap, Δ::GrassmannTangent, α, W′)
    W === checkbase(Δ,Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    U, S, V = Δ.U, Δ.S, Δ.V
    WVd = W*V'
    cS = cos(α*S)
    sS = sin(α*S)
    UdΘ = U'*Θ.Z
    return GrassmannTangent(W′, axpy!(true, U*((cS-one(cS))*UdΘ) - WVd*(sS*UdΘ), Θ.Z))
end
transport(Θ::GrassmannTangent, W::AbstractTensorMap, Δ::GrassmannTangent, α, W′) =
    transport!(copy(Θ), W, Δ, α, W′)

end
