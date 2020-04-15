module Stiefel

# isometric W
# tangent vectors Δ = W*A + Z where Z = W⟂*B = Q*R = U*S*V
# with thus A' = -A, W'*Z = 0, W'*U = 0

using TensorKit
import TensorKit: similarstoragetype, fusiontreetype, StaticLength, SectorDict
import ..TensorKitManifolds: base, checkbase, projecthermitian!, projectantihermitian!

# special type to store tangent vectors using A and Z = W⟂*B,
# add SVD of Z = U*S*V upon first creation, as well as A2 = [V*A*V' -S; S 0]
mutable struct StiefelTangent{T<:AbstractTensorMap,
                                TA<:AbstractTensorMap,
                                TU<:AbstractTensorMap,
                                TS<:AbstractTensorMap,
                                TV<:AbstractTensorMap,
                                TA2<:AbstractTensorMap}
    W::T
    A::TA
    Z::T
    U::Union{Nothing,TU}
    S::Union{Nothing,TS}
    V::Union{Nothing,TV}
    A2::Union{Nothing,TA2} # A2 = [V*A*V' -S; S 0]
    function StiefelTangent(W::AbstractTensorMap{S,N₁,N₂},
                            A::AbstractTensorMap{S,N₂,N₂},
                            Z::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
        T = typeof(W)
        TA = typeof(A)
        TT = promote_type(float(eltype(W)), eltype(A), eltype(Z))
        G = sectortype(W)
        M = similarstoragetype(W, TT)
        Mr = similarstoragetype(W, real(TT))
        if G === Trivial
            TU = TensorMap{S,N₁,1,G,M,Nothing,Nothing}
            TS = TensorMap{S,1,1,G,Mr,Nothing,Nothing}
            TV = TensorMap{S,1,N₂,G,M,Nothing,Nothing}
            TA2 = TensorMap{S,1,1,G,M,Nothing,Nothing}
            return new{T,TA,TU,TS,TV,TA2}(W, A, Z, nothing, nothing, nothing, nothing)
        else
            F = fusiontreetype(G, StaticLength(1))
            F1 = fusiontreetype(G, StaticLength(N₁))
            F2 = fusiontreetype(G, StaticLength(N₂))
            D = SectorDict{G,M}
            Dr = SectorDict{G,Mr}
            TU = TensorMap{S,N₁,1,G,D,F1,F}
            TS = TensorMap{S,1,1,G,Dr,F,F}
            TV = TensorMap{S,1,N₂,G,D,F,F2}
            TA2 = TensorMap{S,1,1,G,D,F,F}
            return new{T,TA,TU,TS,TV,TA2}(W, A, Z, nothing, nothing, nothing, nothing)
        end
    end
end
function Base.copy(Δ::StiefelTangent)
    Δ′ = StiefelTangent(Δ.W, copy(Δ.A), copy(Δ.Z))
    if Base.getfield(Δ, :U) !== nothing
        Base.setfield!(Δ′, :U, copy(Base.getfield(Δ, :U)))
        Base.setfield!(Δ′, :S, copy(Base.getfield(Δ, :S)))
        Base.setfield!(Δ′, :V, copy(Base.getfield(Δ, :V)))
        Base.setfield!(Δ′, :A2, copy(Base.getfield(Δ, :A2)))
    end
    return Δ′
end

Base.getindex(Δ::StiefelTangent) = Δ.W * Δ.A + Δ.Z
base(Δ::StiefelTangent) = Δ.W
checkbase(Δ₁::StiefelTangent, Δ₂::StiefelTangent) = Δ₁.W == Δ₂.W ? Δ₁.W :
    throw(ArgumentError("tangent vectors with different base points"))

function Base.getproperty(Δ::StiefelTangent, sym::Symbol)
    if sym ∈ (:W, :A, :Z)
        return Base.getfield(Δ, sym)
    elseif sym ∈ (:U, :S, :V, :A2)
        v = Base.getfield(Δ, sym)
        v !== nothing && return v
        U, S, V, = tsvd(Δ.Z)
        Base.setfield!(Δ, :U, U)
        Base.setfield!(Δ, :S, S)
        Base.setfield!(Δ, :V, V)
        A = Δ.A
        A2 = catdomain(catcodomain(V*A*V', S), catcodomain(-S, zero(S)))
        Base.setfield!(Δ, :A2, A2)
        v = Base.getfield(Δ, sym)
        @assert v !== nothing
        return v
    else
        error("type StiefelTangent has no field $sym")
    end
end
function Base.setproperty!(Δ::StiefelTangent, sym::Symbol, v)
    error("type StiefelTangent does not allow to change its fields")
end

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
    if Base.getfield(Δ, :S) !== nothing
        rmul!(Δ.S, α)
        rmul!(Δ.A2, α)
    end
    return Δ
end
function TensorKit.lmul!(α::Real, Δ::StiefelTangent)
    lmul!(α, Δ.A)
    lmul!(α, Δ.Z)
    if Base.getfield(Δ, :S) !== nothing
        lmul!(α, Δ.S)
        lmul!(α, Δ.A2)
    end
    return Δ
end
function TensorKit.axpy!(α::Real, Δx::StiefelTangent, Δy::StiefelTangent)
    checkbase(Δx, Δy)
    axpy!(α, Δx.A, Δy.A)
    axpy!(α, Δx.Z, Δy.Z)
    Base.setfield!(Δy, :U, nothing)
    Base.setfield!(Δy, :S, nothing)
    Base.setfield!(Δy, :V, nothing)
    Base.setfield!(Δy, :A2, nothing)
    return Δy
end
function TensorKit.axpby!(α::Real, Δx::StiefelTangent, β::Real, Δy::StiefelTangent)
    checkbase(Δx, Δy)
    axpby!(α, Δx.A, β, Δy.A)
    axpby!(α, Δx.Z, β, Δy.Z)
    Base.setfield!(Δy, :U, nothing)
    Base.setfield!(Δy, :S, nothing)
    Base.setfield!(Δy, :V, nothing)
    Base.setfield!(Δy, :A2, nothing)
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
        return project_canonical!(W, W)
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
        throw(ArgumentError("unknown algorithm: `alg = $metric`"))
    end
end
function transport!(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent, α::Real, W′;
                    alg = :exp)
    if alg == :exp
        return transport_exp!(Θ, W, Δ, α, W′)
    elseif alg == :cayley
        return transport_cayley!(Θ, W, Δ, α, W′)
    else
        throw(ArgumentError("unknown algorithm: `alg = $metric`"))
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
    return StiefelTangent(W, A, Z)
end
project_canonical(X, W) = project_canonical!(copy(X), W)

# geodesic retraction for canonical metric using exponential
# can be computed efficiently: O(np^2) + O(p^3)
function retract_exp(W::AbstractTensorMap, Δ::StiefelTangent, α::Real)
    W == base(Δ) || throw(ArgumentError("not a valid tangent vector at base point"))
    A, Z, U, S, V, A2 = Δ.A, Δ.Z, Δ.U, Δ.S, Δ.V, Δ.A2
    UU = catdomain(W*V', U)
    VV = catcodomain(V, zero(V))
    SVV = catcodomain(zero(V), S*V)
    E = exp(α*A2)
    # UU′, = leftorth!(UU*E; alg = QRpos()) # additional QRpos for stability
    UU′ = UU*E # no additional QRpos because it changes domain
    W′ = UU′*VV
    A′ = A
    Z′ = UU′*SVV
    return W′, StiefelTangent(W′, A′, Z′)
end

# vector transport compatible with above `retract`: also differentiated retraction
# isometric for both euclidean and canonical metric
# not parallel transport for either metric as the corresponding connection has torsion
# can be computed efficiently: O(np^2) + O(p^3)
function transport_exp!(Θ::StiefelTangent, W::AbstractTensorMap,
                        Δ::StiefelTangent, α::Real, W′)
    W == checkbase(Δ,Θ) || throw(ArgumentError("not a valid tangent vector at base point"))
    U, S, V, A2 = Δ.U, Δ.S, Δ.V, Δ.A2
    UU = catdomain(W*V', U)
    P = catcodomain(zero(S), one(S))
    E = exp(α*A2)

    A′ = Θ.A
    Z′ = Θ.Z + UU*(((E-one(E))*P)*(U'*Θ.Z))
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
    W′ = (2*W+α*Z)*iX - W
    A′ = projectantihermitian!((A - (α/2)*ZdZ)*iX)
    Z′ = (Z-α*(W+α/2*Z)*(iX*ZdZ))
    Z′ = Z′*projecthermitian!(iX)
    return W′, StiefelTangent(W′, A′, Z′)
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
    Z′ = axpy!(-α, (W+(α/2)*Z)*(X\ZdZ), Θ.Z)
    return StiefelTangent(W′, A′, Z′)
end
transport_cayley(Θ::StiefelTangent, W::AbstractTensorMap, Δ::StiefelTangent, α::Real, W′) =
    transport_cayley!(copy(Θ), W, Δ, α, W′)

end
