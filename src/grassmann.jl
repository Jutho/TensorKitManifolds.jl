module Grassmann

# isometric W (representative of equivalence class)
# tangent vectors Δ = W⟂ * B = Q * R = U * S * V with W'*Z = W'*U = 0
# tangent vectors are represented as either `Δ`, `(Q,R)` or `U, S, V`
using TensorKit
import TensorKit: similarstoragetype, fusiontreetype, StaticLength

mutable struct GrassmannTangent{T<:AbstractTensorMap,T1,T2,T3}
    tangent::T
    svd::Union{Nothing,Tuple{T1,T2,T3}}
    function GrassmannTangent(t::AbstractTensorMap{S,N₁,N₂}) where {S,N₁,N₂}
        T = typeof(t)
        TT = float(eltype(t))
        G = sectortype(t)
        if G === Trivial
            T1 = TensorMap{S,N₁,1,G,similarstoragetype(t,TT),Nothing,Nothing}
            T2 = TensorMap{S,1,1,G,similarstoragetype(t,real(TT)),Nothing,Nothing}
            T3 = TensorMap{S,1,N₂,G,similarstoragetype(t,TT),Nothing,Nothing}
            return new{T,T1,T2,T3}(t, nothing)
        else
            F1 = fusiontreetype(G, StaticLength(N₁))
            F3 = fusiontreetype(G, StaticLenght(1))
            F3 = fusiontreetype(G, StaticLength(N₂))
            T1 = TensorMap{S,N₁,1,G,SectorDict{G,similarstoragetype(t,TT)},F1,F2}
            T2 = TensorMap{S,1,1,G,SectorDict{G,similarstoragetype(t,real(TT))},F2,F2}
            T3 = TensorMap{S,1,N₂,G,SectorDict{G,similarstoragetype(t,TT)},F2,F3}
            return new{T,T1,T2,T3}(t, nothing)
        end
    end
end
Base.getindex(Δ::GrassmannTangent) = Δ.tangent
function TensorKit.tsvd(Δ::GrassmannTangent)
    if Δ.svd === nothing
        U,S,V, = tsvd(Δ[])
        Δ.svd = (U,S,V)
    end
    return Δ.svd
end

# Basic vector space behaviour
Base.:+(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) = GrassmannTangent(Δ₁[] + Δ₂[])
Base.:-(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) = GrassmannTangent(Δ₁[] - Δ₂[])
Base.:-(Δ::GrassmannTangent) = (-1)*Δ

Base.:*(Δ::GrassmannTangent, α::Number) = rmul!(deepcopy(Δ), α)
Base.:*(α::Number, Δ::GrassmannTangent) = lmul!(α, deepcopy(Δ))
Base.:/(Δ::GrassmannTangent, α::Number) = rmul!(deepcopy(Δ), inv(α))
Base.:\(α::Number, Δ::GrassmannTangent) = lmul!(inv(α), deepcopy(Δ))

function TensorKit.rmul!(t::GrassmannTangent, α::Number)
    rmul!(t[], α)
    if t.svd !== nothing
        rmul!(t.svd[2], α)
    end
    return t
end
function TensorKit.lmul!(α::Number, t::GrassmannTangent)
    lmul!(α, t[])
    if t.svd !== nothing
        lmul!(α, t.svd[2])
    end
    return t
end
function TensorKit.axpy!(α::Number, tx::GrassmannTangent, ty::GrassmannTangent)
    axpy!(α, tx[], ty[])
    ty.svd = nothing
    return ty
end
function TensorKit.axpby!(α::Number, tx::GrassmannTangent, β::Number, ty::GrassmannTangent)
    axpy!(α, tx[], β, ty[])
    ty.svd = nothing
    return ty
end

TensorKit.dot(Δ₁::GrassmannTangent, Δ₂::GrassmannTangent) = dot(Δ₁[], Δ₂[])
TensorKit.norm(Δ::GrassmannTangent, p::Real = 2) = norm(Δ[], p)

# tangent space methods

function project!(X::AbstractTensorMap, W::AbstractTensorMap)
    P = W'*X
    Δ = mul!(X, W, P, -1, 1)
    return GrassmannTangent(Δ)
end
project(X, W) = project!(copy(X), W)

function inner(W::AbstractTensorMap, Δ₁::GrassmannTangent, Δ₂::GrassmannTangent)
    Δ₁ === Δ₂ ? norm(Δ₁)^2 : real(dot(Δ₁,Δ₂))
end

function retract(W::AbstractTensorMap, Δ::GrassmannTangent, α)
    U,S,V = tsvd(Δ)
    WVd = W*V'
    cS = cos(α*S)
    sS = sin(α*S)
    cSV = cS*V
    sSV = sS*V
    SV = S*V
    cSSV = cS*SV
    sSSV = sS*SV
    return (WVd*cSV + U*sSV), GrassmannTangent(-WVd*sSSV + U*cSSV)
end

function transport(Θ::GrassmannTangent, W::AbstractTensorMap, Δ::GrassmannTangent, α, W′)
    U,S,V = tsvd(Δ)
    WVd = W*V'
    cS = cos(α*S)
    sS = sin(α*S)
    UdΘ = U'*Θ[]
    return GrassmannTangent(Θ[] + U*((cS-one(cS))*UdΘ) - WVd*(sS*UdΘ))
end

end
