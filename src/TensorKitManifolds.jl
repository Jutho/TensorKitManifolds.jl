module TensorKitManifolds

export base, checkbase, isisometry, isunitary
export projecthermitian, projecthermitian!, projectantihermitian, projectantihermitian!
export Grassmann, Stiefel, Unitary
export inner, retract, transport, transport!

using TensorKit, Strided

# Every submodule -- Grassmann, Stiefel, and Unitary -- implements their own methods for
# these. The signatures should be
# inner(W, Δ₁::Tangent, Δ₂::Tangent; metric)
# retract(W, Δ::Tangent, α::Real; alg)
# transport(Θ::Tangent, W, Δ::Tangent, α::Real, W′; alg)
# where the keyword arguments `alg` and `metric` should always be accepted, even if there is
# only one option for them and they are ignored. The `Tangent` is just a placeholder for the
# tangent type of each manifold. Similarly each submodule defines a `project!` function,
# which too should accept a keyword argument `metric`, even if it is ignored.
function inner end
function retract end
function transport end
function transport! end
function base end
function checkbase end
checkbase(x, y, z, args...) = checkbase(checkbase(x, y), z, args...)

function isisometry(W::AbstractTensorMap; tol = eps(real(eltype(W))))
    WdW = W'*W
    s = zero(float(real(eltype(W))))
    for (c,b) in blocks(WdW)
        _subtractone!(b)
        s += dim(c)*length(b)
    end
    return norm(WdW) <= tol*sqrt(s)
end

isunitary(W::AbstractTensorMap; tol = eps(real(eltype(W)))) =
    isisometry(W) && isisometry(W')

function projecthermitian!(W::AbstractTensorMap)
    codomain(W) == domain(W) ||
        throw(DomainError("Tensor with distinct domain and codomain cannot be hermitian."))
    for (c,b) in blocks(W)
        _projecthermitian!(b)
    end
    return W
end
function projectantihermitian!(W::AbstractTensorMap)
    codomain(W) == domain(W) ||
        throw(DomainError("Tensor with distinct domain and codomain cannot be anithermitian."))
    for (c,b) in blocks(W)
        _projectantihermitian!(b)
    end
    return W
end
function projectisometric!(W::AbstractTensorMap;
                            alg::TensorKit.OrthogonalFactorizationAlgorithm = QRpos())
    Q, = leftorth!(W; alg = alg)
    return TensorMap(Q.data, space(W))
end
function projectcomplement!(X::AbstractTensorMap, W::AbstractTensorMap;
                                tol = max(10*eps(real(eltype(X))), eps(norm(X))))
    P = W'*X
    nP = norm(P)
    while nP > tol
        X = mul!(X, W, P, -1, 1)
        P = W'*X
        nP = norm(P)
    end
    return X
end

projecthermitian(W::AbstractTensorMap) = projecthermitian!(copy(W))
projectantihermitian(W::AbstractTensorMap) = projectantihermitian!(copy(W))
projectisometric(W::AbstractTensorMap) = projectisometric!(copy(W))
projectcomplement(X::AbstractTensorMap, W::AbstractTensorMap) =
    projectcomplement!(copy(X), W)

include("auxiliary.jl")
include("grassmann.jl")
include("stiefel.jl")
include("unitary.jl")

end
