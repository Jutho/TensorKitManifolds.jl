module TensorKitManifolds

export base, checkbase, isisometry, isunitary
export projecthermitian, projecthermitian!, projectantihermitian, projectantihermitian!
export Grassmann, Stiefel, Unitary

using TensorKit, Strided

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
projecthermitian(W::AbstractTensorMap) = projecthermitian!(copy(W))
projectantihermitian(W::AbstractTensorMap) = projectantihermitian!(copy(W))

include("auxiliary.jl")
include("grassmann.jl")
include("stiefel.jl")
include("unitary.jl")

end
