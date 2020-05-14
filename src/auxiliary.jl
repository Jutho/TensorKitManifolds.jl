using LinearAlgebra

function _projecthermitian!(a::StridedMatrix)
    L = size(a,1)
    for J = 1:16:L
        for I = 1:16:L
            if I == J
                for j = J:min(J+15,L)
                    for i = J:j-1
                        x = (a[i,j] + conj(a[j,i]))/2
                        a[i,j] = x
                        a[j,i] = conj(x)
                    end
                    a[j,j] = (a[j,j] + conj(a[j,j]))/2
                end
            else
                for j = J:min(J+15,L)
                    for i = I:min(I+15,L)
                        x = (a[i,j] + conj(a[j,i]))/2
                        a[i,j] = x
                        a[j,i] = conj(x)
                    end
                end
            end
        end
    end
    return a
end
function _projecthermitian!(a::AbstractMatrix)
    a .= (a .+ a')./2
    return a
end

function _projectantihermitian!(a::StridedMatrix)
    L = size(a,1)
    for J = 1:16:L
        for I = 1:16:L
            if I == J
                for j = J:min(J+15,L)
                    for i = J:j-1
                        x = (a[i,j] - conj(a[j,i]))/2
                        a[i,j] = x
                        a[j,i] = -conj(x)
                    end
                    a[j,j] = (a[j,j] - conj(a[j,j]))/2
                end
            else
                for j = J:min(J+15,L)
                    for i = I:min(I+15,L)
                        x = (a[i,j] - conj(a[j,i]))/2
                        a[i,j] = x
                        a[j,i] = -conj(x)
                    end
                end
            end
        end
    end
    return a
end
function _projectantihermitian!(a::AbstractMatrix)
    a .= (a .- a')./2
    return a
end

function _addone!(a::AbstractMatrix)
    view(a, diagind(a)) .= view(a, diagind(a)) .+ 1
    return a
end
function _subtractone!(a::AbstractMatrix)
    view(a, diagind(a)) .= view(a, diagind(a)) .- 1
    return a
end
function _one!(A::AbstractMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        for i = 1:m
            A[i,j] = i==j
        end
    end
    return A
end

function _polarsdd!(A::StridedMatrix)
    U, S, V = svd!(A; alg = LinearAlgebra.DivideAndConquer())
    return mul!(A, U, V')
end
function _polarsvd!(A::StridedMatrix)
    U, S, V = svd!(A; alg = LinearAlgebra.QRIteration())
    return mul!(A, U, V')
end
function _polarnewton!(A::StridedMatrix; tol = 10*eleps(A), maxiter = 5)
    m, n = size(A)
    @assert m >= n
    A2 = copy(A)
    Q, R = qr!(A2)
    Ri = ldiv!(UpperTriangular(R)', _one!(similar(R)))
    R, Ri = _avgdiff!(R, Ri)
    i = 1
    R2 = view(A, 1:n, 1:n)
    fill!(view(A, n+1:m, 1:n), zero(eltype(A)))
    copyto!(R2, R)
    while maximum(abs, Ri) > tol
        if i == maxiter # if not converged by now, fall back to sdd
            _polarsdd!(Ri)
            break
        end
        Ri = ldiv!(lu!(R2)', _one!(Ri))
        R, Ri = _avgdiff!(R, Ri)
        copyto!(R2, R)
        i += 1
    end
    return lmul!(Q, A)
end
# in place computation of the average and difference of two arrays
function _avgdiff!(A::AbstractArray, B::AbstractArray)
    axes(A) == axes(B) || throw(DimensionMismatch())
    @simd for I in eachindex(A, B)
        @inbounds begin
            a = A[I]
            b = B[I]
            A[I] = (a+b)/2
            B[I] = b-a
        end
    end
    return A, B
end

# _stiefelexp(W, A, Z, α)
# given an isometry W, and a Stiefel tangent vector Δ = W*A + Z, compute the building blocks
# W′, Q′, R′ of the geodesic with respect to the canonical metric in the direction of α*Δ.
# Here, W′is the new isometry, and the local tangent vector is given by
# Δ′ = W′ * A + Z′ with Z′ = Q′*R′
# Here, Q′ is a set of orthogonal columns to the colums in W′.
function _stiefelexp(W::StridedMatrix, A::StridedMatrix, Z::StridedMatrix, α)
    n, p = size(W)
    r = min(2*p, n)
    QQ, _ = qr!([W Z])
    Q = similar(W, n, r-p)
    @inbounds for j = Base.OneTo(r-p)
        for i = Base.OneTo(n)
            Q[i,j] = (i == p+j)
        end
    end
    Q = lmul!(QQ, Q)
    R = Q'*Z
    A2 = similar(A, min(2*p, n), min(2*p, n))
    A2[1:p, 1:p] .= α .* A
    A2[p+1:end, 1:p] .= α .* R
    A2[1:p, p+1:end] .= (-α) .* (R')
    A2[p+1:end, p+1:end] .= 0
    U = [W Q]*exp(A2)
    U = _polarnewton!(U)
    W′ = U[:,1:p]
    Q′ = U[:,p+1:end]
    R′ = R
    return W′, Q, Q′, R′
end

function _stiefellog(Wold::StridedMatrix, Wnew::StridedMatrix;
                        tol = 10*eleps(Wold), maxiter = 100)
    n, p = size(Wold)
    r = min(2*p, n)
    P = Wold'*Wnew
    dW = Wnew - Wold*P
    QQ, _ = qr!([Wold dW])
    Q = similar(Wold, n, r-p)
    @inbounds for j = Base.OneTo(r-p)
        for i = Base.OneTo(n)
            Q[i,j] = (i == p+j)
        end
    end
    Q = lmul!(QQ, Q)
    R = Q'*dW
    Wext = [Wold Q];
    F = qr!([P; R])
    U = lmul!(F.Q, _one!(similar(P, r, r)))
    U[1:p, 1:p] .= P
    U[p+1:r, 1:p] .= R
    X = view(U, 1:p, p+1:r)
    Y = view(U, p+1:r, p+1:r)
    if p < n
        YSVD = svd!(Y)
        mul!(X, X*(YSVD.V), (YSVD.U)')
        UsqrtS = YSVD.U
        @inbounds for j = 1:size(UsqrtS, 2)
            s = sqrt(YSVD.S[j])
            @simd for i = 1:size(UsqrtS, 1)
                UsqrtS[i,j] *= s
            end
        end
        mul!(Y, UsqrtS, UsqrtS')
    end
    logU = _projectantihermitian!(log(U))
    if eltype(U) <: Real
        @assert mapreduce(abs ∘ imag, max, logU; init = abs(zero(eltype(logU))) ) <= tol
        K = real(logU)
    else
        K = logU
    end
    C = view(K, p+1:r, p+1:r)
    i = 1
    τ = mapreduce(abs, max, C; init = abs(zero(eltype(C))))
    while τ > tol
        if i > maxiter
            @warn "Stiefel logarithm: not converged in $maxiter iterations, τ = $τ"
            break
        end
        eC = exp(rmul!(C,-1))
        X .= X*eC
        Y .= Y*eC
        logU = _projectantihermitian!(log(U))
        if eltype(U) <: Real
            @assert mapreduce(abs ∘ imag, max, logU; init = abs(zero(eltype(logU))) ) <= tol
            K .= real.(logU)
        else
            K .= logU
        end
        τ = maximum(abs, C)
        i += 1
    end
    return K[1:p,1:p], Q, K[p+1:r, 1:p]
end
