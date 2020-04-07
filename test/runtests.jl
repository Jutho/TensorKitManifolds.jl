using TensorKitIsometries
using TensorKit
using Test

spaces = (ℂ^4, ℤ₂Space(2,2), U₁Space(0=>2,1=>1,-1=>1), SU₂Space(0=>2,1/2=>1))

@testset "Grassmann with space $V" for V in spaces
    for T in (Float64, ComplexF64)
        W, = leftorth(TensorMap(randn, T, V*V*V, V*V))
        X = TensorMap(randn, T, space(W))
        Y = TensorMap(randn, T, space(W))
        Δ = Grassmann.project(X, W)
        Θ = Grassmann.project(Y, W)
        α = randn(T)
        Ξ = -Δ + α*Θ
        @test norm(W'*Δ[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Θ[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Ξ[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(dot(Δ[], Θ[]))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(dot(X, Θ[]))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(dot(Δ[],Y))
        @test Grassmann.inner(W, Δ, Δ) ≈ norm(Δ[])^2
        W2, Δ2 = Grassmann.retract(W, Δ, 1e-5)
        @test W2 ≈ W + 1e-5*Δ[]
        W2, Δ2 = Grassmann.retract(W, Δ, 0.75)
        @test norm(W2'*Δ2[]) <= sqrt(eps(real(T)))*dim(domain(W))
        Θ2 = Grassmann.transport(Θ, W, Δ, 0.75, W2)
        Ξ2 = Grassmann.transport(Ξ, W, Δ, 0.75, W2)
        @test norm(W2'*Θ2[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + α * Θ2[]
        @test Grassmann.inner(W2, Δ2, Θ2) ≈ Grassmann.inner(W, Δ, Θ)
        @test Grassmann.inner(W2, Ξ2, Θ2) ≈ Grassmann.inner(W, Ξ, Θ)
    end
end


@testset "Stiefel with space $V" for V in spaces
    for T in (Float64, ComplexF64)
        W, = leftorth(TensorMap(randn, T, V*V*V, V*V))
        X = TensorMap(randn, T, space(W))
        Y = TensorMap(randn, T, space(W))
        Δ = Stiefel.project_euclidean(X, W)
        Θ = Stiefel.project_canonical(Y, W)
        α = rand()
        Ξ = -Δ + α*Θ
        @test norm(W'*Δ[] + Δ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Θ[] + Θ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Ξ[] + Ξ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(dot(Δ[], Θ[]))
        @test Stiefel.inner_canonical(W, Δ, Θ) ≈ real(dot(Δ[], Θ[] - W*(W'*Θ[])/2))
        @test Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(dot(X, Θ[]))
        @test !(Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(dot(Δ[],Y)))
        @test !(Stiefel.inner_canonical(W, Δ, Θ) ≈ real(dot(X, Θ[])))
        @test Stiefel.inner_canonical(W, Δ, Θ) ≈ real(dot(Δ[],Y))
        @test Stiefel.inner_euclidean(W, Δ, Δ) ≈ norm(Δ[])^2
        @test Stiefel.inner_canonical(W, Δ, Δ) ≈ (1//2)*norm(W'*Δ[])^2 + norm(Δ[]-W*(W'Δ[]))^2
        W2, Δ2 = Stiefel.retract(W, Δ, 1e-5)
        @test W2 ≈ W + 1e-5*Δ[]
        W2, Δ2 = Stiefel.retract(W, Δ, 0.75)
        @test norm(W2'*Δ2[] + Δ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        Θ2 = Stiefel.transport(Θ, W, Δ, 0.75, W2)
        Ξ2 = Stiefel.transport(Ξ, W, Δ, 0.75, W2)
        @test norm(W2'*Θ2[] + Θ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[] + Ξ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + α * Θ2[]
        @test Stiefel.inner_euclidean(W2, Δ2, Θ2) ≈ Stiefel.inner_euclidean(W, Δ, Θ)
        @test Stiefel.inner_euclidean(W2, Ξ2, Θ2) ≈ Stiefel.inner_euclidean(W, Ξ, Θ)
        @test Stiefel.inner_canonical(W2, Δ2, Θ2) ≈ Stiefel.inner_canonical(W, Δ, Θ)
        @test Stiefel.inner_canonical(W2, Ξ2, Θ2) ≈ Stiefel.inner_canonical(W, Ξ, Θ)
    end
end
