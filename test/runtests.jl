using TensorKit, TensorKitManifolds
using Test

spaces = (ℂ^4, ℤ₂Space(2,2), U₁Space(0=>2,1=>1,-1=>1), SU₂Space(0=>2,1/2=>1))
const ϵ = 1e-7
const α = 0.75

@testset "Grassmann with space $V" for V in spaces
    for T in (Float64,)
        W, = leftorth(TensorMap(randn, T, V*V*V, V*V); alg = Polar())
        X = TensorMap(randn, T, space(W))
        Y = TensorMap(randn, T, space(W))
        Δ = @inferred Grassmann.project(X, W)
        Θ = Grassmann.project(Y, W)
        γ = randn(T)
        Ξ = -Δ + γ*Θ
        @test norm(W'*Δ[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Θ[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Ξ[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(zero(W)) == 0
        @test (@inferred Grassmann.inner(W, Δ, Θ)) ≈ real(dot(Δ[], Θ[]))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(dot(X, Θ[]))
        @test Grassmann.inner(W, Δ, Θ) ≈ real(dot(Δ[],Y))
        @test Grassmann.inner(W, Δ, Δ) ≈ norm(Δ[])^2

        W2, = @inferred Grassmann.retract(W, Δ, ϵ)
        @test W2 ≈ W + ϵ*Δ[]
        W2, Δ2′ = Grassmann.retract(W, Δ, α)
        @test norm(W2'*Δ2′[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Δ2′[] ≈ (first(Grassmann.retract(W, Δ, α + ϵ/2)) -
                        first(Grassmann.retract(W, Δ, α - ϵ/2)))/(ϵ)
        Δ2 = @inferred Grassmann.transport(Δ, W, Δ, α, W2)
        Θ2 = Grassmann.transport(Θ, W, Δ, α, W2)
        Ξ2 = Grassmann.transport(Ξ, W, Δ, α, W2)
        @test Δ2[] ≈ Δ2′[]
        @test norm(W2'*Δ2[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Θ2[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[]) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Grassmann.inner(W2, Δ2, Θ2) ≈ Grassmann.inner(W, Δ, Θ)
        @test Grassmann.inner(W2, Ξ2, Θ2) ≈ Grassmann.inner(W, Ξ, Θ)

        Wend = TensorMap(randhaar, T, codomain(W), domain(W))
        Δ3, V = Grassmann.invretract(W, Wend)
        @test Wend ≈ retract(W, Δ3, 1)[1] * V
        U = Grassmann.matchgauge(W, Wend)
        V2 = Grassmann.invretract(W, Wend * U)[2]
        @test V2 ≈ one(V2)
    end
end

@testset "Stiefel with space $V" for V in spaces
    for T in (Float64, ComplexF64)
        W, = leftorth(TensorMap(randn, T, V*V*V, V*V); alg = Polar())
        X = TensorMap(randn, T, space(W))
        Y = TensorMap(randn, T, space(W))
        Δ = @inferred Stiefel.project_euclidean(X, W)
        Θ = Stiefel.project_canonical(Y, W)
        γ = rand()
        Ξ = -Δ + γ*Θ
        @test norm(W'*Δ[] + Δ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Θ[] + Θ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Ξ[] + Ξ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(zero(W)) == 0
        @test (@inferred Stiefel.inner_euclidean(W, Δ, Θ)) ≈ real(dot(Δ[], Θ[]))
        @test (@inferred Stiefel.inner_canonical(W, Δ, Θ)) ≈
                                                        real(dot(Δ[], Θ[] - W*(W'*Θ[])/2))
        @test Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(dot(X, Θ[]))
        @test !(Stiefel.inner_euclidean(W, Δ, Θ) ≈ real(dot(Δ[],Y)))
        @test !(Stiefel.inner_canonical(W, Δ, Θ) ≈ real(dot(X, Θ[])))
        @test Stiefel.inner_canonical(W, Δ, Θ) ≈ real(dot(Δ[],Y))
        @test Stiefel.inner_euclidean(W, Δ, Δ) ≈ norm(Δ[])^2
        @test Stiefel.inner_canonical(W, Δ, Δ) ≈ (1//2)*norm(W'*Δ[])^2 + norm(Δ[]-W*(W'Δ[]))^2

        W2, = @inferred Stiefel.retract_exp(W, Δ, ϵ)
        @test W2 ≈ W + ϵ*Δ[]
        W2, Δ2′ = Stiefel.retract_exp(W, Δ, α)
        @test norm(W2'*Δ2′[] + Δ2′[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Δ2′[] ≈ (first(Stiefel.retract_exp(W, Δ, α+ϵ/2)) -
                        first(Stiefel.retract_exp(W, Δ, α-ϵ/2)))/(ϵ)
        Δ2 = @inferred Stiefel.transport_exp(Δ, W, Δ, α, W2)
        Θ2 = Stiefel.transport_exp(Θ, W, Δ, α, W2)
        Ξ2 = Stiefel.transport_exp(Ξ, W, Δ, α, W2)
        @test Δ2′[] ≈ Δ2[]
        @test norm(W2'*Δ2[] + Δ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Θ2[] + Θ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[] + Ξ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Stiefel.inner_euclidean(W2, Δ2, Θ2) ≈ Stiefel.inner_euclidean(W, Δ, Θ)
        @test Stiefel.inner_euclidean(W2, Ξ2, Θ2) ≈ Stiefel.inner_euclidean(W, Ξ, Θ)
        @test Stiefel.inner_canonical(W2, Δ2, Θ2) ≈ Stiefel.inner_canonical(W, Δ, Θ)
        @test Stiefel.inner_canonical(W2, Ξ2, Θ2) ≈ Stiefel.inner_canonical(W, Ξ, Θ)

        W2, = @inferred Stiefel.retract_cayley(W, Δ, ϵ)
        @test W2 ≈ W + ϵ*Δ[]
        W2, Δ2′ = Stiefel.retract_cayley(W, Δ, α)
        @test norm(W2'*Δ2′[] + Δ2′[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Δ2′[] ≈ (first(Stiefel.retract_cayley(W, Δ, α+ϵ/2)) -
                        first(Stiefel.retract_cayley(W, Δ, α-ϵ/2)))/(ϵ)
        @test norm(Δ2′) <= norm(Δ)
        Δ2 = @inferred Stiefel.transport_cayley(Δ, W, Δ, α, W2)
        Θ2 = Stiefel.transport_cayley(Θ, W, Δ, α, W2)
        Ξ2 = Stiefel.transport_cayley(Ξ, W, Δ, α, W2)
        @test !(Δ2′[] ≈ Δ2[])
        @test norm(W2'*Δ2[] + Δ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Θ2[] + Θ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[] + Ξ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Stiefel.inner_euclidean(W2, Δ2, Θ2) ≈ Stiefel.inner_euclidean(W, Δ, Θ)
        @test Stiefel.inner_euclidean(W2, Ξ2, Θ2) ≈ Stiefel.inner_euclidean(W, Ξ, Θ)
        @test Stiefel.inner_canonical(W2, Δ2, Θ2) ≈ Stiefel.inner_canonical(W, Δ, Θ)
        @test Stiefel.inner_canonical(W2, Ξ2, Θ2) ≈ Stiefel.inner_canonical(W, Ξ, Θ)
    end
end

@testset "Unitary with space $V" for V in spaces
    for T in (Float64, ComplexF64)
        W, = leftorth(TensorMap(randn, T, V*V*V, V*V); alg = Polar())
        X = TensorMap(randn, T, space(W))
        Y = TensorMap(randn, T, space(W))
        Δ = @inferred Unitary.project(X, W)
        Θ = Unitary.project(Y, W)
        γ = randn()
        Ξ = -Δ + γ*Θ
        @test norm(W'*Δ[] + Δ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Θ[] + Θ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W'*Ξ[] + Ξ[]'*W) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(zero(W)) == 0
        @test (@inferred Unitary.inner(W, Δ, Θ)) ≈ real(dot(Δ[], Θ[]))
        @test Unitary.inner(W, Δ, Θ) ≈ real(dot(X, Θ[]))
        @test Unitary.inner(W, Δ, Θ) ≈ real(dot(Δ[],Y))
        @test Unitary.inner(W, Δ, Δ) ≈ norm(Δ[])^2

        W2, = @inferred Unitary.retract(W, Δ, ϵ)
        @test W2 ≈ W + ϵ*Δ[]
        W2, Δ2′ = Unitary.retract(W, Δ, α)
        @test norm(W2'*Δ2′[] + Δ2′[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Δ2′[] ≈ (first(Unitary.retract(W, Δ, α+ϵ/2)) -
                        first(Unitary.retract(W, Δ, α-ϵ/2)))/(ϵ)

        Δ2 = @inferred Unitary.transport_parallel(Δ, W, Δ, α, W2)
        Θ2 = Unitary.transport_parallel(Θ, W, Δ, α, W2)
        Ξ2 = Unitary.transport_parallel(Ξ, W, Δ, α, W2)
        @test Δ2′[] ≈ Δ2[]
        @test norm(W2'*Θ2[] + Θ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[] + Ξ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Unitary.inner(W2, Δ2, Θ2) ≈ Unitary.inner(W, Δ, Θ)
        @test Unitary.inner(W2, Ξ2, Θ2) ≈ Unitary.inner(W, Ξ, Θ)

        Δ2 = @inferred Unitary.transport_stiefel(Δ, W, Δ, α, W2)
        Θ2 = Unitary.transport_stiefel(Θ, W, Δ, α, W2)
        Ξ2 = Unitary.transport_stiefel(Ξ, W, Δ, α, W2)
        @test Δ2′[] ≈ Δ2[]
        @test norm(W2'*Δ2[] + Δ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Θ2[] + Θ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test norm(W2'*Ξ2[] + Ξ2[]'*W2) <= sqrt(eps(real(T)))*dim(domain(W))
        @test Ξ2[] ≈ -Δ2[] + γ * Θ2[]
        @test Unitary.inner(W2, Δ2, Θ2) ≈ Unitary.inner(W, Δ, Θ)
        @test Unitary.inner(W2, Ξ2, Θ2) ≈ Unitary.inner(W, Ξ, Θ)
    end
end
