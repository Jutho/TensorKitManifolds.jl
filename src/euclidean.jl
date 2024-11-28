using Manifolds
using ManifoldsBase
using VectorInterface: One, VectorInterface

ManifoldsBase.TypeParameter(V::HomSpace) = ManifoldsBase.TypeParameter{V}()
ManifoldsBase.get_parameter(::ManifoldsBase.TypeParameter{V}) where {V<:HomSpace} = V

function Manifolds.Euclidean(V::HomSpace; field=ManifoldsBase.ℝ, parameter::Symbol=:field)
    V′ = Manifolds.wrap_type_parameter(parameter, V)
    return Euclidean{typeof(V′),field}(V′)
end

Base.@propagate_inbounds function Manifolds.distance(M::Euclidean, p::AbstractTensorMap,
                                                     q::AbstractTensorMap)
    return LinearAlgebra.norm(p - q)
end

Base.exp(::Euclidean, p::AbstractTensorMap, X::AbstractTensorMap, t::Real) = add(p, X, t)
function Manifolds.exp!(::Euclidean, q::AbstractTensorMap, p::AbstractTensorMap,
                        X::AbstractTensorMap, t::Number)
    return add!(scale!(q, p, One()), X, t)
end

function Manifolds.inner(::Euclidean, p::AbstractTensorMap, X::AbstractTensorMap,
                         Y::AbstractTensorMap)
    return VectorInterface.inner(X, Y)
end

Base.log(::Euclidean, p::AbstractTensorMap, q::AbstractTensorMap) = q - p
function Manifolds.log!(::Euclidean, X::AbstractTensorMap, p::AbstractTensorMap,
                        q::AbstractTensorMap)
    return add!(scale!(X, q, One()), p, -One())
end

function Manifolds.manifold_dimension(M::Euclidean{<:HomSpace,𝔽}) where {𝔽}
    return dim(get_parameter(M.size)) * Manifolds.real_dimension(𝔽)
end

Base.copyto!(p::AbstractTensorMap, q::AbstractTensorMap) = copy!(p, q)

function ManifoldsBase.allocate_result_array(M::AbstractManifold, f, T::Type, V::HomSpace)
    return TensorMap{T}(undef, V)
end
