import LinearAlgebra: diagind

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
