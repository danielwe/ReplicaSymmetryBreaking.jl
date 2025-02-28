## Order parameters
struct Orderparameters{T<:AbstractFloat}
    a::Vector{T}
    sqrtdq::Vector{T}

    function Orderparameters(a::AbstractVector{T}, sqrtdq::AbstractVector{T}) where {T}
        if !(length(sqrtdq) == (length(a) + 2))
            throw(ArgumentError("must have length(sqrtdq) == (length(a) + 2)"))
        end
        return new{T}(a, sqrtdq)
    end
end

# Heuristic initial guess for problem
function Orderparameters(problem::AbstractRSBProblem{T,inf}) where {T,inf}
    (; k, β) = problem
    a = range(oneunit(β), 0, k + 1)[begin:(end-1)]
    q = if inf
        [oneunit(T); range(oneunit(T), zero(T), k + 2)]
    else
        range(oneunit(T), zero(T), k + 3)
    end
    sqrtdq = sqrt.(-diff(q))
    op = Orderparameters(a, sqrtdq)
    checkorderparameters(op, problem)
    return op
end

# Validate order parameters
function checkorderparameters(
    op::Orderparameters{T}, problem::AbstractRSBProblem{T,inf}; checksums=true
) where {T,inf}
    (; a, sqrtdq) = op
    (; k, β) = problem
    sumdq = sum(x -> x^2, sqrtdq)  # Overflow not an issue since sumdq ≈ 1 unless invalid
    if !(
        # (length(sqrtdq) == (k + 2))  # Enforced by constructor if length(a) == k
        ((!checksums) || isapprox(sumdq, 1))
        && (inf ? (sqrtdq[begin] == 0) : (sqrtdq[begin] > 0))
        && all(sqrtdqi -> (sqrtdqi > 0), @view sqrtdq[(begin+1):end])
        && (length(a) == k)
        && ((!checksums) || (β > a[begin]))
        && all(i -> (a[i] > a[i+1]), eachindex(a)[begin:(end-1)])
        && (a[end] > 0)
    )
        throw(OrderparametersError(op, problem, sumdq))
    end
    return nothing
end

EnzymeRules.inactive(::typeof(checkorderparameters), args...) = true

struct OrderparametersError{
    T,O<:Orderparameters{T},P<:AbstractRSBProblem{T}
} <: AbstractRSBException
    op::O
    problem::P
    sumdq::T
end

function Base.showerror(io::IO, (; op, problem, sumdq)::OrderparametersError)
    (; k, β) = problem
    println(io, "Invalid order parameters for problem with k = $k, β = $β")
    println(io, "op = ", op)
    print(io, "sum(op.sqrtdq.^2) = ", sumdq)
    return nothing
end
