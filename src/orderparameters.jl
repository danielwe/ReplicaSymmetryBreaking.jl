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

function _make_zero!(op::Orderparameters)
    _make_zero!(op.a)
    _make_zero!(op.sqrtdq)
    return nothing
end

# Heuristic initial guess for problem
function Orderparameters(problem::AbstractRSBProblem{T}) where {T}
    (; k, β) = problem
    ξ = if isinf(β)
        range(one(β), 0, k + 2)[2:(end - 1)]
    else
        range(β / (4 + β), 0, k + 1)[1:(end - 1)]
    end
    a = 2 .* ξ ./ (1 .- ξ)
    q = if isinf(β)
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
    op::Orderparameters{T}, problem::AbstractRSBProblem{T}
) where {T}
    (; a, sqrtdq) = op
    (; k, β) = problem
    sumdq = sum(x -> x^2, sqrtdq)  # Overflow not an issue since sumdq ≈ 1 unless invalid
    if !(
        # (length(sqrtdq) == (k + 2))  # Enforced by constructor if length(a) == k
        isapprox(sumdq, 1)
        && all(sqrtdqi -> (sqrtdqi > 0), @view sqrtdq[(begin + 1):end])
        && ((isinf(β) && (sqrtdq[begin] == 0)) || (!isinf(β) && (sqrtdq[begin] > 0)))
        && (length(a) == k)
        && (β > a[begin])
        && all(i -> (a[i] > a[i + 1]), eachindex(a)[begin:(end - 1)])
        && (a[end] > 0)
    )
        throw(OrderparametersError(op, problem, sumdq))
    end
    return nothing
end

EnzymeRules.inactive(::typeof(checkorderparameters), args...) = true

struct OrderparametersError{T,P<:AbstractRSBProblem{T}} <: AbstractRSBException
    op::Orderparameters{T}
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
