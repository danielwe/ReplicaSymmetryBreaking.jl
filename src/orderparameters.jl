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
        # β < Inf => large initial step to approximate breakpoint
        ξ1 = β / (2 + β)
        range(ξ1^2, 0, k + 1)[1:(end - 1)]
    end .|> sqrt
    a = ξ ./ (1 .- ξ)
    q1 = if isinf(β)
        # β = Inf => q0 = q1 = 1
        oneunit(T)
    else
        # β < Inf => large initial step to approximate breakpoint
        oneunit(T) * (1 - inv(β)) * k / (k + 1)
    end
    q = [oneunit(T); range(q1, zero(T), k + 2)]
    sqrtdq = sqrt.(-diff(q))
    op = Orderparameters(a, sqrtdq)
    @assert validorderparameters(op, problem) "bug in Orderparameters(::AbstractRSBProblem)"
    return op
end

# Validate order parameters
function validorderparameters(
    (; a, sqrtdq)::Orderparameters{T}, (; k, β)::AbstractRSBProblem{T}
) where {T}
    return (
        (length(a) == k)
        && (β > a[begin])
        && all(i -> (a[i] > a[i + 1]), eachindex(a)[begin:(end - 1)])
        && (a[end] > 0)
        # && (length(sqrtdq) == (k + 2))  # Enforced by constructor if length(a) == k
        && ((isinf(β) && (sqrtdq[begin] == 0)) || (!isinf(β) && (sqrtdq[begin] > 0)))
        && all(sqrtdqi -> (sqrtdqi > 0), @view sqrtdq[(begin + 1):end])
        && isapprox(sum(sqrtdqi -> sqrtdqi^2, sqrtdq), 1)  # too expensive?
    )
end

function checkorderparameters(
    op::Orderparameters{T}, problem::AbstractRSBProblem{T}
) where {T}
    validorderparameters(op, problem) || throw(OrderparametersError(op, problem))
    return nothing
end

EnzymeRules.inactive(::typeof(validorderparameters), args...) = true
EnzymeRules.inactive(::typeof(checkorderparameters), args...) = true

struct OrderparametersError{T,P<:AbstractRSBProblem{T}} <: AbstractRSBException
    op::Orderparameters{T}
    problem::P
end

function Base.showerror(io::IO, (; op, problem)::OrderparametersError)
    (; k, β) = problem
    println(io, "Invalid order parameters for problem with k = $k, β = $β")
    print(io, op)
    return nothing
end
