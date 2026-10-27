### RSB solver
struct RSBSolver{T<:AbstractFloat,P<:AbstractDisorderProblem{T},O<:AbstractOrderparameters{T}}
    problem::P
    orderparameters::O  # problem-specific but not RSB-specific order parameters (magnetization, self-overlap, et cetera)
    a::Vector{T}        # β-scaled block size parameters
    sqrtdq::Vector{T}   # overlap parameter steps

    function RSBSolver(
        problem::AbstractDisorderProblem,
        orderparameters::AbstractOrderparameters,
        a::AbstractVector{T},
        sqrtdq::AbstractVector{T},
    ) where {T}
        # TODO: add more consistency checks, like q[i] <= q[0]
        # TODO: should we check consistency between typeof(problem) and typeof(orderparameters)
        # there would have to be a function mapping problem type to orderparameter type
        if !(length(sqrtdq) == (length(a) + 2))
            throw(ArgumentError("must have length(sqrtdq) == (length(a) + 2)"))
        end
        return new{T}(problem, orderparameters, a, sqrtdq)
    end
end

"""
    problem(solver::RSBSolver)::AbstractRSBProblem

Return the problem pertaining to `solver`.
"""
problem(solver::RSBSolver) = solver.problem

"""
    orderparameters(solver::RSBSolver)::AbstractOrderparameters

Return the problem-specific orderparameters in the current state of the `solver`.
"""
orderparameters(solver::RSBSolver) = solver.orderparameters

"""
    blockparams(solver::RSBSolver{T})::Vector{T}

Return the overlap matrix block size parameters `a[i]`.

These are the temperature-scaled block size parameters introduced in @Schmidt2008Replica and
differ from the conventional Parisi block size parameters `m[i]` by a factor of the inverse
temperature `β`, that is, `a[i] = β * m[i]`. This removes the singularity at `β = 0`, where
`m[i] == 0` for every `i`, while the `a[i]` are finite.

The parameters are positive and ordered from large to small.
"""
blockparams(solver::RSBSolver) = solver.a

"""
    overlapparams(solver::RSBSolver{T})::Vector{T}

Return the overlap value parameters `sqrtdq[i] == √(q[i - 1] - q[i])`.

`RSBSolver` stores the square root of the difference of successive overlaps, rather than the
overlaps themselves, since this is the most convenient representation when calculating the
free energy. The overlaps themselves can be recovered by recalling that
`q[0] == selfoverlap(orderparameters(solver))` and `q[end] == 0`.

We only support quasi-isotropic models, so there is only a single family of overlaps as per
@Schmidt2008Replica Eqs. (2.67--2.71).

The parameters are ordered in correspondence to the block size parameters
`a = blockparams(op)`; that is, `q[i]` is the value of the discretized overlap function
`q(a)` for `a[i] <= a < a[i - 1]`, or equivalently, the function takes a step of magnitude
`sqrtdq[i]^2` at `a = a[i - 1]`.
"""
overlapparams(solver::RSBSolver) = solver.sqrtdq

function blockiterator(solver::RSBSolver)
    return zip(blockparams(solver), @view overlapparams(solver)[(begin+1):(end-1)])
end
Base.length(solver::RSBSolver) = length(blockiterator(solver)) + !iszerotemp(problem(solver))
function Base.eltype(::Type{<:RSBSolver{T,P,O}}) where {T,P,O}
    return Tuple{T,T,Base.promote_op(auxiliaryparams, P, O)}
end

@inline function Base.iterate(solver::RSBSolver)
    adqiter = blockiterator(solver)
    b0 = auxparams(problem(solver), orderparameters(solver))
    adqinit_state = if iszerotemp(problem(solver))
        # Here we assume that b1 -> b0 as β -> ∞, so we don't need to iterate the auxiliary
        # parameters. This is justified under the assumptions we're working within: in this
        # limit, the first integral collapses to a trivial identity operator, so ℰ(b1, μ)
        # and ℰ(b0, μ) should be the same function of μ, hence b1 == b0.
        iterate(adqiter)
    else
        ((inversetemp(problem(solver)), first(overlapparams(solver))),)
    end
    return iterate(solver, (adqiter, adqinit_state, b0))
end

@inline function Base.iterate(solver::RSBSolver, (adqiter, adqi_state, bi))
    isnothing(adqi_state) && return nothing
    adqi, state... = adqi_state
    value = (adqi..., bi)
    bnext = auxparams(problem(solver), value)
    return value, (adqiter, iterate(adqiter, state...), bnext)
end

# simple initial guess
function initorderparameters(problem::AbstractDisorderProblem{T,inf}) where {T,inf}
    k = rsborder(problem)
    a = range(oneunit(T), 0, k + 1)[begin:(end-1)]
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

# validation
struct OrderparametersError{
    T,O<:Orderparameters{T},P<:AbstractDisorderProblem{T}
} <: AbstractRSBException
    op::O
    problem::P
end

function Base.showerror(io::IO, (; op, problem)::OrderparametersError)
    k = rsborder(problem)
    β = inversetemp(problem)
    sumdq = sum(abs2, overlapparams(op))
    println(io, "Invalid order parameters for problem with k = $k, β = $β")
    println(io, "op = ", op)
    print(io, "sum(dq) = ", sumdq)
    return nothing
end

function checkorderparameters(
    op::Orderparameters{T}, problem::AbstractDisorderProblem{T,inf}; checksums=true
) where {T,inf}
    a = blockparams(op)
    sqrtdq = overlapparams(op)
    k = rsborder(problem)
    β = inversetemp(problem)
    ispos(x) = (x > zero(x))
    if !(
        (inf ? iszero(sqrtdq[begin]) : ispos(sqrtdq[begin]))
        && all(ispos, @view sqrtdq[(begin+1):end])
        && (length(a) == k)  # length(sqrtdq) == length(a) + 2 enforced by constructor
        && all(i -> (a[i] > a[i+1]), eachindex(a)[begin:(end-1)])
        && ispos(a[end])
        && ((!checksums) || (isapprox(sum(abs2, sqrtdq), selfoverlap(op)) && (β > a[begin])))
    )
        throw(OrderparametersError(op, problem))
    end
    return nothing
end

EnzymeRules.inactive(::typeof(checkorderparameters), args...) = true
