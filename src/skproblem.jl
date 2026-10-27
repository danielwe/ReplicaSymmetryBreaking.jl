struct SKProblem{T,zerotemp} <: AbstractDisorderProblem{T,zerotemp}
    k::Int
    β::T
    h::T

    function SKProblem(k, β::T, h::T) where {T}
        if k < 0
            throw(ArgumentError("require k > 0, got k = $k"))
        end
        if !(β > 1)
            throw(ArgumentError("require β > 1, got k = $β"))
        end
        return new{T,β == Inf}(k, β, h)
    end
end

SKProblem(; k, β, h=zero(β)) = SKProblem(k, β, h)

# rsborder(problem::SKProblem) = problem.k  # TODO: is this necessary? shouldn't the size of parameter vectors be sufficient?
inversetemp(problem::SKProblem) = problem.β
externalfield(problem::SKProblem) = problem.h

G0(::SKProblem{T,true}, ::T) where {T} = zero(T)
function G0(problem::SKProblem{T,false}, μ::T) where {T}
    β = inversetemp(problem)
    return log1pexp(-2(β * μ)) / β
end

# TODO: adapt to generalized trace term implementation; fewer terms in the field term
function fieldterm(problem::SKProblem{T,zerotemp}, rsbp::RSBParameters{T}) where {T,zerotemp}
    a, sqrtdq = blockparams(rsbp), overlapparams(rsbp)
    dq0 = first(sqrtdq)^2  # 1 - q_1  (q_0 == 1 always)
    if zerotemp
        @assert iszero(dq0)
    else
        dq0_d = 2dq0       # 2 - 2q_1
    end
    cdqi = zero(dq0)       # q_1 - q_1
    # TODO: rewrite this to iterate rsbp directly
    sqrtdqbulk = @view sqrtdq[(begin+1):(end-1)]  # √(q_1 - q_2):√(q_κ - q_κ+1)
    overlapterm_m = zero(eltype(a)) * (zero(typeof(dq0))^2)
    for i in eachindex(a, sqrtdqbulk)
        # the first dqs _after dq0_ are the smallest, so minimize accumulated error by
        # iterating in order but keeping dq0 outside the accumulator
        dqi = sqrtdqbulk[i]^2   # q_i - q_i+1 == (1 - q_i+1) - (1 - q_i)
        cdqnext = cdqi + dqi    # q_1 - q_i+1
        qcsum = cdqi + cdqnext  # 2q_1 - q_i - q_i+1 == (q_1 - q_i) + (q_1 - q_i+1)
        if !zerotemp
            qcsum += dq0_d      # 2 - q_i - q_i+1 == (1 - q_i+1) + (1 - q_i)
        end
        overlapterm_m = muladd(a[i], dqi * qcsum, overlapterm_m)  # += a_i ((1 - q_i+1)^2 - (1 - q_i)^2)
        cdqi = cdqnext          # q_1 - q_i in next iteration
    end
    if !zerotemp
        β = inversetemp(problem)
        overlapterm_m = muladd(β, dq0^2, overlapterm_m)           # += β * (1 - q_1)^2
    end
    return -overlapterm_m / 4
end

struct SKOrderparameters{T} <: AbstractOrderparameters{T} end

orderparameters(::SKProblem{T}, _) where {T} = SKOrderparameters{T}()
norderparameters(::SKProblem) = 0
selfoverlap(::SKOrderparameters{T}) where {T} = oneunit(T)
# XXX: We currently do not support a nonzero mean coupling and magnetization
# magnetization(::SKOrderparameters{T}) where {T} = zero(T)

auxiliaryparams(::SKProblem{T}, ::SKOrderparameters{T}, _...) where {T} = nothing
function auxiliarytype(
    ::Type{T}, ::Type{<:SKProblem{T}}, ::Type{<:SKOrderparameters{T}}
) where {T}
    return Nothing
end
