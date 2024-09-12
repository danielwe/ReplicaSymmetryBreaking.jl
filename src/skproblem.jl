struct SKProblem{T} <: AbstractRSBProblem{T}
    k::Int
    β::T
    h::T

    SKProblem(k, β::T, h::T) where {T} = new{T}(k, β, h)
end

SKProblem(; k, β, h=zero(β)) = SKProblem(k, β, h)

function fieldterm((; β)::SKProblem{T}, (; a, sqrtdq)::Orderparameters{T}) where {T}
    qci = dq0 = first(sqrtdq)^2
    overlapterm = zero(eltype(a)) * (zero(dq0)^2)
    sqrtdqbulk = @view sqrtdq[(begin + 1):(end - 1)]
    for i in eachindex(a, sqrtdqbulk)
        dq = sqrtdqbulk[i]^2
        qcinext = qci + dq
        overlapterm -= a[i] * dq * (qci + qcinext)
        qci = qcinext
    end
    isinf(β) && return overlapterm / 4
    return muladd(-β, dq0^2, overlapterm) / 4
end

G0((; β)::SKProblem{T}, μ::T) where {T} = log1pexp(-2β * μ) / β

heff(problem::SKProblem) = abs(problem.h)
