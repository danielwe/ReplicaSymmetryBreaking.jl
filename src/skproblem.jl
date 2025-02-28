struct SKProblem{T,inf} <: AbstractRSBProblem{T,inf}
    k::Int
    β::T
    h::T

    SKProblem(k, β::T, h::T) where {T} = new{T,β == Inf}(k, β, h)
end

SKProblem(; k, β, h=zero(β)) = SKProblem(k, β, h)

function fieldterm((; β)::SKProblem{T,inf}, (; a, sqrtdq)::Orderparameters{T}) where {T,inf}
    qci = dq0 = first(sqrtdq)^2
    sqrtdqbulk = @view sqrtdq[(begin+1):(end-1)]
    overlapterm = zero(eltype(a)) * (zero(typeof(dq0))^2)
    for i in eachindex(a, sqrtdqbulk)
        dqi = sqrtdqbulk[i]^2
        qcinext = qci + dqi
        overlapterm = muladd(-a[i], dqi * (qci + qcinext), overlapterm)
        qci = qcinext
    end
    if !inf
        overlapterm = muladd(-β, dq0^2, overlapterm)
    end
    return overlapterm / 4
end

G0(::SKProblem{T,true}, ::T) where {T} = zero(T)
G0((; β)::SKProblem{T,false}, μabs::T) where {T} = log1pexp(-2(β * μabs)) / β

geth(problem::SKProblem) = problem.h
