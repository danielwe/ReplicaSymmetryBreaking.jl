module ChebhshevClenshawPatch

### Monkey patch code that Enzyme otherwise differentiates incorrectly

__precompile__(false)  # overwriting a method, so not precompileable

import ApproxFun

function ApproxFun.ApproxFunBase.chebyshev_clenshaw(c::AbstractVector, x)
    Base.require_one_based_indexing(c)
    N,T = length(c),promote_type(eltype(c),typeof(x))
    if N == 0
        return zero(float(x) * oneunit(eltype(c)))
    elseif N == 1 # avoid issues with NaN x
        return first(c) * oneunit(float(x))
    end

    x = 2x
    bk1,bk2 = zero(T),zero(T)
    @views for cc in c[end:-1:2]
        bk2, bk1 = bk1, muladd(x,bk1,cc-bk2)
    end

    muladd(x/2,bk1,c[1]-bk2)
end

end
