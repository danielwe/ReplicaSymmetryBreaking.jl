module Patches

### Implement other packages' missing pieces

## Import rrule for AbstractFFTs.Plan into Enzyme
using AbstractFFTs: AbstractFFTs
using Enzyme: Enzyme, EnzymeRules, Const, Duplicated, make_zero, make_zero!

Enzyme.EnzymeRules.inactive_type(::Type{<:AbstractFFTs.Plan}) = true

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(*)},
    ::Type,
    P::Const{<:AbstractFFTs.Plan{T}},
    x::Duplicated{<:StridedArray{T}},
) where {T}
    # we can never skip the forward pass because we don't know a priori whether P is an
    # in-place plan, and in-place mutation for non-NoNeed arguments must be performed
    # regardless of needs_primal
    xval = x.val
    yval = P.val * xval
    inplace = Base.mightalias(yval, xval)
    if inplace
        @assert yval === xval  # otherwise I don't know what to do
    end
    needs_primal = EnzymeRules.needs_primal(config)
    primal = needs_primal ? yval : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        if needs_primal || inplace
            make_zero(yval)
        else  # might as well reuse yval as shadow then
            make_zero!(yval)
            yval
        end
    else
        nothing
    end
    tape = (inplace, shadow)  # since * is linear, we don't care whether x is overwritten
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(*)},
    ::Type,
    tape,
    P::Const{<:AbstractFFTs.Plan{T}},
    x::Duplicated{<:StridedArray{T}},
) where {T}
    inplace, shadow = tape
    Padj = adjoint(P.val)
    dx = x.dval
    if inplace
        out = Padj * dx
        @assert out === dx  # sanity check that adjoint(P) is in-place when P is
    end
    if !isnothing(shadow)
        dx .+= Padj * shadow  # mul! not yet supported for adjoint plans
        make_zero!(shadow)
    end
    return (nothing, nothing)
end

## Implement AdjointStyle and adjoint_mul for ChebyshevTransformPlan
## This makes ChebyshevTransformPlan differentiable via ChainRules
using FastTransforms: FastTransforms
using FFTW: FFTW

struct ChebyshevAdjointStyle <: AbstractFFTs.AdjointStyle end

AbstractFFTs.AdjointStyle(::FastTransforms.ChebyshevTransformPlan) = ChebyshevAdjointStyle()

function AbstractFFTs.adjoint_mul(
    P::FastTransforms.ChebyshevTransformPlan{T,kind,K,inplace}, x::AbstractArray{T}, ::ChebyshevAdjointStyle
) where {T,kind,K,inplace}
    # chebX_rescale! are symmetric operators, so only need to move from after transform to before
    y = inplace ? x : copy(x)
    if kind == 1
        FastTransforms._cheb1_rescale!(P.plan.region, y)
    elseif kind == 2
        FastTransforms._cheb2_rescale!(P.plan.region, y)
    else
        throw(ArgumentError("invalid kind $kind"))
    end
    return adjoint(P.plan) * y
end

struct R2RFFTAdjointStyle <: AbstractFFTs.AdjointStyle end

AbstractFFTs.AdjointStyle(::FFTW.r2rFFTWPlan) = R2RFFTAdjointStyle()

function AbstractFFTs.adjoint_mul(
    p::FFTW.r2rFFTWPlan{T}, x::AbstractVector{T}, ::R2RFFTAdjointStyle
) where {T}
    (length(p.kinds) == 1) || throw(ArgumentError("Multidimensional r2r transforms not yet supported"))
    (only(p.kinds) == 5) || throw(ArgumentError("r2r kinds other than REDFT10 not yet supported"))
    dc = first(x)  # save this first in case p is in-place
    unscaled_pinv = inv(p).p
    y = unscaled_pinv * x
    # REDFT10 is unitary except for the first row, so the unscaled inverse is the adjoint
    # except for the first column. To obtain the true adjoint, add more DC.
    y .+= dc
    return y
end

end  # module
