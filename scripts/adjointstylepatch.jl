module AdjointStylePatch

### Implement missing AdjointStyle and adjoint_mul implementations

import AbstractFFTs
import FastTransforms
import FFTW

struct ChebyshevAdjointStyle <: AbstractFFTs.AdjointStyle end

AbstractFFTs.AdjointStyle(::FastTransforms.ChebyshevTransformPlan) = ChebyshevAdjointStyle()

function AbstractFFTs.adjoint_mul(
    P::FastTransforms.ChebyshevTransformPlan{T,kind}, x::AbstractArray{T}, ::ChebyshevAdjointStyle
) where {T,kind}
    # chebX_rescale! are symmetric operators, so just need to move from last to first
    y = if kind == 1
        FastTransforms._cheb1_rescale!(P.plan.region, copy(x))
    elseif kind == 2
        FastTransforms._cheb2_rescale!(P.plan.region, copy(x))
    else
        throw(ArgumentError("invalid kind $kind"))
    end
    return (P.plan)' * y
end

struct R2RFFTAdjointStyle <: AbstractFFTs.AdjointStyle end

AbstractFFTs.AdjointStyle(::FFTW.r2rFFTWPlan) = R2RFFTAdjointStyle()

function AbstractFFTs.adjoint_mul(
    p::FFTW.r2rFFTWPlan{T}, x::AbstractVector{T}, ::R2RFFTAdjointStyle
) where {T}
    (length(p.kinds) == 1) || throw(ArgumentError("Multidimensional r2r transforms not yet supported"))
    (only(p.kinds) == 5) || throw(ArgumentError("r2r kinds other than REDFT10 not yet supported"))
    unscaled_pinv = inv(p).p
    y = unscaled_pinv * x
    # REDFT10 is unitary except for the first row, so the unscaled inverse is the adjoint
    # except for the first column. To obtain the true adjoint, add more DC.
    y .+= first(x)
    return y
end

end  # module
