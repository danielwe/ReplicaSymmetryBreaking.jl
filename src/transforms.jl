abstract type AbstractTransform end

(transform::AbstractTransform)(t) = apply(transform, t)
getinverse(transform::AbstractTransform) = (x -> applyinverse(transform, x))
getjacobian(transform::AbstractTransform) = (t -> jacobian(transform, t))

struct Identity <: AbstractTransform end

@inline apply(::Identity, t) = t
@inline applyinverse(::Identity, x) = x
@inline jacobian(::Identity, t) = one(t)

## transform t ∈ [0, Inf) to x ∈ [0, Inf)
struct Expm1{T} <: AbstractTransform
    scale::T
    function Expm1(scale)
        if !(isfinite(scale) && (scale > zero(scale)))
            throw(ArgumentError("require 0 < scale < Inf, got $scale"))
        end
        return new{typeof(scale)}(scale)
    end
end

@inline function _check_t(::Expm1, t)
    (t >= zero(t)) || throw(TransformError(zero(t), t, nothing, :t))
    return nothing
end
@inline function _check_x(::Expm1, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

@inline apply(tf::Expm1, t) = (_check_t(tf, t); expm1(tf.scale * t))
@inline applyinverse(tf::Expm1, x) = (_check_x(tf, x); log1p(x) / tf.scale)
@inline jacobian(tf::Expm1, t) = (_check_t(tf, t); tf.scale * exp(tf.scale * t))

struct DoubleExpm1{T} <: AbstractTransform
    scale::T
    function DoubleExpm1(scale)
        if !(isfinite(scale) && (scale > zero(scale)))
            throw(ArgumentError("require 0 < scale < Inf, got $scale"))
        end
        return new{typeof(scale)}(scale)
    end
end

@inline function _check_t(::DoubleExpm1, t)
    (t >= zero(t)) || throw(TransformError(zero(t), t, nothing, :t))
    return nothing
end
@inline function _check_x(::DoubleExpm1, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

@inline apply(tf::DoubleExpm1, t) = (_check_t(tf, t); expm1(expm1(tf.scale * t)))
@inline applyinverse(tf::DoubleExpm1, x) = (_check_x(tf, x); log1p(log1p(x)) / tf.scale)
@inline jacobian(tf::DoubleExpm1, t) = (_check_t(tf, t); st = tf.scale * t; tf.scale * exp(st + expm1(st)))

struct Sinh{T} <: AbstractTransform
    scale::T
    function Sinh(scale)
        if !(isfinite(scale) && (scale > zero(scale)))
            throw(ArgumentError("require 0 < scale < Inf, got $scale"))
        end
        return new{typeof(scale)}(scale)
    end
end

@inline function _check_t(::Sinh, t)
    (t >= zero(t)) || throw(TransformError(zero(t), t, nothing, :t))
    return nothing
end
@inline function _check_x(::Sinh, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

@inline apply(tf::Sinh, t) = (_check_t(tf, t); sinh(tf.scale * t))
@inline applyinverse(tf::Sinh, x) = (_check_x(tf, x); asinh(x) / tf.scale)
@inline jacobian(tf::Sinh, t) = (_check_t(tf, t); tf.scale * cosh(tf.scale * t))

struct DoubleSinh{T} <: AbstractTransform
    scale::T
    function DoubleSinh(scale)
        if !(isfinite(scale) && (scale > zero(scale)))
            throw(ArgumentError("require 0 < scale < Inf, got $scale"))
        end
        return new{typeof(scale)}(scale)
    end
end

@inline function _check_t(::DoubleSinh, t)
    (t >= zero(t)) || throw(TransformError(zero(t), t, nothing, :t))
    return nothing
end
@inline function _check_x(::DoubleSinh, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

@inline apply(tf::DoubleSinh, t) = (_check_t(tf, t); sinh(sinh(tf.scale * t)))
@inline applyinverse(tf::DoubleSinh, x) = (_check_x(tf, x); asinh(asinh(x)) / tf.scale)
@inline jacobian(tf::DoubleSinh, t) = (_check_t(tf, t); st = tf.scale * t; tf.scale * cosh(st) * cosh(sinh(st)))

struct Square{T} <: AbstractTransform
    scale::T
    function Square(scale)
        if !(isfinite(scale) && (scale > zero(scale)))
            throw(ArgumentError("require 0 < scale < Inf, got $scale"))
        end
        return new{typeof(scale)}(scale)
    end
end

@inline function _check_t(::Square, t)
    (t >= zero(t)) || throw(TransformError(zero(t), t, nothing, :t))
    return nothing
end
@inline function _check_x(::Square, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

@inline apply(tf::Square, t) = (_check_t(tf, t); t + tf.scale * (t^2))
@inline applyinverse(tf::Square, x) = (_check_x(tf, x); (sqrt(1 + 4tf.scale * x) - 1) / (2tf.scale))
@inline jacobian(tf::Square, t) = (_check_t(tf, t); 1 + 2tf.scale * t)

## transform t ∈ (a, b) to x ∈ (0, Inf)
struct Halfstretch{stretch,T<:Real} <: AbstractTransform
    a::T
    b::T
    scale::T
    function Halfstretch{stretch}(a, b, scale) where {stretch}
        if !(stretch in (:power, :exp, :tan, :tanpi))
            throw(ArgumentError("stretch must be :exp, :power, :tan, or :tanpi, got $stretch")) #=lazy=#
        end
        a_, b_, scale_ = promote(a, b, scale)
        if !(isfinite(a_) && isfinite(b_) && (a_ < b_))
            throw(ArgumentError("require -Inf < a < b < Inf, got a = $a, b = $b")) #=lazy=#
        end
        if !(isfinite(scale_) && (scale_ > zero(scale_)))
            throw(ArgumentError("require 0 < scale < Inf, got $scale")) #=lazy=#
        end
        return new{stretch,typeof(a_)}(a_, b_, scale_)
    end
end

@inline function _check_t((; a, b)::Halfstretch, t)
    (a <= t <= b) || throw(TransformError(a, t, b, :t))
    return nothing
end
@inline function _check_x(::Halfstretch, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

@inline apply(tf::Halfstretch, t) = (_check_t(tf, t); _apply(tf, t))
@inline applyinverse(tf::Halfstretch, x) = (_check_x(tf, x); _applyinv(tf, x))
@inline jacobian(tf::Halfstretch, t) = (_check_t(tf, t); _jacobian(tf, t))

# power: x = s * (t - a) / (b - t); t = (b * x + a * s) / (x + s); dx = dt * s / (b - t)^2
@inline _apply((; a, b, scale)::Halfstretch{:power}, t) = scale * (t - a) / (b - t)
@inline function _applyinv((; a, b, scale)::Halfstretch{:power}, x)
    # guaranteed by constructors/checks: 0 < scale < Inf; -Inf < a < b < Inf; 0 <= x <= Inf
    # thus (isinf(num) || isinf(den)) => (t == b); no further cases/signs to consider
    num, den = (b * x + a * scale, x + scale)
    return (isinf(num) || isinf(den)) ? (oftype(num, b) / oneunit(den)) : (num / den)
end
@inline _jacobian((; a, b, scale)::Halfstretch{:power}, t) = scale * (b - a) / ((b - t)^2)

# exp: x = -s * log(1 + (a - t) / (b - a)); t = a + (b - a) * (1 - exp(-x / s)); dx = dt * s / (b - t)
@inline _apply((; a, b, scale)::Halfstretch{:exp}, t) = -scale * log1p((a - t) / (b - a))
@inline _applyinv((; a, b, scale)::Halfstretch{:exp}, x) = a + (a - b) * expm1(-x / scale)
@inline _jacobian((; b, scale)::Halfstretch{:exp}, t) = scale / (b - t)

# tanpi: x = s * tanpi((t - a) / 2(b - a)); t = a + 2(b - a) * atan(x / s) / π; dx = dt * s * π / (2(b - a) * (cospi((t - a) / 2(b - a))^2))
@inline function _apply((; a, b, scale)::Halfstretch{:tanpi}, t)
    s, c = sincospi((t - a) / (2(b - a)))  # don't use tanpi directly; lacks AD rule
    return scale * s / c
end
@inline function _applyinv((; a, b, scale)::Halfstretch{:tanpi}, x)
    arg = atan(x / scale) / π
    return a + 2(b - a) * arg
end
@inline function _jacobian((; a, b, scale)::Halfstretch{:tanpi}, t)
    d = 2(b - a)
    return π * scale / (d * (cospi((t - a) / d)^2))
end

struct TransformError{L,V,U} <: AbstractRSBException
    lower::L
    value::V
    upper::U
    symbol::Symbol
end

function Base.showerror(io::IO, (; lower, value, upper, symbol)::TransformError)
    if !isnothing(lower) && !isnothing(upper)
        print(io, "require $lower <= $symbol <= $upper, ")
    elseif !isnothing(lower)
        print(io, "require $symbol >= $lower, ")
    elseif !isnothing(upper)
        print(io, "require $symbol <= $upper, ")
    end
    print(io, "got $symbol = $value")
    return nothing
end
