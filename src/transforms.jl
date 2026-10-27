### Common error type
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

# TODO: It's universally assumed, both in implementations here and in other places in the
# codebase, that transforms always have positive jacobian, i.e., won't change the "arrow" of
# the real line and flip the order of integration limits. This is separately guaranteed by
# each constructor; can we elevate this guarantee to some shared higher-up interface thing?
# Or is the test suite the only place we can enforce such invariants across all subtypes?

### Common private interface
apply(tf::AbstractTransform, t::Real) = (_checkt(tf, t); _apply(tf, t))
applyinverse(tf::AbstractTransform, x::Real) = (_checkx(tf, x); _applyinv(tf, x))
applyjacobian(tf::AbstractTransform, t::Real) = (_checkt(tf, t); _applyjacobian(tf, t))

### trivial
struct Identity <: AbstractTransform end

_checkt(::Identity, _) = nothing
_checkx(::Identity, _) = nothing
_apply(::Identity, t) = t
_applyinverse(::Identity, x) = x
_applyjacobian(::Identity, t) = one(t)

## [a, b) -> [0, Inf); slope at t == a equals s / (b - a)
struct IntervalToRplus{type,T<:Real} <: AbstractTransform
    a::T
    b::T
    scale::T
    function IntervalToRplus{type}(a, b, s) where {type}
        # TODO: should we replace symbols with an enum for type?
        validtypes = (:geometric, :log, :tan)
        if !(type in validtypes)
            throw(ArgumentError("type must be one of $validtypes; got type = $type")) #=lazy=#
        end
        a_, b_, s_ = promote(a, b, s)
        if !(isfinite(a_) && isfinite(b_) && (a_ < b_))
            throw(ArgumentError("require -Inf < a < b < Inf, got a = $a, b = $b")) #=lazy=#
        end
        if !(isfinite(s_) && (s_ > zero(s_)))
            throw(ArgumentError("require 0 < s < Inf, got s = $s")) #=lazy=#
        end
        # precompute appropriate scale factor to achieve slope s / (b - a) at t == a
        scale = if type === :geometric
            s_ * (b_ - a_)
        elseif type === :tan
            (2s_) / π
        else
            s_
        end
        return new{type,typeof(a_)}(a_, b_, scale)
    end
end

function _checkt((; a, b)::IntervalToRplus, t)
    (a <= t <= b) || throw(TransformError(a, t, b, :t))
    return nothing
end
function _checkx(::IntervalToRplus, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

## geometric
## -  x = s * (b - a) * (t - a) / (b - t)
## -  t = (b * x + a * s * (b - a)) / (x + s * (b - a))
## - dx = dt * s * (b - a) / (b - t)^2
## scale = s * (b - a)
_apply((; a, b, scale)::IntervalToRplus{:geometric}, t) = scale * (t - a) / (b - t)
function _applyinv((; a, b, scale)::IntervalToRplus{:geometric}, x)
    # guaranteed by constructors/checks: 0 < scale < Inf; -Inf < a < b < Inf; 0 <= x <= Inf
    # thus (isinf(num) || isinf(den)) => (t == b); no further cases/signs to consider
    num, den = (b * x + a * scale, x + scale)
    return (isinf(num) || isinf(den)) ? (oftype(num, b) / oneunit(den)) : (num / den)
end
function _applyjacobian((; a, b, scale)::IntervalToRplus{:geometric}, t)
    return scale * (b - a) / ((b - t)^2)
end

## log
## -  x = -s * log(1 + (a - t) / (b - a))
## -  t = a + (b - a) * (1 - exp(-x / s))
## - dx = dt * s / (b - t)
## scale = s
_apply((; a, b, scale)::IntervalToRplus{:log}, t) = -scale * log1p((a - t) / (b - a))
_applyinv((; a, b, scale)::IntervalToRplus{:log}, x) = a + (a - b) * expm1(-x / scale)
_applyjacobian((; b, scale)::IntervalToRplus{:log}, t) = scale / (b - t)

## tan
## -  x = (s / (π / 2)) * tan((π / 2) * (t - a) / (b - a))
## -  t = a + (b - a) * atan(x / (s / (π / 2))) / (π / 2)
## - dx = dt * s / ((b - a) * (cos((π / 2) * (t - a) / (b - a))^2))
## scale = s / (π / 2)
function _apply((; a, b, scale)::IntervalToRplus{:tan}, t)
    si, co = sincospi((t - a) / (2(b - a)))  # don't use tanpi directly; lacks AD rule
    return scale * si / co
end
function _applyinv((; a, b, scale)::IntervalToRplus{:tan}, x)
    arg = atan(x / scale) / π
    return a + 2(b - a) * arg
end
function _applyjacobian((; a, b, scale)::IntervalToRplus{:tan}, t)
    d = 2(b - a)
    return π * scale / (d * (cospi((t - a) / d)^2))
end

### [0, Inf) -> [0, Inf); slope at t == 0 equals s
struct RplusToRplus{type,T<:Real} <: AbstractTransform
    scale::T
    function RplusToRplus{type}(s) where {type}
        validtypes = (:linear, :quadratic, :exp, :sinh)
        if !(type in validtypes)
            throw(ArgumentError("type must be one of $validtypes; got type = $type")) #=lazy=#
        end
        if !(isfinite(s) && (s > zero(s)))
            throw(ArgumentError("require 0 < s < Inf, got s = $s"))
        end
        return new{type,typeof(s)}(s)
    end
end

function _checkt(::RplusToRplus, t)
    (t >= zero(t)) || throw(TransformError(zero(t), t, nothing, :t))
    return nothing
end
function _checkx(::RplusToRplus, x)
    (x >= zero(x)) || throw(TransformError(zero(x), x, nothing, :x))
    return nothing
end

## linear
## -  x = s * t
## -  t = x / s
## - dx = dt * s
_apply(tf::RplusToRplus{:linear}, t) = tf.scale * t
_applyinverse(tf::RplusToRplus{:linear}, x) = x / tf.scale
_applyjacobian(tf::RplusToRplus{:linear}, _) = tf.scale

## quadratic
## -  x = (s / 2) * ((1 + t)^2 - 1)
## -  t = sqrt(1 + x / (s / 2)) - 1
## - dx = dt * s * (1 + t)
_apply(tf::RplusToRplus{:quadratic}, t) = tf.scale * t * (1 + (t / 2))
_applyinverse(tf::RplusToRplus{:quadratic}, x) = expm1(log1p(2x / tf.scale) / 2)
_applyjacobian(tf::RplusToRplus{:quadratic}, t) = tf.scale * (1 + t)

## exp
## -  x = s * (exp(t) - 1)
## -  t = log(1 + x / s)
## - dx = dt * s * exp(t)
_apply(tf::RplusToRplus{:exp}, t) = tf.scale * expm1(t)
_applyinverse(tf::RplusToRplus{:exp}, x) = log1p(x / tf.scale)
_applyjacobian(tf::RplusToRplus{:exp}, t) = tf.scale * exp(t)

## sinh
## -  x = s * sinh(t)
## -  t = asinh(x / s)
## - dx = dt * s * cosh(t)
_apply(tf::RplusToRplus{:sinh}, t) = sinh(t)
_applyinverse(tf::RplusToRplus{:sinh}, x) = asinh(x / tf.scale)
_applyjacobian(tf::RplusToRplus{:sinh}, t) = tf.scale * cosh(t)
