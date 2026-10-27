# Musing about tractable problem types
#
# * In general: introducing a nonzero mean coupling J_0 means that we're now solving for a
#   saddle point, not a minimum/maximum, so we should use a nonlinear solver rather than an
#   optimizer. At that point, there's a question of whether we should derive and code up the
#   gradient manually after all, for performance and quadrature accuracy.
# * O(1) models can be solved at any temperature.
# * Quasi-isotropic O(n > 1) models can be solved at zero temperature.
# * Quasi-isotropic O(n > 1) models can possibly be solved at finite temperature with a little
#   extra effort, but it requires either a reformulation or some extra regularization.
#   - Except... I'm not convinced. I don't see why we can't just redefine the kernel
#     correction functions to absorb the logarithmic divergence. That will lead to a different
#     initial condition, but that doesn't seem problematic?
#   - OK, the issue is that such a kernel correction function won't vanish as μ → ∞. Instead,
#     we need to absorb a term that cancels the logarithmic divergence at μ = 0 but vanishes
#     as μ → ∞. That's what the Schmidt's regularized version was all about, but I think we
#     can do better/nicer.
#   - Yeah, the trick is to replace log(x) with log(x / tanh(x)) in that exponent. Correct
#     asymptotics, but finite x → 0 limit. Also seems to keep G_i > 0, which has been an
#     important assumption.
#   - Making a Zotero note in the pdf

### Define abstract types to be subtyped for specific RSB problems and used in their free
### energy calculation and optimization. Add stubs and docstrings for functions that
### concrete subtypes must implement.
###
### Most of these types are registered as inactive types, because they will never hold
### active values even if they are involved in active actions; for example, a quadrature
### type may hold quadrature weights that are used to calculate a differentiable result, but
### it is never relevant to differentiate with respect to the weights.

"""
    AbstractDisorderProblem{T<:AbstractFloat,zerotemp::Bool}

Supertype for types representing disordered statistical mechanics problem.

This package implements functionality for studying such problems using the replica method.

A concrete subtype `MyProblem <: AbstractDisorderProblem` must indicate whether it
represents a zero-temperature or finite-temperature system in the second type parameter as
indicated in the signature, and must implement the following methods (see their respective
docstrings for details):

* `G0(::MyProblem, μ)`
* `Gasymptote(::MyProblem, b, μ)`
* `auxparams(::MyProblem, ::MyOrderparameters)`
* `auxparams(::MyProblem{T}, (a, √Δq, b))`
* `norderparameters(::MyProblem)`
* `orderparameters(::MyProblem, ::AbstractVector)::MyOrderparameters`
* `fieldterm(::MyProblem, ::RSBSolver)`

The following methods have fallbacks that work if `MyProblem` has standard structure, but
must be specialized otherwise (see their respective docstrings for what the fallback
assumes):

* `inversetemp(::MyProblem)`
* `externalfield(::MyProblem)`

The followins methods have fallbacks that always work, but may be specialized for
performance and/or numerical precision:

* `dGasymptote(::MyProblem, (a, √Δq, b), μnext, μ)`

The following method should never need specialization if `MyProblem` implements type
parameters correctly:

* `iszerotemp(::MyProblem)`

At the moment I'm not fully clear on how general a class of problems the implementation can
handle. It's based on the approach in Schmidt, M. J. (2008), Replica Symmetry Breaking at
Low Temperatures for studying the SK model, but generalized as follows:

* Support for arbitrary single-spin trace functions, including those that cannot be
  integrated analytically
* Support for arbitrary kernel correction functions with corresponding asymptotic
  renormalization group flows

It does not (yet) support the following:

* Non-zero mean couplings
* Vector-valued trace integrals/integrands

Thus, the following classes of models should work:

* Fully connected iid. isotropic couplings between O(N) spins, including soft-spins

The following may or may not work---I don't know the field well enough to tell whether they
fit into this framework:

* Finite-dimensional models, e.g., the Edwards-Anderson model

The following might work with minimal extra effort:

* Quasi-isotropic models (uncertain point: will optimizer happily converge to saddle point?)

The following will probably never work:

* Fully anisotropic models
"""
abstract type AbstractDisorderProblem{T<:AbstractFloat,zerotemp} end

EnzymeRules.inactive_type(::Type{<:AbstractDisorderProblem}) = true

"""
    iszerotemp(problem::AbstractDisorderProblem)

Return whether `problem` is a zero-temperature problem.

This should be equivalent to `isinf(inversetemp(problem))` but stored in the type domain.
"""
iszerotemp(::AbstractDisorderProblem{T,zerotemp}) where {T,zerotemp} = zerotemp

EnzymeRules.inactive(::typeof(iszerotemp), args...) = true

"""
    inversetemp(problem::AbstractDisorderProblem{T})::T

Return the inverse temperature `β` specified by the problem `problem`.
"""
inversetemp(problem::AbstractDisorderProblem) = problem.β

EnzymeRules.inactive(::typeof(inversetemp), args...) = true

"""
    externalfield(problem::AbstractDisorderProblem{T})::T

Return the external field `h` specified by the problem `problem`.
"""
externalfield(problem::AbstractDisorderProblem) = problem.h

EnzymeRules.inactive(::typeof(externalfield), args...) = true

# TODO: should we have this? If so, should it be a function of both problem and some
# state/order parameter instance? Point being, the problem instance doesn't necessarily need
# to specify the rsb order
# """
#     rsborder(problem::AbstractDisorderProblem)::Int
#
# Return the order of replica symmetry breaking `k` specified by the problem `problem`.
# """
# function rsborder end

# XXX: We currently do not support a nonzero mean coupling and magnetization
# """
#     meancoupling(problem::AbstractDisorderProblem{T})::T
#
# Return the mean coupling `J0` specified by the problem `problem`.
# We only support quasi-isotropic models, so the mean coupling is a scalar as per
# @Schmidt2008Replica Eqs. (2.67--2.71).
# """
# meancoupling(problem::AbstractDisorderProblem) = problem.J0
#
# EnzymeRules.inactive(::typeof(meancoupling), args...) = true

"""
    G0(problem::AbstractDisorderProblem{T}, μ::T)::T

Evaluate the initial kernel correction function for the RSB problem `problem` at `μ`.

This is the starting point for the nested integrals in the trace term. Will only be
evaluated for `μ >= 0` internally, so it's OK if the implementation fails or is incorrect
for `μ < 0`.
"""
function G0 end

"""
    Gasymptote(problem::MyProblem{T}, b, μ::T)::T

Asymptotic form of the kernel correction function for `problem`, as a function of `μ`.

This takes a problem-dependent auxiliary parameter `b`, see [`auxparams`](@ref).
"""
function Gasymptote end

"""
    dGasymptote(problem::MyProblem{T}, (a, √Δq, b), μnext::T, μ::T)::T

Returns `Gasymptote(problem, bnext, μnext) - Gasymptote(problem, b, μ)`

Here, `bnext = auxparams(problem, (a, √Δq, b))`, and `a, √Δq` are the block size and overlap
parameters at the same level as `b`.

A trivial fallback implementation is provided, but a specialized method can be added for
performance and/or precision.
"""
function dGasymptote(problem::AbstractDisorderProblem, (a, sqrtdq, b), μnext, μ)
    bnext = auxparams(problem, (a, sqrtdq, b))
    return Gasymptote(problem, bnext, μnext) - Gasymptote(problem, b, μ)
end

"""
    norderparameters(problem::AbstractDisorderProblem{T})::Int

Return the number of orderparameters (excluding RSB block parameters) pertaining to `problem`.

This is used to separate regular order parameters from RSB block parameters when combined in
a flat optimization state vector.
"""
function norderparameters end

"""
    orderparameters(
        problem::AbstractDisorderProblem{T}, values::AbstractVector{T}
    )::AbstractOrderparameters

Return an instance of an appropriate subtype of `AbstractOrderparameters` for `problem`.

The values in the returned object are set by the vector `values`. The values are typically
part of a larger optimization state vector that also includes RSB block parameters; this
function should derive the values in the returned instance from the first
`norderparameters(problem)` elements of `values` and not touch the remainder of the vector.
"""
function orderparameters end

# XXX: this is incorrect! `solver` now contains `problem`. how do we specialize?
"""
    fieldterm(problem::AbstractDisorderProblem{T}, solver::RSBSolver{T})::T

Calculate the RSB free energy field term for the problem `problem` with RSB solover `solver`.

`solver` is an instance of `RSBSolver`, which is a struct that contains both the generic
RSB block parameters and problem-specific non-RSB orderparameters contained in an instance
of the appropriate `AbstractOrderparameters` subtype.

The field term consists of everything except the trace term, that is, the term comprising
`k` nested integrals.
"""
function fieldterm end

### Order parameter supertype
### Subtypes of this hold order parameters that are problem-specific but not RSB-specific,
### such as magnetization, self-overlap, et cetera
"""
    AbstractOrderParameters{T<:AbstractFloat}

Supertype for types representing non-replica order parameters specific to problem types.

A concrete subtype `MyOrderparameters <: AbstractOrderparameters` must implement the
following methods (see their respective docstrings for details):

* `selfoverlap(::MyOrderparameters)`
* `auxparams(::MyProblem, ::MyOrderparameters)`
"""
abstract type AbstractOrderparameters{T<:AbstractFloat} end

"""
    selfoverlap(op::AbstractOrderparameters{T})::T

Get the self-overlap (overlap matrix diagonal) from the order parameter object `op`.
"""
function selfoverlap end

# XXX: We currently do not support a nonzero mean coupling and magnetization
# """
#     magnetization(op::AbstractOrderparameters{T})::T
#
# Get the magnetization from the order parameter object `op`.
# """
# function magnetization end

"""
    b_0 = auxparams(problem::AbstractDisorderProblem, orderparameters::AbstractOrderparameters)
    bnext = auxparams(problem::AbstractDisorderProblem, (a_i, √Δq_i, b_i))

Calculate auxiliary RSB block parameters needed in the field and/or trace term.

This works similarly to an iterator, except it takes the generic block parameters as extra
state to compute the next order of auxiliary parameters from the current.

This function is used internally to make it so that iterating an `RSBSolver` instance
returns successive values of `(a_i, √Δq_i, b_i)`. Here, we labeled the auxiliary parameters
`b_i`, where the index `i` matches that of the block parameters `a_i` and overlap deltas
`Δq_i`.
"""
function auxparams end

### R -> R function transform supertype
abstract type AbstractTransform end

EnzymeRules.inactive_type(::Type{<:AbstractTransform}) = true

# TODO: we don't need these, do we?
# (tf::AbstractTransform)(t) = apply(tf, t)
# Base.inv(tf::AbstractTransform) = (x -> applyinverse(tf, x))
# jacobian(tf::AbstractTransform) = (t -> applyjacobian(tf, t))

"""
    apply(tf::AbstractTransform, t::T)::T

Apply the transform `tf` to the value `t`.
"""
function apply end

"""
    applyinverse(tf::AbstractTransform, x::T)::T

Apply the inverse of the transform `tf` to the value `x`, returning the value `t` such that
`apply(tf, t) ≈ x`.
"""
function applyinverse end

"""
    applyjacobian(tf::AbstractTransform, t::T)::T

Apply the jacobian of the transform `tf` to the value `t`, returning the derivative of
`apply(tf, t)` with respect to `t`.
"""
function applyjacobian end

### Quadrature supertype
abstract type AbstractQuadrature{T<:AbstractFloat} end

EnzymeRules.inactive_type(::Type{<:AbstractQuadrature}) = true

"""
    integrate(f, quad::AbstractQuadrature{T}, breakpoints=(); scale::T=zero(T))::T

Integrate the function `f` over an interval on the real line defined by `quad`.

# Arguments

* `f`: Function to integrate.
* `quad`: Instance of `AbstractQuadrature` defining the domain of integration and the
  quadrature method, including parameters such as quadrature order and relative tolerance.
* `breakpoints` (optional): Ordered tuple of points subdividing the interval for improved
  convergence. For example, if `f` has a singularity, `breakpoints` should contain the
  singular point; if `f` has a narrow feature that must be resolved, for example a narrow
  gaussian envelope, consider including two points closely framing the feature, for example
  `(μ - σ, μ + σ)` where `μ` and `σ` are the mean and standard deviation of the gaussian
  envelope. It is assumed that `breakpoints` is sorted and only contains points within the
  domain of integration defined by `quad`.
* `scale` (optional keyword): Ratio between the relative and absolute tolerance, that is,
  `atol = scale * rtol`. A good choice for this is usually the maximum absolute value of `f`
  on the domain of integration, or a number of similar magnitude.
"""
function integrate end

"""
    limits(quad::AbstractQuadrature{T})::Tuple{T,T}

Return the lower and upper limits of `quad`'s domain of integration.
"""
function limits end

EnzymeRules.inactive(::typeof(limits), args...) = true

"""
    gettransform(quad::AbstractQuadrature)::AbstractTransform


Return a transform for turning `integrate(f, quad)` into an integral over `[0, Inf)`.

To integrate `g` from `0` to `Inf`, do
`integrate(t -> g(apply(tf, t)) * jacobian(tf, f), quad)`, where `tf = gettransform(quad)`.
"""
function gettransform end

EnzymeRules.inactive(::typeof(gettransform), args...) = true

"""
    integratewithapproximant(F, gapprox, quad::AbstractQuadrature{T}, args...; kwargs...)::T

Integrate the function `t -> F(t, g(t))` over an interval on the real line defined by `quad`.

The argument `gapprox` should be an approximant as returned by
`approximate((xs, _) -> g.(xs), quad)`. This function is useful when several integrals
with the same expensive `g` but different `F` are to be computed.

`gapprox` can be any type and in particular does not have to be callable; it is up to the
respective methods of `approximate` and `integratewithapproximant` for the given
`typeof(quad)` to implement a consistent approach to sampling and approximating `g`.

Optional arguments and keyword arguments are the same as for `integrate`.
"""
function integratewithapproximant end

abstract type AbstractOrder end
struct Ascending <: AbstractOrder end
struct Descending <: AbstractOrder end
struct Unordered <: AbstractOrder end

EnzymeRules.inactive_type(::Type{<:AbstractOrder}) = true

"""
    approximate((xs, order) -> g.(xs), quad::AbstractQuadrature)

Sample `g` and return an approximant that can be used in `integratewithapproximant`.

`g` is sampled and approximated on the domain of integration specified by `quad`.

The returned object can be any type and in particular does not have to be callable; it is up
to the respective methods of `approximate` and `integratewithapproximant` for the given
`typeof(quad)` to implement a consistent approach to sampling and approximating `g`. The
approximation should have enough accuracy that the difference between `integrate(g, quad)`
and `integrate(gapprox, quad)` is within the quadrature noise, where
`gapprox = approximate((x, _) -> g.(xs))`.

As indicated by the signature, `approximate` should take a vectorized wrapper of `g` that
should also accept a second argument `order`; `order` should be an `AbstractOrder` instance
indicating how the sample points `xs` are ordered. This vectorized and order-aware interface
allows the caller to exploit structure in `g` that allows for more efficient evaluation if
points are ordered. For example, if `g` asymptotically converges to zero, vectorized
evaluation can start at the nonzero end and only proceed until the value is zero to machine
precision, setting all remaining samples to zero.
"""
function approximate end

# TODO: there must be a better way than this
"""
    approximantiscallable(quad::AbstractQuadrature)::Bool

Indicate whether `integratewithapproximant(F, gapprox, quad, ...)` takes a callable `gapprox`.

This also implies that approximants returned by `approximate((xs, _) -> g.(xs), quad)` are
callable.

The utility of this trait is that the caller can add a wrapper around the approximant before
use if it so chooses, for example to filter out under-/overshoot or enforce known
invariants. Examples may be `gapprox_wrapped(t) = (gt = gapprox(t); max(zero(gt), gt))` to
enforce nonnegativity, or `gapprox_wrapped(t) = (t > tZ) ? zero(t) : gapprox(t)` to clamp to
zero beyond some threshold.
"""
function approximantiscallable end

### Callable interpolant supertype
abstract type AbstractInterpolator{T<:AbstractFloat} end

# TODO: how to add these docstrings without overwriting?
# """
#     limits(app::AbstractInterpolator{T})::Tuple{T,T}
#
# Return the lower and upper limits of `app`'s domain of interpolation.
# """
# function limits end
#
# """
#     approximate((xs, order) -> g.(xs), app::AbstractInterpolator)
#
# Sample `g` and return an interpolant on the interval defined by `limits(app)`.
#
# Unlike the similar `AbstractQuadrature` methods, the approximant returned by this method
# should be callable. See the docstring for those methods for further details about the
# interface and the meaning of the `order` argument.
# """
# function approximate end

### RSB exception supertype
abstract type AbstractRSBException <: Exception end

EnzymeRules.inactive_type(::Type{<:AbstractRSBException}) = true
