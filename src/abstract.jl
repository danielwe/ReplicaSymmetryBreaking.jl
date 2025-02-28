## RSB problem supertype
abstract type AbstractRSBProblem{T<:AbstractFloat,inf} end

EnzymeRules.inactive_type(::Type{<:AbstractRSBProblem}) = true

# RSB problems must implement methods for the following functions
function fieldterm end  # fieldterm(::AbstractRSBProblem{T}, ::Orderparameters{T}) where {T}
function G0 end         # G0(::AbstractRSBProblem{T}, μabs::T >= 0) >= 0: initial trace kernel
function geth end       # geth(::AbstractRSBProblem) >= 0: trace effective field

## RSB kernel weighted quadrature supertype
abstract type AbstractRSBQuadrature{T<:AbstractFloat,stretch} end
abstract type AbstractRSBQuadGK{T,stretch} <: AbstractRSBQuadrature{T,stretch} end

EnzymeRules.inactive_type(::Type{<:AbstractRSBQuadrature}) = true

# RSB quadratures must implement methods for the following functions
function integrate end  # integrate(f, quad::AbstractRSBQuadrature; quadbuffer=createbuffer(quad))
# - int f(x)
# integrate(f, quad::AbstractRSBQuadrature, values; quadbuffer=createbuffer(quad))
# - int f(x, G(values, x)) where `values` contains the result of
#   evaluating G at getnodes(quad) and then appying
#   transform!(values, quad) (see below; getnodes must have
#   one-based indexing)
# - in both cases, `quadbuffer` allows for preallocating
# intermediate storage
function createbuffer end  # createbuffer(quad::AbstractRSBQuadrature)
# - preallocate buffers used internally by `quad`

# RSB quadratures must implement methods for the following functions unless defauls suffice
cutoff(quad::AbstractRSBQuadrature) = quad.cutoff              # upper integration limit
getnodes(quad::AbstractRSBQuadrature) = quad.nodes             # interpolation nodes
lengthnodes(quad::AbstractRSBQuadrature) = length(quad.nodes)  # number of nodes
Base.@propagate_inbounds getnode(quad::AbstractRSBQuadrature, i::Integer) = quad.nodes[i]  # i-th interpolation node
Base.@propagate_inbounds getnode(quad::AbstractRSBQuadrature, I) = @view quad.nodes[I]     # slice of interpolation nodes
transform!!(x, ::AbstractRSBQuadrature) = x                    # transform f.(nodes) as needed

# RSB quadratures must include a field `atol` of type `Base.RefValue{T}`
# function set_atol!(quad::AbstractRSBQuadrature{T}, atol::T) where {T}
#     quad.atol[] = atol
#     return nothing
# end
#
# EnzymeRules.inactive(::typeof(set_atol!), args...) = true

## RSB exception supertype
abstract type AbstractRSBException <: Exception end

EnzymeRules.inactive_type(::Type{<:AbstractRSBException}) = true
