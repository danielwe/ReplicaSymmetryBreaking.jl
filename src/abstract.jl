## RSB problem supertype
abstract type AbstractRSBProblem{T<:AbstractFloat} end

# RSB problems must implement methods for the following functions
function fieldterm end  # fieldterm(::AbstractRSBProblem{T}, ::Orderparameters{T}) where {T}
function G0 end         # G0(::AbstractRSBProblem{T}, μ::T >= 0) >= 0: initial trace kernel
function heff end       # heff(::AbstractRSBProblem) >= 0: trace effective field

## RSB kernel weighted quadrature supertype
abstract type AbstractRSBQuadrature{T<:AbstractFloat} end

EnzymeRules.inactive_type(::Type{<:AbstractRSBQuadrature}) = true

# RSB quadratures must implement methods for the following functions
function cutoff end   # cutoff(::AbstractRSBQuadrature{T}) where {T}: upper integration limit
function getnode end  # getnode(quad::AbstractRSBQuadrature, i): nodes at which to eval kernel

# RSB quadratures must include a field `abstolscale` of type `Base.RefValue{T}`
function set_abstolscale!(quad::AbstractRSBQuadrature{T}, abstolscale::T) where {T}
    quad.abstolscale[] = abstolscale
    return nothing
end

EnzymeRules.inactive(::typeof(set_abstolscale!), args...) = true

## RSB exception supertype
abstract type AbstractRSBException <: Exception end

EnzymeRules.inactive_type(::Type{<:AbstractRSBException}) = true
