## RSB problem supertype
abstract type AbstractRSBProblem{T<:AbstractFloat} end

# RSB problems must implement methods for the following functions
function fieldterm end  # fieldterm(::AbstractRSBProblem{T}, ::Orderparameters{T}) where {T}
function G0 end         # G0(::AbstractRSBProblem{T}, μ::T >= 0) >= 0: initial trace kernel
function heff end       # heff(::AbstractRSBProblem) >= 0: trace effective field
