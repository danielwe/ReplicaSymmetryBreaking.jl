include("../commontestsets.jl")
using AllocCheck: check_allocs
using ReplicaSymmetryBreaking: ReplicaSymmetryBreaking
using Test: @testset, @test

function noallocs(f, args)
    allocations = check_allocs(f, typeof.(args))
    isempty(allocations) && return true
    @show allocations
    return false
end

testnoallocs(f, args) = @test noallocs(f, args)

@testset "AllocCheck.jl child process" commontestsets(Dict(objective => testnoallocs))
