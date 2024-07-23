include("../commontestsets.jl")
using AllocCheck: check_allocs
using Test: @testset, @test

function alloccheck(f, args)
    allocations = check_allocs(f, typeof.(args))
    isempty(allocations) && return true
    for allocation in allocations
        @show allocation
    end
    return false
end

testalloccheck(f, args) = @test alloccheck(f, args)
@testset "AllocCheck.jl child process" commontestsets(Dict(rsbobjective => testalloccheck))
