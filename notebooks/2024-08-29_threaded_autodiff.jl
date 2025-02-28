### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 6b1fc869-fc58-46bf-9fea-22a8f3343462
begin
	using Pkg: Pkg
	Pkg.activate(temp=true)
	Pkg.develop(path="/home/daniel/src/upstream-packages/GPUCompiler.jl")
	Pkg.add("Enzyme")
	using Enzyme
end

# ╔═╡ 34c65234-db52-4a69-bfed-d7db1a965c2f
function f!(storage)
    storage .^= 2
    return sum(storage)
end

# ╔═╡ 863cd4af-106f-4026-8ee3-2a0fe89ca584
function f_threaded!(storage)
	tasks = map(eachindex(storage)) do k
		Threads.@spawn storage[k] ^= 2
	end
	foreach(wait, tasks)
    return sum(storage)
end

# ╔═╡ 43c06605-0cd2-4560-9dd3-a5d9fc280a85
let storage=rand(3), dstorage=make_zero(storage)
	autodiff(
	    Reverse,
	    f!,
	    Active,
	    Duplicated(storage, dstorage),
	)
	dstorage
end

# ╔═╡ aababb86-f82d-4df4-b149-654bcd34295b
let storage=rand(3), dstorage=make_zero(storage)
	autodiff(
	    Reverse,
	    f_threaded!,
	    Active,
	    Duplicated(storage, dstorage),
	)
	dstorage
end

# ╔═╡ Cell order:
# ╠═6b1fc869-fc58-46bf-9fea-22a8f3343462
# ╠═34c65234-db52-4a69-bfed-d7db1a965c2f
# ╠═863cd4af-106f-4026-8ee3-2a0fe89ca584
# ╠═43c06605-0cd2-4560-9dd3-a5d9fc280a85
# ╠═aababb86-f82d-4df4-b149-654bcd34295b
