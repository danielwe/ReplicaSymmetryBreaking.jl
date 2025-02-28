using BenchmarkTools
using Enzyme

Enzyme.API.typeWarning!(true)

function f(x, workspace)
    y = only(workspace.yt)
    cumsum!(y, x)
    return sum(y)^2 + sum(xi -> xi^2, x)
end

struct Workspace{T}
    yt::Tuple{Vector{T}}
end

x = rand(10)

workspace = Workspace((similar(x),))
workspace_nt = (; yt=(similar(x),))

@btime f($x, $workspace)
@btime f($x, $workspace_nt)

dx = similar(x)
dworkspace = make_zero(workspace)
dworkspace_nt = make_zero(workspace_nt)

@btime begin
    fill!($dx, 0)
    fill!.($dworkspace.yt, 0)
    autodiff(
        Reverse,
        f,
        Active,
        Duplicated($x, $dx),
        Duplicated($workspace, $dworkspace),
    )
    nothing
end
@btime begin
    fill!($dx, 0)
    fill!.($dworkspace.yt, 0)
    autodiff(
        Reverse,
        f,
        Active,
        Duplicated($x, $dx),
        Duplicated($workspace_nt, $dworkspace_nt),
    )
    nothing
end

function make_df(workspace)
    dworkspace = make_zero(workspace)
    return function df!(dx, x, workspace)
        fill!(dx, 0)
        fill!.(dworkspace.yt, 0)
        autodiff(Reverse, f, Active, Duplicated(x, dx), Duplicated(workspace, dworkspace))
        return nothing
    end
end

df! = make_df(workspace)
df_nt! = make_df(workspace_nt)

@btime $df!($dx, $x, $workspace)
@btime $df_nt!($dx, $x, $workspace_nt)
