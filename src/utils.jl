## Custom loggers with global mechanism for enabling/disabling, useful for AllocCheck tests
loggingenabled() = true

macro maybewarn(exprs...)
    escaped = []
    for ex in exprs
        if (ex isa Expr) && (ex.head == :(=))
            lhs, rhs = ex.args
            push!(escaped, Expr(:(=), lhs, esc(rhs)))
        else
            push!(escaped, esc(ex))
        end
    end
    return :(loggingenabled() && @warn $(escaped...))
end

# Combine error messages from captured/composite exceptions
# Not currently used in the package, but useful downstream, e.g., in callbacks
errormessage(e::Exception) = sprint(showerror, e)
errormessage(e::CapturedException) = errormessage(e.ex)
errormessage(e::CompositeException) = join(map(errormessage, e.exceptions), "\n")

function errormessage(e::TaskFailedException)
    return join(map(e_ -> errormessage(e_.exception), current_exceptions(e.task)), "\n")
end

EnzymeRules.inactive(::typeof(errormessage), args...) = true
