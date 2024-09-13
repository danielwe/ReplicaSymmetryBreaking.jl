# Unpack and handle captured/composite exceptions
unpackexceptions(@nospecialize(e::Exception)) = Exception[e]

function unpackexceptions(e::CompositeException)
    return mapreduce(unpackexceptions, vcat, e)
end

function unpackexceptions(e::TaskFailedException)
    _unpackexceptions(stackex) = unpackexceptions(stackex.exception)
    return mapreduce(_unpackexceptions, vcat, current_exceptions(e.task))
end

function errormessage(e::Exception, args...)
    return errormessage(unpackexceptions(e), args...)
end

function errormessage(stack, delim="\n")
    return join(Iterators.map(ex -> sprint(showerror, ex), stack), delim)
end

function handleexceptions(e::Exception, ::Type{KnownEx}; warn=true) where {KnownEx}
    e_unpacked = unpackexceptions(e)
    for ex in e_unpacked
        isa(ex, KnownEx) || return false
    end
    if warn
        msg = errormessage(e_unpacked, "\n- ")
        @warn "Handling the following exceptions:\n- " * msg
    end
    return true
end

EnzymeRules.inactive(::typeof(unpackexceptions), args...) = true
EnzymeRules.inactive(::typeof(errormessage), args...) = true
EnzymeRules.inactive(::typeof(handleexceptions), args...) = true
