import Base: put!, wait, isready, take!, fetch
mutable struct BufferChannel <: AbstractChannel{Any}
    buffer::Dict{Int, GameHistory}
    cond_take::Condition    # waiting for data to become available
    BufferChannel() = new(Dict(), Condition())
end

"""
Save key and value to remote buffer
"""
function put!(buffer_channel::BufferChannel, k::Int, v::GameHistory)
    buffer_channel.buffer[k] = v
    notify(buffer_channel.cond_take)
    return buffer_channel
end

"""
Pop the key and its associated value from remote buffer
"""
function take!(buffer_channel::BufferChannel, k::Int)
    v=fetch(buffer_channel,k)
    delete!(buffer_channel.buffer, k)
    return v
end

isready(buffer_channel::BufferChannel) = length(buffer_channel.buffer) ≥ 1
isready(buffer_channel::BufferChannel, k::Int) = haskey(buffer_channel.buffer,k)

"""
Obtain Value from the remote buffer for a given Key 
"""
function fetch(buffer_channel::BufferChannel, k::Int)
    wait(buffer_channel,k)
    return buffer_channel.buffer[k]
end

function wait(buffer_channel::BufferChannel, k::Int)
    while !isready(buffer_channel, k)
        wait(buffer_channel.cond_take)
    end
end

function wait(buffer_channel::BufferChannel)
    while !isready(buffer_channel)
        wait(buffer_channel.cond_take)
    end
end

"""
Returns the remote buffer as a Dict
"""
function fetch(buffer_channel::BufferChannel)
    wait(buffer_channel)
    return buffer_channel.buffer
end