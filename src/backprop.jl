# -------------------------------------------
function forward!(net::Network, X::Signal)
    for i in 1:length(net.layers)
        reset!(net.layers[i], X)
        X = forward!(net.layers[i], X)
    end
    X
end

function forward!(l::Layer, X::Signal)
    copy!(l.X, X)
    @into! l.U = l.W * X
    u = @in1! l.U .+ l.b

    forward_activation!(l)

    forward!(l.post_filters, l)
    l.Z
end

function forward_activation!(l::Layer)
    forward!(l.activation, l.U, l.Z)
end

function forward_activation!(l::SimpleRecurrentLayer)
    forward!(l.activation, sub(l.U,:,1), sub(l.Z,:,1))

    @inbounds for t in 2:size(l.U, 2)
        l.Uh[:,t] = l.Wh * l.Z[:,t-1]
        @inbounds for i in 1:size(l.Uh,1)
            l.U[i,t] += l.Uh[i,t]
        end
        forward!(l.activation, sub(l.U,:,t), sub(l.Z,:,t))
    end

    if maximum(l.Z) > 100000.0f0
        @show maximum(l.Z), maximum(l.U), maximum(l.Uh)
        error("Large recurrent forward")
    end
end

# -----------------------------------------------------------
function backprop!(net::Network, X::Signal, Y::Signal)
    forward!(net, X)
    for i in length(net.layers):-1:1
        backprop!(net.layers[i], Y)
        Y = net.layers[i]
    end
end

function backprop!(l::OutputLayer, Y::Signal)
    copy!(l.ΔE, l.loss(:diff, l, Y, l.Z))
    backprop!(l)
end

function backprop!(l::HiddenLayer, l2::Layer)
    Δout = l2.W' * l2.ΔE
    backward!(l.activation, l.U, l.Zd)
    copy!(l.ΔE, (l.Zd .* Δout))
    backprop!(l)
end

function backprop!(l::SimpleRecurrentLayer, l2::Layer)
    Δout = l2.W' * l2.ΔE
    backward!(l.activation, l.U, l.Zd)

    # Recurrent
    l.ΔE[:,end:end] = l.Zd[:,end:end] .* Δout[:,end:end]
    Δfuture = zeros(Float32, size(l.Wh,1))

    @inbounds for t in (size(l.U, 2)-1):-1:1
        @into! Δfuture = l.Wh' * l.ΔE[:,t+1]
        @inbounds for i in 1:size(l.Zd,1)
            l.ΔE[i,t] = l.Zd[i,t] .* (Δout[i,t] + Δfuture[i])
        end
    end

    @into! l.ΔWh = l.ΔE * l.Z'
    _ = @in1! l.ΔWh ./ l.batch_size
    backprop!(l)
end

function backprop!(l::Layer)
    backprop!(l.post_filters, l)
    copy!(l.ΔW, ((l.ΔE * l.X')               ./ l.batch_size))
    copy!(l.Δb, ((l.ΔE * ones(l.batch_size)) ./ l.batch_size))
end
