typealias Weight Matrix{Float32}
typealias Bias   Vector{Float32}
typealias Signal Matrix{Float32}

abstract Optimizer
abstract Layer
abstract Filter
abstract RecurrentLayer <: Layer
abstract ActivationFunc

immutable HiddenLayer <: Layer
    batch_size::Int64
    W::Weight
    ΔW::Weight
    b::Bias
    Δb::Bias
    ΔE::Signal
    X::Signal
    U::Signal
    Z::Signal
    Zd::Signal
    activation::ActivationFunc
    post_filters::Vector{Filter}

    function HiddenLayer(batch_size, x_size, h_size; activation=LeakyRelu(), post_filters=Filter[])
        w_in    = randn(h_size, x_size)  * 0.1
        dw_in   = zeros(h_size, x_size)
        b       = zeros(h_size)
        db      = zeros(h_size)
        de      = zeros(h_size, batch_size)
        x       = zeros(x_size, batch_size)
        u       = zeros(h_size, batch_size)
        z       = zeros(h_size, batch_size)
        dz      = zeros(h_size, batch_size)
        new(batch_size, w_in, dw_in, b, db, de, x, u, z, dz, activation, post_filters)
    end
end

immutable OutputLayer <: Layer
    batch_size::Int64
    W::Weight
    ΔW::Weight
    b::Bias
    Δb::Bias
    ΔE::Signal
    X::Signal
    U::Signal
    Z::Signal
    Zd::Signal
    activation::ActivationFunc
    loss::Function
    post_filters::Vector{Filter}

    function OutputLayer(batch_size, x_size, y_size; activation=IdentityAct(), loss=quadratic_cost, post_filters=Filter[])
        w_in    = randn(y_size, x_size)  * 0.05
        dw_in   = zeros(y_size, x_size)
        b       = zeros(y_size)
        db      = zeros(y_size)
        de      = zeros(y_size, batch_size)
        x       = zeros(x_size, batch_size)
        u       = zeros(y_size, batch_size)
        z       = zeros(y_size, batch_size)
        dz      = zeros(y_size, batch_size)
        new(batch_size, w_in, dw_in, b, db, de, x, u, z, dz, activation, loss, post_filters)
    end
end

immutable SimpleRecurrentLayer <: RecurrentLayer
    batch_size::Int64
    W::Weight
    ΔW::Weight
    Wh::Weight
    ΔWh::Weight
    b::Bias
    Δb::Bias
    ΔE::Signal
    X::Signal
    U::Signal
    Uh::Signal
    Z::Signal
    Zd::Signal
    activation::ActivationFunc
    post_filters::Vector{Filter}

    function SimpleRecurrentLayer(batch_size, x_size, h_size; activation=LeakyRelu(), post_filters=Filter[])
        w_in  = randn(h_size, x_size)  * 0.1
        dw_in = zeros(h_size, x_size)
        w_h   = eye(h_size) * 0.98
        dw_h  = zeros(h_size, h_size)
        b     = zeros(h_size)
        db    = zeros(h_size)
        ΔE   = zeros(h_size, batch_size)
        X     = zeros(x_size, batch_size)
        U     = zeros(h_size, batch_size)
        Uh    = zeros(h_size, batch_size)
        Z     = zeros(h_size, batch_size)
        Zd    = zeros(h_size, batch_size)
        layer = new(batch_size, w_in, dw_in, w_h, dw_h, b, db, ΔE, X, U, Uh, Z, Zd, activation, post_filters)
        reset!(layer, zeros(Float32, x_size, batch_size); force=true)
        layer
    end
end

immutable Network
    batch_size::Int64
    layers::Array{Layer}
    optimizers::Array{Optimizer}
    rnn_optimizers::Array{Optimizer}

    function Network(batch_size)
        new(batch_size, Layer[], Optimizer[], Optimizer[])
    end
end

function push!(net::Network, layer::Layer, optimizer::Optimizer)
    Base.push!(net.layers,         layer)
    Base.push!(net.optimizers,     optimizer)
    Base.push!(net.rnn_optimizers, optimizer)
    true
end

function push!(net::Network, layer::Layer, optimizer::Optimizer, rnn_optimizer::Optimizer)
    Base.push!(net.layers,         layer)
    Base.push!(net.optimizers,     optimizer)
    Base.push!(net.rnn_optimizers, rnn_optimizer)
    true
end

function reset!(l::Layer, nextX::Signal)
    # nothing to do
end

function reset!(l::SimpleRecurrentLayer, nextX::Signal; force=false)
    x_size     = size(l.W, 2)
    h_size     = size(l.W, 1)
    batch_size = size(nextX, 2)

    if force || batch_size != size(l.X,2)
        copy!(l.ΔE, zeros(h_size, batch_size))
        copy!(l.X  , zeros(x_size, batch_size))
        copy!(l.U  , zeros(h_size, batch_size))
        copy!(l.Uh , zeros(h_size, batch_size))
        copy!(l.Z  , zeros(h_size, batch_size))
        copy!(l.Zd , zeros(h_size, batch_size))
    end
end
