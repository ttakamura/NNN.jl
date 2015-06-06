# --------------------------------------------------
# Activation Functions
# --------------------------------------------------
immutable IdentityAct <: ActivationFunc
end

@inline function forward(f::IdentityAct, x::Float32)
    x
end

@inline function backward(f::IdentityAct, x::Float32)
    x
end

# --------------------------------------------------
function sigmoid(x)
    one(x) / (one(x) + exp(-x))
end

function sigmoid(X::Signal)
    Z = zeros(X)
    for (row, col) in zip(findn(X)...)
        Z[row, col] = sigmoid(X[row, col])
    end
    Z
end

function sigmoid(diff::Symbol, X::Signal)
    Zd = zeros(X)
    for (row, col) in zip(findn(X)...)
        s = sigmoid(X[row, col])
        Zd[row, col] = s * (one(eltype(X)) - s)
    end
    Zd
end

# --------------------------------------------------
function relu(X::Signal)
    bottom = zero(eltype(X))
    Z = zeros(X)
    for (row, col) in zip(findn(X)...)
        Z[row, col] = ifelse((X[row, col] < bottom), bottom, X[row, col])
    end
    Z
end

function relu(diff::Symbol, X::Signal)
    bottom = zero(eltype(X))
    Zd = zeros(X)
    for (row, col) in zip(findn(X)...)
        Zd[row, col] = ifelse((X[row, col] < bottom), bottom, one(eltype(X)))
    end
    Zd
end

# --------------------------------------------------
immutable LeakyRelu <: ActivationFunc
    α::Float32
    LeakyRelu() = new(0.05f0)
end

@inline function forward(f::LeakyRelu, x::Float32)
    ifelse(x < 0.0f0, x * f.α, x)
end

@inline function backward(f::LeakyRelu, x::Float32)
    ifelse(x < 0.0f0, f.α, 1.0f0)
end


# --------------------------------------------------
immutable Softmax <: ActivationFunc
end

function forward!(f::Softmax, X::Signal, Z::Signal)
    @inbounds for n in 1:size(X,2)
        x     = X[:,n]
        exp_x = exp(x .- maximum(x))
        total = sum(exp_x)
        @inbounds for k in 1:size(Z,1)
            Z[k,n] = exp_x[k] / total
        end
    end
    Z
end

function backward!(f::Softmax, X::Signal, Z::Signal)
    copy!(Z, X)
end

# --------------------------------------------------
function forward!(f::ActivationFunc, U::Signal, Z::Signal)
    @inbounds for col=1:size(Z,2), row=1:size(Z,1)
        Z[row, col] = forward(f, U[row,col])
    end
    Z
end

function backward!(f::ActivationFunc, U::Signal, Z::Signal)
    @inbounds for col=1:size(Z,2), row=1:size(Z,1)
        Z[row, col] = backward(f, U[row,col])
    end
    Z
end

function forward!(f::ActivationFunc, U::AbstractArray{Float32,1}, Z::AbstractArray{Float32,1})
    @inbounds for i=1:length(Z)
        Z[i] = forward(f, U[i])
    end
    Z
end

function backward!(f::ActivationFunc, U::AbstractArray{Float32,1}, Z::AbstractArray{Float32,1})
    @inbounds for i=1:length(Z)
        Z[i] = backward(f, U[i])
    end
    Z
end
