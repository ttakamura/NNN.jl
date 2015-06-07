# include(Pkg.dir("KUnet/test/mnist.jl"))

reload("src/NNN.jl")

const batch_size  = 784
const input_size  = 1
const output_size = 10
const hidden_size = 100

function toy_data(n)
    x = zeros(Float32, input_size,  n*batch_size)
    y = zeros(Float32, output_size, n*batch_size)

    for i in 0:batch_size:((n-1)*batch_size)
        mnis_i = floor(Int64, rand() * 50000) + 1
        mnis_x = MNIST.xtrn[:, mnis_i]
        mnis_y = MNIST.ytrn[:, mnis_i]

        for j in 1:batch_size
            pos = i + j
            x[:, pos] = mnis_x[j]
            y[:, pos] = mnis_y
        end
    end

    (x, y)
end

function rnn_train(xtrn, ytrn, xtst, ytst, iter)
    net = NNN.Network(batch_size)
    NNN.push!(net,
             NNN.SimpleRecurrentLayer(batch_size, input_size, hidden_size;
                                     activation=NNN.LeakyRelu()),
             NNN.AdaGrad(input_size, hidden_size;
                        ε=0.2,
                        μ=0.05,
                        λ=0.0000001),
             NNN.AdaGrad(hidden_size, hidden_size;
                        ε=0.2,
                        μ=0.05,
                        λ=0.000006))
    NNN.push!(net,
             NNN.OutputLayer(batch_size, hidden_size, output_size;
                            activation=NNN.Softmax(),
                            loss=NNN.rnn_cross_entropy_cost),
             NNN.SGD(;λ=0.000001,
                    ε=0.05,
                    gradient_clip=10.0f0))

    @time NNN.train(xtrn, ytrn, net; iter=iter, batch_size=batch_size)
    errors = NNN.test(xtst, ytst, net; batch_size=batch_size)

    (net, errors)
end

xtrn, ytrn = toy_data(2000);
xtst, ytst = toy_data(100);
net, errors = rnn_train(xtrn, ytrn, xtst, ytst, 1); errors

# NNN.train(xtrn, ytrn, net; iter=10, batch_size=batch_size)
