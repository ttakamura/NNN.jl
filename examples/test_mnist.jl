# include(Pkg.dir("KUnet/test/mnist.jl"))
using MNIST: xtrn, ytrn, xtst, ytst

reload("src/NNN.jl")

const batch_size  = 100
const input_size  = 784
const output_size = 10
const hidden_size = 100

function mnist_train(xtrn, ytrn, xtst, ytst)
    drop = NNN.Dropout(0.5)

    net = NNN.Network(batch_size)
    NNN.push!(net,
             NNN.HiddenLayer(batch_size, input_size,  hidden_size;
                            activation=NNN.relu,
                            post_filters=NNN.Filter[ drop ]),
             NNN.AdaGradRDA(input_size, hidden_size;
                           μ=0.1,
                           λ=0.00001))
    NNN.push!(net,
             NNN.OutputLayer(batch_size, hidden_size, output_size;
                            activation=NNN.softmax,
                            loss=NNN.cross_entropy_cost),
             NNN.SGD(;λ=0.0000001,
                    ε=0.2))

    drop.enabled = false
    NNN.train(xtrn, ytrn, net; iter=3, batch_size=batch_size)
    NNN.test(xtst,  ytst, net; batch_size=batch_size)
end

mnist_train(xtrn, ytrn, xtst, ytst)
