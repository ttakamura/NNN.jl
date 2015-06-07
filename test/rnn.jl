function toy_rnn()
    batch_size  = 4
    input_size  = 1
    output_size = 1
    hidden_size = 10

    function toy_data(n)
        x = rand(Float32,  input_size,  n*batch_size)
        y = zeros(Float32, output_size, n*batch_size)
        for i in 1:n
            for t in 1:batch_size
                y[i*batch_size] += x[1,(i-1)*batch_size+t]
            end
        end
        (x, y)
    end

    function toy_net(xtrn, ytrn, xtst, ytst, iter)
        net = NNN.Network(batch_size)
        NNN.push!(net,
                  NNN.SimpleRecurrentLayer(batch_size, input_size, hidden_size;
                                           activation=NNN.LeakyRelu()),
                  NNN.SGD(;ε=0.05),
                  NNN.SGD(;ε=0.05))
        NNN.push!(net,
                  NNN.OutputLayer(batch_size, hidden_size, output_size;
                                  activation=NNN.IdentityAct(),
                                  loss=NNN.rnn_quadratic_cost),
                  NNN.SGD(;ε=0.05))

        NNN.train(xtrn, ytrn, net; iter=iter, batch_size=batch_size)
        errors = NNN.test(xtst, ytst, net; batch_size=batch_size)

        (net, errors)
    end

    xtrn, ytrn  = toy_data(1000);
    xtst, ytst  = toy_data(100);
    net, errors = toy_net(xtrn, ytrn, xtst, ytst, 5)
    (net, errors, xtst, ytst)
end

net, errors, xtst, ytst = toy_rnn()

facts("Simple RNN") do
    @fact sum(abs(NNN.forward!(net, xtst[:,1:4])[4] - ytst[:,4])) => less_than(1.0)
end
