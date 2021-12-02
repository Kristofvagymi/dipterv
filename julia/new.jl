u0 = Float32[2.0; 0.0] # Initial condition
datasize = 30 # Number of data points
tspan = (0.0f0, 1.5f0) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps for each data point

# Function that will generate the data we are trying to fit
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)' # Need transposes to make the matrix multiplication work
end

# Define the problem with the function above
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Solve and take just the solution array
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Make a neural net with a NeuralODE layer
dudt2 = FastChain((x, p) -> x.^3, # Guess a cubic function
                  FastDense(2, 50, tanh), # Multilayer perceptron for the part we don't know
                  FastDense(50, 2))

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

# Array of predictions from NeuralODE with parameters p starting at initial condition u0
function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred) # Just sum of squared error
    return loss, pred
end

# Callback function to observe training
callback = function (p, l, pred; doplot = true)
  display(l)
  # plot current prediction against data
  plt = scatter(tsteps, ode_data[1,:], label = "data")
  scatter!(plt, tsteps, pred[1,:], label = "prediction")
  if doplot
    display(plot(plt))
  end
  return false
end

# Parameters are prob_neuralode.p
result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          cb = callback)

plot( solve(prob_trueode, Tsit5(), saveat = tsteps) )
