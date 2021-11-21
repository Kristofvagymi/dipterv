
using DifferentialEquations, DiffEqFlux, Flux
using Plots
using CSV, DataFrames
using Missings
using CUDA
import Random
Random.seed!(1234)
using IterTools: ncycle


df_input = DataFrame
df_input = CSV.read("smoothed_normalized.csv", df_input)
df_input = select!(df_input, Not(:time_value))

df_input = df_input[46:end,:]

# End of second wave
df_selected = df_input[186:291, :]

# Start of second wave
df_selected = df_input[160:240, :]

trainData = disallowmissing(transpose(Matrix(df_selected)))
plot(transpose(trainData))

dudt = FastChain(
  FastDense(6,12,swish),
  FastDense(12,6,tanh))

θ = initial_params(dudt)

k = 32
train_loader = Flux.Data.DataLoader(trainData, batchsize = k, shuffle=true)

function predict_node(batch)
  u0 = batch[:,1]
  time_batch = range(tspan[1],tspan[2],length=length(batch[1,:]))
  _prob = remake(prob,u0=batch[:,1],p=θ)
  Array(solve(_prob, Tsit5(),saveat = time_batch))
end

function loss(batch)
  sum(abs2, batch - predict_node(batch))
end

function dudt_(u,p,t)
    dudt(u, p)
end

u0 = trainData[:,1]
tspan = (0.0f0,10.0f0)
prob = ODEProblem{false}(dudt_, u0, tspan, θ)

function cb()
  display(loss(trainData))
end

numEpochs = 100
opt=ADAM(0.1)
Flux.train!(loss, Flux.params(θ), ncycle(train_loader,numEpochs), opt; cb = cb)

t = range(tspan[1],tspan[2],length=length( trainData[1,:]))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t, abstol=1e-2,reltol=1e-2)
plot(transpose(n_ode(u0)))
plot!(transpose(trainData))
