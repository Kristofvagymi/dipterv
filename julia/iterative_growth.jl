using DifferentialEquations, DiffEqFlux, Flux
using Plots
using CSV, DataFrames
using Missings
using CUDA
import Random
using Statistics
using IterTools: ncycle
Random.seed!(1234)

df_input = DataFrame
df_input = CSV.read("..\\python\\data\\adsmi.csv", df_input)
df_input = select!(df_input, Not(:time_value))

ads1 = df_input[51:end,:]

plot( Matrix(ads1) )

ads1 = ads1[!,["jhu-csse_confirmed_7dav_incidence_prop", "jhu-csse_deaths_7dav_incidence_prop","safegraph_bars_visit_prop", "doctor-visits_smoothed_adj_cli", "google-symptoms_sum_anosmia_ageusia_smoothed_search", "safegraph_restaurants_visit_prop", "hospital-admissions_smoothed_adj_covid19_from_claims", "immune"]]

ads = disallowmissing(transpose(Matrix(ads1)))

fullTrain = ads[:,30:150]

validationData = ads[:,151:180]

plot(transpose(hcat(fullTrain, validationData)), legend = false)
vline!([121],ls=:dash,c=:black)

dudt = FastChain(
  FastDense(8,64,swish),
  FastDense(64,32,swish),
  FastDense(32,16,swish),
  FastDense(16,8,swish))

θ = initial_params(dudt)

function dudt_(u,p,t)
    dudt(u, p)
end

train_losses = []
val_losses = []
function cb()
  push!(train_losses, loss_mshoot(trainData))
  push!(val_losses, loss_mshoot(validationData))
end

RMSE(fact, predict) =sqrt( mean( (fact - predict) .* (fact - predict) ) )

MSE(fact, predict) = mean( (fact - predict) .* (fact - predict) )

function predict_mshoot(u)
    _prob = remake(prob,u0=u,p=θ)
    Array(solve(_prob, Tsit5(), saveat = t[1:2]))
end

function loss_mshoot(batch)
  losVal = 0

  for index in range(1,step = 2, length = floor(Int32,size(batch)[2] / 2))
    pred = predict_mshoot(batch[:,index])
    losVal +=  MSE(batch[:,index+1], pred[:,2])
  end

  return losVal
end

function fit(data)

  train_loader = CustomDataLoader(data, batchsize = k, shuffle=true)

  u0 = data[:,1]
  prob = ODEProblem{false}(dudt_, u0, tspan, θ)

  numEpochs = 200
  opt = ADAM(0.01)

  Flux.train!(loss_mshoot, Flux.params(θ, u0), ncycle(train_loader, numEpochs), opt, cb = cb)
end

k = 16
tspan = (0.0f0,10.0f0)
t = range(tspan[1], tspan[2], length=size(fullTrain)[2])

step1 = fullTrain[:,1:71]
fit(step1)
step2 = fullTrain
fit(step2)

u0 = fullTrain[:,1]

_prob = remake(prob,u0=u0,p=θ)
result = solve(_prob, Tsit5(),saveat = t)

plot(transpose(fullTrain), legend = false)
plot!(transpose(result), legend = false)

inf_u0 = result[:,end]

t_v = range(tspan[1], tspan[2], length=size(validationData)[2])
_prob = remake(prob,u0=inf_u0,p=θ)
result_v = solve(_prob, Tsit5(),saveat = t_v)


plot(transpose(hcat(hcat(result.u...),hcat(result_v.u...))), legend = false)
plot!(transpose(hcat(trainData, validationData)), legend = false)
vline!([71],ls=:dash,c=:black)

train_losses
val_losses

plot(train_losses)
plot!(val_losses)
