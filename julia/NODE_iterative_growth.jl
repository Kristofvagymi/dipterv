using DifferentialEquations, DiffEqFlux, Flux
using Plots
using Plots: grid
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
Matrix(ads1)[1:3,:]
plot( Matrix(ads1) )
plot( Matrix(ads1)[30:130,:], legend = false )
plot( Matrix(ads1)[100:270,:], legend = false )
plot( Matrix(ads1)[300:450,:], legend = false)
plot( Matrix(ads1)[450:600,:], legend = false )

columns = ["jhu-csse_confirmed_7dav_incidence_prop", "jhu-csse_deaths_7dav_incidence_prop","safegraph_bars_visit_prop", "doctor-visits_smoothed_adj_cli", "google-symptoms_sum_anosmia_ageusia_smoothed_search", "safegraph_restaurants_visit_prop", "hospital-admissions_smoothed_adj_covid19_from_claims", "immune"]
columns_short = ["7day_confirmed_prop", "7day_deaths_prop","bars_visit_prop", "doctor-visits", "google_anosmia_ageusia_search", "restaurants_visit_prop", "hospital_admissions", "immune"]
ads1 = ads1[!,columns]

ads = disallowmissing(transpose(Matrix(ads1)))

lowerBound = 30
upperBound = 130
valLen = 30

fullTrain = ads[:,lowerBound:upperBound]

validationData = ads[:,upperBound + 1:upperBound + 1 + valLen]

plot(transpose(hcat(fullTrain, validationData)), legend = false)
vline!([upperBound - lowerBound],ls=:dash,c=:black)

dudt = FastChain(
  FastDense(8,64,swish),
  FastDense(64,32,swish),
  FastDense(32,16,swish),
  FastDense(16,8))

θ = initial_params(dudt)

dudt = FastChain(
  FastDense(8,32,swish),
  FastDense(32,16,swish),
  FastDense(16,8,swish))

θ = initial_params(dudt)

function dudt_(u,p,t)
    dudt(u, p)
end

train_losses = []
val_losses = []
function cb()
  push!(train_losses, loss_mshoot(trainingData))
  push!(val_losses, loss_mshoot(validationData))
end

RMSE(fact, predict) =sqrt( mean( (fact - predict) .* (fact - predict) ) )

MSE(fact, predict) = mean( (fact - predict) .* (fact - predict) )

function predict_mshoot(u)
    _prob = remake(prob,u0 = u,p = θ)
    Array(solve(_prob, Tsit5(), saveat = t[1:mshoot_len]))
end

function loss_mshoot(batch)
  losVal = 0
  batch_size = floor(Int32,size(batch)[2] / mshoot_len)
  for index in range(1,step = mshoot_len, length = batch_size)
    pred = predict_mshoot(batch[:,index])
    losVal +=  MSE(batch[:,index:index+mshoot_len - 1], pred[:,1:mshoot_len])
  end

  return losVal# / batch_size
end

plotall = function (df1, df2, dash, title)

  l = @layout [a{0.01h}; grid(3,3)]
  pList = []
  push!(pList, plot(title = title, grid = false, showaxis = false, ticks = false))

  for i in 1:1:size(columns,1)
    p = plot(df1[i,:], title = columns_short[i], titlefontsize = 8, label="")
    p = plot!(df2[i,:])
    p = vline!([dash],ls=:dash,c=:black)

    push!(pList, p)
  end

  push!(pList, plot(grid = false, showaxis = false, ticks = false))

  fullPlot = plot(pList..., legend = false, layout = l)

  mkpath("plots")

  randHash = bytes2hex(rand(UInt8, 4))
  savefig(fullPlot,"plots/$title-$randHash.png")

  plot(train_losses, label = "train")
  plt = plot!(val_losses, label = "validation")
  savefig(plt,"plots/$title-losses-$randHash.png")
end

function fit(data, numEpochs, title, lr = 0.0005, batch_size = 16, mshoot_len = 2)

  u0 = data[:,1]
  global prob = ODEProblem{false}(dudt_, u0, tspan, θ)
  k = batch_size
  train_loader = CustomDataLoader(data, batchsize = k, shuffle=true, mshoot_len = mshoot_len)

  opt = ADAM(lr)
  global trainingData = data
  Flux.train!(loss_mshoot, Flux.params(θ), ncycle(train_loader, numEpochs), opt, cb = cb)

  _prob = remake(prob, u0 = u0, p = θ)
  result = solve(_prob, Tsit5(),saveat = t)

  inf_u0 = result[:,end]
  _prob = remake(prob,u0=inf_u0,p=θ)
  result_v = solve(_prob, Tsit5(),saveat = t)

  plotall(hcat(hcat(result.u...),hcat(result_v.u...)[:,1:size(validationData,2)]), hcat(data, validationData), size(data, 2), title)
end

mshoot_len = 2
tspan = (0.0f0,10.0f0)
step1 = ads[:,30:130]
validationData = ads[:,131:161]

t = range(tspan[1], tspan[2], length=size(step1,2))
println()
@time fit(step1, 300, "641100mse")

_prob = remake(prob, u0 = step1[:,1], p = θ)
result = solve(_prob, Tsit5(),saveat = t)

plotall(result, step1, 100, "Main title")

upperBound = 300
for recNum in 1:100:upperBound
  step = ads[:,recNum : recNum + 100]
  fit(step, 100)
end

inf_u0 = result[:,end]

_prob = remake(prob,u0=inf_u0,p=θ)
result_v = solve(_prob, Tsit5(),saveat = t)

plotall(hcat(hcat(result.u...),hcat(result_v.u...)[:,1:size(validationData,2)]), hcat(step1, validationData), 100, "")

train_losses
val_losses
