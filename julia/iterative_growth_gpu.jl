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

columns = ["jhu-csse_confirmed_7dav_incidence_prop", "jhu-csse_deaths_7dav_incidence_prop","safegraph_bars_visit_prop", "doctor-visits_smoothed_adj_cli", "google-symptoms_sum_anosmia_ageusia_smoothed_search", "safegraph_restaurants_visit_prop", "hospital-admissions_smoothed_adj_covid19_from_claims", "immune"]
columns_short = ["7day_confirmed_prop", "7day_deaths_prop","bars_visit_prop", "doctor-visits", "google_anosmia_ageusia_search", "restaurants_visit_prop", "hospital_admissions", "immune"]
ads1 = ads1[!,columns]

ads = disallowmissing(transpose(Matrix(ads1)))

dudt = FastChain(
  FastDense(8,64,swish),
  FastDense(64,32,swish),
  FastDense(32,16,swish),
  FastDense(16,8))

θ = initial_params(dudt) |> gpu

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

function loss_mshoot(batch)
  batch = batch |> gpu
  losVal = 0
  batch_size = floor(Int32,size(batch)[2] / mshoot_len)
  for index in range(1,step = mshoot_len, length = batch_size)
    pred = prob_neuralode(batch[:,index], θ)
    losVal +=  MSE(batch[:,index:index+mshoot_len - 1], hcat(pred.u...))
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
end

function fit(data, numEpochs, title)
  u0 = data[:,1] |> gpu

  global train_losses = []
  global val_losses = []

  global prob = ODEProblem{false}(dudt_, u0, tspan, θ)
  k = 16
  train_loader = CustomDataLoader(data, batchsize = k, shuffle=true, mshoot_len = mshoot_len)

  opt = ADAM(0.0005)
  global trainingData = data
  Flux.train!(loss_mshoot, Flux.params(θ), ncycle(train_loader, numEpochs), opt)#, cb = cb)

  node = NeuralODE(dudt, tspan, Tsit5(), saveat = t)
  plotall( node(step1[:,1] |> gpu,  θ), step1, 100, "")
  #plotall(hcat(hcat(result.u...),hcat(result_v.u...)[:,1:size(validationData,2)]), hcat(data, validationData), size(data, 2), "")
end

mshoot_len = 2
tspan = (0.0f0,10.0f0)
step1 = ads[:,30:130]
validationData = ads[:,131:161]
prob_neuralode = NeuralODE(dudt, tspan, Tsit5(), saveat = t[1:mshoot_len])
t = range(tspan[1], tspan[2], length=size(step1,2))
fit(step1, 200, "64, 1:100, mse")

node = NeuralODE(dudt, tspan, Tsit5(), saveat = t)
plotall( node(step1[:,1] |> gpu,  θ), step1, 100, "")

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

plot(train_losses, label = "train")
plot!(val_losses, label = "validation")
