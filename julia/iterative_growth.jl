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
ads1 = ads1[!,columns]

ads = disallowmissing(transpose(Matrix(ads1)))

lowerBound = 30
upperBound = 180
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

plotall = function (df1, df2, dash)
  pList = []
  for i in 1:1:size(columns,1)
    p = plot(df1[i,:])
    p = plot!(df2[i,:])
    p = title!(columns[i])
    p = vline!([dash],ls=:dash,c=:black)

    push!(pList, p)
  end
  print(pList)
  plot(pList..., legend = false)
end

function fit(data, numEpochs)
  u0 = data[:,1]

  global train_losses = []
  global val_losses = []

  global prob = ODEProblem{false}(dudt_, u0, tspan, θ)
  k = 16
  train_loader = CustomDataLoader(data, batchsize = k, shuffle=true)

  opt = ADAM(0.01)
  global trainingData = data
  Flux.train!(loss_mshoot, Flux.params(θ), ncycle(train_loader, numEpochs), opt, cb = cb)

  _prob = remake(prob, u0 = u0, p = θ)
  result = solve(_prob, Tsit5(),saveat = t)

  plot(transpose(data), legend = false)
  plot!(transpose(result), legend = false)

  plotall(data, result)
end

step1 = ads[:,200:300]
tspan = (0.0f0,10.0f0)
t = range(tspan[1], tspan[2], length=size(step1)[2])
fit(step1, 100)

upperBound = 300
for recNum in 1:100:upperBound
  step = ads[:,recNum : recNum + 100]
  fit(step, 100)
end

fit(ads[:,1:301], 100)

step2 = fullTrain
fit(step2)

inf_u0 = result[:,end]

t_v = range(tspan[1], tspan[2], length=size(validationData)[2])
_prob = remake(prob,u0=inf_u0,p=θ)
result_v = solve(_prob, Tsit5(),saveat = t_v)


plot(transpose(hcat(hcat(result.u...),hcat(result_v.u...))), legend = false)
plot!(transpose(hcat(step1, validationData)), legend = false)
vline!([71],ls=:dash,c=:black)

train_losses
val_losses

plot(train_losses, label = "train")
plot!(val_losses, label = "validation")

plotall = function (df1, df2, dash)
  pList = []
  for i in 1:1:size(columns,1)
    p = plot(df1[i,:])
    p = plot!(df2[i,:])
    p = title!(columns[i])
    p = vline!([dash],ls=:dash,c=:black)

    push!(pList, p)
  end
  print(pList)
  plot(pList..., legend = false)
end
plotall(ads, result,100)
