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

df_input_mn = DataFrame
df_input_ny = DataFrame
df_input_mi = DataFrame
df_input_mn = CSV.read("..\\python\\data\\adsmn.csv", df_input_mn)
df_input_ny = CSV.read("..\\python\\data\\adsny.csv", df_input_ny)
df_input_mi = CSV.read("..\\python\\data\\adsmi.csv", df_input_mi)

df_input_mn = select!(df_input_mn, Not(:time_value))
df_input_ny = select!(df_input_ny, Not(:time_value))
df_input_mi = select!(df_input_mi, Not(:time_value))

ads1 = df_input_mn[51:end,:]
ads1 = df_input_ny[51:end,:]
ads1 = df_input_mi[51:end,:]

columns = ["jhu-csse_confirmed_7dav_incidence_prop", "jhu-csse_deaths_7dav_incidence_prop","safegraph_bars_visit_prop", "doctor-visits_smoothed_adj_cli", "google-symptoms_sum_anosmia_ageusia_smoothed_search", "safegraph_restaurants_visit_prop", "hospital-admissions_smoothed_adj_covid19_from_claims", "immune"]
columns_short = ["7day_confirmed_prop", "7day_deaths_prop","bars_visit_prop", "doctor-visits", "google_anosmia_ageusia_search", "restaurants_visit_prop", "hospital_admissions", "immune"]
ads1 = ads1[!,columns]

plot( Matrix(ads1) )

ads = disallowmissing(transpose(Matrix(ads1)))

lowerBound = 30
upperBound = 130
valLen = 30

fullTrain = ads[:,lowerBound:upperBound]

validationData = ads[:,upperBound + 1:upperBound + 1 + valLen]

plot(transpose(hcat(fullTrain, validationData)), legend = false)
vline!([upperBound - lowerBound],ls=:dash,c=:black)

function dudt_(u,p,t)
    dudt(u, p)
end

function cb()
  push!(train_losses, loss_mshoot(trainingData))
  push!(val_losses, loss_mshoot(validationData))
end

RMSE(fact, predict) =sqrt( mean( (fact - predict) .* (fact - predict) ) )

MSE(fact, predict) = mean( (fact - predict) .* (fact - predict) )

MAE(fact, predict) = mean( abs((fact - predict)) )

function predict_mshoot(u)
    _prob = remake(prob,u0 = u,p = θ)
    Array(solve(_prob, Tsit5(), saveat = t[1:mshoot_len]))
end

function loss_mshoot(batch)
  losVal = 0
  batch_size = floor(Int32,size(batch)[2] / mshoot_len)
  for index in range(1,step = mshoot_len, length = batch_size)
    pred = predict_mshoot(batch[:,index])
    if loss_method == 'MSE'
      losVal +=  MSE(batch[:,index:index+mshoot_len - 1], pred[:,1:mshoot_len])
    elseif loss_method == 'RMSE'
      losVal +=  RMSE(batch[:,index:index+mshoot_len - 1], pred[:,1:mshoot_len])
    elseif loss_method == 'MAE'
      losVal +=  MAE(batch[:,index:index+mshoot_len - 1], pred[:,1:mshoot_len])
    end
  end

  return losVal / batch_size
end

plotall = function (df1, df2, dash, title)

  l = @layout [a{0.01h}; grid(3,3)]
  pList = []
  push!(pList, plot(title = title, grid = false, showaxis = false, ticks = false, titlefontsize = 8))

  for i in 1:1:size(columns,1)
    p = plot(df1[i,:], title = columns_short[i], titlefontsize = 8, label="")
    p = plot!(df2[i,:])
    p = vline!([dash],ls=:dash,c=:black)

    push!(pList, p)
  end

  push!(pList, plot(grid = false, showaxis = false, ticks = false))

  fullPlot = plot(pList..., legend = false, layout = l)

  mkpath("plots/$title")

  randHash = bytes2hex(rand(UInt8, 4))
  savefig(fullPlot,"plots/$title/$title-$randHash.png")

  plot(train_losses, label = "train", title = title, titlefontsize = 8)
  plt = plot!(val_losses, label = "validation")
  savefig(plt,"plots/$title/$title-losses-$randHash.png")
end

function fit(data, numEpochs, lr = 0.0005, batch_size = 16, mshoot_len = 2, title = "")
  global train_losses = []
  global val_losses = []

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

lrs = [0.001, 0.0005, 0.0001, 0.00005]
batch_sizes = [8, 16, 32]
mshoot_lens = [2, 2, 2, 2, 3, 4]
dudt_sizes = [64, 32]
epochs = [50, 100, 200, 300, 500]

data_batches = [ads[:,30:130], ads[:,200:350]]
val_batches = [ads[:,131:180], ads[:,350:400]]
data_names = ["30_130", "200_350"]

data_batches = [ads[:,30:130], ads[:,100:270], ads[:,300:450], ads[:,450:580]]
val_batches = [ads[:,131:180], ads[:,271:320], ads[:,451:500], ads[:,581:630]]
data_names = ["30_130", "100_270", "300_450", "450_580"]

loss_methods = ["MSE", "RMSE","MAE"]

paramsTried = []
for i in 1:1:200
  lr = lrs[rand(1:size(lrs,1))]
  batch_size = batch_sizes[rand(1:size(batch_sizes,1))]
  global mshoot_len = mshoot_lens[rand(1:size(mshoot_lens,1))]
  dudt_size = dudt_sizes[rand(1:size(dudt_sizes,1))]
  epoch = epochs[rand(1:size(epochs,1))]
  global loss_method = loss_methods[rand(1:size(loss_methods,1))]

  data_index = rand(1:size(data_batches,1))
  global trainingData = data_batches[data_index]
  global validationData = val_batches[data_index]
  data_name = data_names[data_index]

  current_param = (lr, batch_size, mshoot_len, dudt_size, epoch, data_index)
  if !(current_param in paramsTried)
    push!(paramsTried, current_param)
    tspan = (0.0f0,10.0f0)
    global t = range(tspan[1], tspan[2], length=size(trainingData,2))

    if dudt_size == 64
      global dudt = FastChain(
        FastDense(8,64,swish),
        FastDense(64,32,swish),
        FastDense(32,16,swish),
        FastDense(16,8))

      global θ = initial_params(dudt)
    else
      global dudt = FastChain(
        FastDense(8,32,swish),
        FastDense(32,24,swish),
        FastDense(24,16,swish),
        FastDense(16,8))

      global θ = initial_params(dudt)
    end
    mainTitle = "NY_avg_$loss_method-$data_name-lr_$lr-epoch_$epoch-dudtsize_$dudt_size-mshoot_lens_$mshoot_len-batch_size_$batch_size"

    println(mainTitle)

    fit(trainingData, epoch, lr, batch_size, mshoot_len, mainTitle)
  end
end
