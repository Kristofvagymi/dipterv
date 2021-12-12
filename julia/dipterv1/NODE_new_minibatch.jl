using DifferentialEquations, DiffEqFlux, Flux
using Plots
using CSV, DataFrames
using Missings
using CUDA
import Random
using Statistics
Random.seed!(1234)

df_input = DataFrame
df_input = CSV.read("..\\python\\data\\adsmi.csv", df_input)
df_input = select!(df_input, Not(:time_value))

ads1 = df_input[51:end,:]
ads1 = ads1[!,["jhu-csse_confirmed_7dav_incidence_prop", "jhu-csse_deaths_7dav_incidence_prop","safegraph_bars_visit_prop", "doctor-visits_smoothed_adj_cli", "google-symptoms_sum_anosmia_ageusia_smoothed_search", "safegraph_restaurants_visit_prop", "hospital-admissions_smoothed_adj_covid19_from_claims"]]

trainData = plot( Matrix(ads1) )

trainData = disallowmissing(transpose(Matrix(ads1)))

dudt = Chain(
  Dense(7,10,swish),
  Dense(10,7, tanh))

tspan = (0.0f0,25.0f0)
t = range(tspan[1],tspan[2],length=2)

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t, abstol=1e-2,reltol=1e-2)

u0 = trainData[:,1]
predict = n_ode(u0).u[2]

loss_n_ode(fact, predict) = sqrt( mean( (fact - predict) .* (fact - predict) ) )

k = 32
train_loader = Flux.Data.DataLoader(trainData, batchsize = k, shuffle=true)

θ = Flux.params(n_ode)
opt = ADAM(0.03)

for batch in ncycle(train_loader,1)
  losses = []
  for j in range(1,step = 2, length = size(batch)[2])
    index = floor(Int8,j)

    u0 = batch[:,index]
    u1 = batch[:,index + 1]

    result = n_ode(u0).u[2]
    loss = loss_n_ode(result, u1)
    push!(losses, loss)

  end
end
