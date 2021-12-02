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

plot( Matrix(ads1[100:200,:]) )

ads1 = ads1[!,["jhu-csse_confirmed_7dav_incidence_prop", "jhu-csse_deaths_7dav_incidence_prop","safegraph_bars_visit_prop", "doctor-visits_smoothed_adj_cli", "google-symptoms_sum_anosmia_ageusia_smoothed_search", "safegraph_restaurants_visit_prop", "hospital-admissions_smoothed_adj_covid19_from_claims", "immune"]]

trainData = disallowmissing(transpose(Matrix(ads1)))

u0 = trainData[:,1]

tspan = (0.0f0, 10f0)
tsteps = range(tspan[1], tspan[2], length = size(trainData)[2])

dudt = FastChain(
  FastDense(8,32,swish),
  FastDense(32,16,swish),
  FastDense(16,8,swish))

dudt = FastChain(
	  FastDense(8,32,swish),
	  FastDense(32,8,tanh))

θ = initial_params(dudt)

group_size = 3
continuity_term = 200
neuralode = NeuralODE(dudt, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->dudt(u,p), u0, tspan, θ)

function loss_function(data, pred)
	loss = sum(abs2, data - pred)
	return loss
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, trainData, tsteps, prob_node,loss_function, Tsit5(),
                          group_size; continuity_term)
end

res_ms = DiffEqFlux.sciml_train(loss_multiple_shooting, θ, maxiters = 1000)

solution = solve(prob_node, Tsit5(), saveat = tsteps)

plot(transpose(trainData))

plot(solution)
