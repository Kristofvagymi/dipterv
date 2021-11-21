using CSV
using DataFrames
using Plots
using Flux, DiffEqFlux
using DifferentialEquations

url_confirmed_cases= "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
url_death = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

download(url_confirmed_cases, "confirmed_cases_global.csv");
download(url_death, "death_global.csv");
download(url_recovered, "recovered_global.csv");

# read, clean and view the data:
data_confirmed_cases = DataFrame
data_death = DataFrame
data_recovered = DataFrame

data_confirmed_cases = CSV.read("confirmed_cases_global.csv", data_confirmed_cases)
data_death = CSV.read("death_global.csv",data_death)
data_recovered = CSV.read("recovered_global.csv",data_recovered)

data_us_confirmed = data_confirmed_cases[data_confirmed_cases[!,"Country/Region"] .== "US",:]
data_death = data_death[data_death[!,"Country/Region"] .== "US",:]
data_recovered = data_recovered[data_recovered[!,"Country/Region"] .== "US",:]

us_vec_confirmed = vec(Array(data_us_confirmed)[75:end])
us_vec_death = vec(Array(data_death)[75:end])
us_vec_recovered = vec(Array(data_recovered)[75:end])
us_vec_recovered[length(us_vec_recovered)] = 6349082

plot(us_vec_confirmed,xlabel = "Days", label = "Infected")
plot!(us_vec_death, label ="Death")
plot!(us_vec_recovered, label = "Recovered")

datasize = Float32(length(us_vec_confirmed))

US_population = 328000000

I = us_vec_confirmed ./ US_population .* 100
R = ( (us_vec_death .+ us_vec_recovered) ./ US_population ) .* 100
S =  (((vec((zeros(1,length(us_vec_confirmed)) .+ US_population) ) .- us_vec_confirmed ) .- us_vec_death ) ./ US_population ) .* 100

trainData = transpose(hcat(S, I, R))

trainData


# NerualODE
trainData = transpose(hcat(S, I, R))
u0 = trainData[1:3,1]

trainData


tspan = (0.0f0, 1.5f0)
t = range(tspan[1],tspan[2],length=length(us_vec_confirmed))

dudt = Chain(Dense(3,20,tanh), Dense(20,3))

dudt = FastChain((x,p)->[x;x[1]*x[2]],
    FastDense(4,3,tanh),
    FastDense(3,3)
)
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t, reltol=1e-7,abstol=1e-9)

function predict_n_ode()
  n_ode(u0)
end

loss_n_ode() = sum(abs2,trainData[1:3,:] .- predict_n_ode()[1:3,:])

cb = function () #callback function to observe training
  display(loss_n_ode())
end

ps = Flux.params(n_ode)
data = Iterators.repeated((), 100)
opt = ADAM(0.1)

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

plot(hcat(S, I, R)[:,2:3], label=["True I" "True R"])
plot!(transpose(predict_n_ode())[:,2:3], label=["Predicted I" "Predicted R"])





#Param tuning

D = us_vec_death ./ US_population .* 100

u0 = [S[1],I[1],R[1], D[1]] # S,I.R, D
p = [0.05,10.0,0.25,0.05]; # β,c,γ
trainData = transpose(hcat(S, I, R, D))

tspan = (0.0f0, 1.5f0)
t = range(tspan[1],tspan[2],length=length(us_vec_confirmed))

function SIR(du,u,p,t)
  (S,I,R) = u
  (β,c,γ,α) = p
  N = S+I+R
  infection = β*c*I/N*S
  recovery = γ*I
  dead = α*I

  du[1] = -infection
  du[2] = infection - recovery
  du[3] = recovery
  du[4] = dead
end

params = Flux.params(p)

prob = ODEProblem(SIR,u0,tspan,p)

function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p,saveat=t, reltol=1e-7,abstol=1e-9) # override with new parameters
end

loss_ode() = sum(abs2,trainData[1:3,:] .- predict_rd()[1:3,:])

data = Iterators.repeated((), 500)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_ode())
end

Flux.train!(loss_ode, params, data, opt, cb = cb)

plot(hcat(S, I, R, D), label=["Predicted S" "True I" "True R" "True D"])
plot!(transpose(predict_rd()), label=["Predicted S" "Predicted I" "Predicted R" "Predicted D"])



new_tspan = (0f0, 8f0)
t = range(new_tspan[1],new_tspan[2],length=length(us_vec_confirmed)*4)

new_prob = ODEProblem(SIR,u0,new_tspan,p)

full_sol = solve(new_prob,Tsit5(),p=p,saveat=t)
plot(transpose( full_sol ),xlabel="Days",ylabel="Percentage", label=["Predicted S" "Predicted I" "Predicted R" "Predicted D"])




#N_ode on full data
trainData = full_sol[1:3,:]
u0 = trainData[1:3,1]

new_tspan = (0f0, 8f0)
t = range(new_tspan[1],new_tspan[2],length=length(us_vec_confirmed)*4)

length(t)

dudt = FastChain((x,p)->[x;x[1]*x[2]],
    FastDense(4,10,tanh),
    FastDense(10,3)
)
n_ode_f = NeuralODE(dudt,new_tspan,Tsit5(),saveat=t, reltol=1e-7,abstol=1e-9)

function predict_n_ode()
  n_ode_f(u0)
end

loss_n_ode() = sum(abs2,trainData[1:3,:] .- predict_n_ode()[1:3,:])

cb = function () #callback function to observe training
  display(loss_n_ode())
end

ps = Flux.params(n_ode_f)
data = Iterators.repeated((), 100)
opt = ADAM(0.1)

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

plot(transpose(trainData), label=["True S" "True I" "True R"])
plot!(transpose(predict_n_ode()), label=["Predicted S" "Predicted I" "Predicted R"])
