using DifferentialEquations, DiffEqFlux, Flux
using Plots
using CSV, DataFrames
using Missings
using CUDA
import Random
Random.seed!(1234)

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

dudt = Chain(
  Dense(6,6,swish),
  Dense(6,6, tanh)) |> gpu

tspan = (0.0f0,25.0f0)
t = range(tspan[1],tspan[2],length=length( trainData[1,:]))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t, abstol=1e-2,reltol=1e-2)

u0 = trainData[:,1] |> gpu
trainData = trainData |>gpu

function predict_n_ode()
  n_ode(u0)
end

loss_n_ode() = sum(abs2, trainData - hcat(predict_n_ode().u...))

losses = []
cb = function () #callback function to observe training
  loss = loss_n_ode()
  display(loss)
  push!(losses, loss)
end

ps = Flux.params(n_ode)
data = Iterators.repeated((), 250)
opt = ADAM(0.03)

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

plot(transpose(trainData))
plot!(transpose(predict_n_ode()))

plot(losses, legend=false)
title!("Loss")
xlabel!("Epoch")

ps

plotall = function (df1, df2, dash)
  p1 = plot(df1[1,:])
  p1 = plot!(df2[1,:])
  p1 = title!("Related doctor visits")
  p1 = vline!([dash],ls=:dash,c=:black)

  p2 = plot(df1[2,:])
  p2 = plot!(df2[2,:])
  p2 = vline!([dash],ls=:dash,c=:black)
  p2 = title!("Confirmed cases")

  p3 = plot(df1[3,:])
  p3 = plot!(df2[3,:])
  p3 = vline!([dash],ls=:dash,c=:black)
  p3 = title!("FB CLI")

  p4 = plot(df1[4,:])
  p4 = plot!(df2[4,:])
  p4 = vline!([dash],ls=:dash,c=:black)
  p4 = title!("FB CMNTY CLI")

  p5 = plot(df1[5,:])
  p5 = title!("Google search")
  p5 = plot!(df2[5,:])
  p5 = vline!([dash],ls=:dash,c=:black)

  p6 = plot(df1[6,:])
  p = title!("Hospital admission")
  p6 = plot!(df2[6,:])
  p6 = vline!([dash],ls=:dash,c=:black)

  plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), legend = false)
end
plotall(trainData, predict_n_ode(),0)

u_inf = predict_n_ode()[end]

training = predict_n_ode() |> cpu
infer = n_ode(u_inf) |> cpu

infer_start = infer[:,2:45]

real_data = disallowmissing(transpose(Matrix(df_input[186:336, :])))
real_data = disallowmissing(transpose(Matrix(df_input[160:285, :])))

plotall(real_data, hcat(training, infer_start), 106)
plotall(real_data, hcat(training, infer_start), 81)


u_inf_n = predict_n_ode()[end]

l_fulld = length( hcat(training, infer_start)[2,:] )

x = collect(l_fulld - 43:1:l_fulld)

plot(hcat(training, infer_start)[2,:], color="blue", label="Real data")
xlabel!("Days")
ylabel!("Value")
vline!([107],ls=:dash,c=:black, label="")
plot!(x, n_ode(u_inf .+ 0.3)[:,2:45][2,:], color="red", label="Projection")
plot!(x, n_ode(u_inf .+ 0.5)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .+ 0.6)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .+ 0.7)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .+ 0.2)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .- 0.2)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .- 0.3)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .- 1)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .- 0.5)[:,2:45][2,:], color="red", label="")
plot!(x, n_ode(u_inf .- 0.7)[:,2:45][2,:], color="red", label="")





plot(trainData[2,:],legend=:topleft, color="blue", label="Real data")
xlabel!("Days")
ylabel!("Value")
plot!(n_ode(u0)[2,:], color="red", label="Projection")
plot!(n_ode(u0 .+ 0.1)[2,:], color="red", label="")
plot!(n_ode(u0 .+ 0.15)[2,:], color="red", label="")
plot!(n_ode(u0 .+ 0.2)[2,:], color="red", label="")
plot!(n_ode(u0 .+ 0.3)[2,:], color="red", label="")
plot!(n_ode(u0 .- 0.1)[2,:], color="red", label="")
plot!(n_ode(u0 .- 0.15)[2,:], color="red", label="")
plot!(n_ode(u0 .- 0.2)[2,:], color="red", label="")
plot!(n_ode(u0 .- 0.3)[2,:], color="red", label="")
