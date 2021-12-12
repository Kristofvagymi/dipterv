
using DifferentialEquations, DiffEqFlux, Flux, Optim
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
  Dense(6,24,swish),
  Dense(24,6, tanh)) |> gpu

function predict_n_ode()
  n_ode(u0)
end

cb = function () #callback function to observe training
  display(loss_n_ode())
end


function loss_n_ode()
  pred = hcat(predict_n_ode().u...)

  len = length(pred[1,:])

  true_d = trainData[:,1:len]
  sum(abs2, pred .- true_d)
end

u0 = trainData[:,1] |> gpu
trainData = trainData |>gpu

tspan = (0.0f0,25.0f0)
t = range(tspan[1],tspan[2],length=length( trainData[1,:]))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t[t .<= 8.0], abstol=1e-2,reltol=1e-2)

ps = Flux.params(n_ode)
data = Iterators.repeated((), 128)
opt = ADAM(0.05)

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)


plot(transpose(trainData))
plot!(transpose(predict_n_ode()))

t

plotall = function (df1, df2, dash)
  p1 = plot(df1[1,:])
  p1 = plot!(df2[1,:])
  p1 = vline!([dash],ls=:dash,c=:black)

  p2 = plot(df1[2,:])
  p2 = plot!(df2[2,:])
  p2 = vline!([dash],ls=:dash,c=:black)

  p3 = plot(df1[3,:])
  p3 = plot!(df2[3,:])
  p3 = vline!([dash],ls=:dash,c=:black)

  p4 = plot(df1[4,:])
  p4 = plot!(df2[4,:])
  p4 = vline!([dash],ls=:dash,c=:black)

  p5 = plot(df1[5,:])
  p5 = plot!(df2[5,:])
  p5 = vline!([dash],ls=:dash,c=:black)

  p6 = plot(df1[6,:])
  p6 = plot!(df2[6,:])
  p6 = vline!([dash],ls=:dash,c=:black)

  plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), legend = false)
end
plotall(trainData, predict_n_ode(),0)


#train_loader = Flux.Data.DataLoader(ode_data, t, batchsize = k)
#trainData


u_inf = predict_n_ode()[end]

training = predict_n_ode() |> cpu
infer = n_ode(u_inf) |> cpu

infer_start = infer[:,2:45]

real_data = disallowmissing(transpose(Matrix(df_input[186:336, :])))
real_data = disallowmissing(transpose(Matrix(df_input[160:285, :])))



plotall(real_data, hcat(training, infer_start), 106)
