using Flux, CUDA
using DifferentialEquations
using Plots

f(u,p,t) = -15.0*u
fst(u,p,t) = -21 * u + ℯ ^ (-t)

u0 = 1.0
u0st = 0.0

tspan = (0.0,1.0)
h = 0.125

x = tspan[1]:h:tspan[2]

xt = tspan[1]:h/10:tspan[2]

ex_sol = ℯ .^ (-15 * xt)



prob = ODEProblem(f,u0,tspan)
probst = ODEProblem(fst,u0st,tspan)


sol = solve(prob, Euler(), dt=h, saveat=0.01)
plot(ex_sol)
plot!(sol.u)

sol = solve(prob, Euler(), dt=0.01, saveat=0.01)
plot(ex_sol)
plot!(sol.u)

sol = solve(prob, RK4(), dt=h, saveat=0.01)
plot(ex_sol)
plot!(sol.u)

sol = solve(prob, AB3(), dt=h, saveat=0.01)
plot(ex_sol)
plot!(sol.u)


solst = solve(probst, ImplicitEuler(), dt=h, saveat=0.01)
plot(solst)


solst = solve(probst, Euler(), dt=h, saveat=0.01)
plot(solst)


function f(x)
    x + 5

    x + 3
end

f(5)

function linear(in, out)
  W = randn(out, in)
  b = randn(out)
  x -> W * x .+ b
end

linear1 = linear(5, 3)

linear2 = linear(3, 2)


using Flux

f(x) = 3x^2 + 2x + 1;

df(x) = gradient(f, x); # df/dx = 6x + 2

df(2)
