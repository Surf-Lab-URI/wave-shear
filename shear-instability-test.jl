using Pkg; Pkg.activate(@__DIR__)
using Oceananigans, CUDA
using Random

Random.seed!(404)

# setting up grid
const Nx = 768
const Ny = 768
const Nz = 512 #round(Int, Ny/2)

const Ly = 0.1
const Lx = 0.1
const Lz = Ly/2

const refinement = 1.5 # controls spacing near surface (higher means finer spaced)
const stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

grid = RectilinearGrid(GPU(); size=(Nx, Ny, Nz), 
                                x = (0, Lx),
                                y = (0, Ly),
                                z = k -> Lz * (ζ₀(k) * Σ(k) - 1), 
                                topology = (Periodic, Periodic, Bounded))
H(x) = x < 0 ? 0.0 : 1.0
const tp = 7
uₛ(x,y,t) = 0.1 + H(t-tp)*0.01*sin(2*2π/Lx*x-0.5t)*sin(2*2π/Ly*y)*(1-exp(-(t-tp)))
u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(uₛ), bottom = ValueBoundaryCondition(0))
boundary_conditions = (u = u_bcs,);

model = NonhydrostaticModel(grid;
                            boundary_conditions,
                            advection = UpwindBiased(order=5), 
                            closure = ScalarDiffusivity(ν=1e-6))



using Statistics
u,v,w = model.velocities

# uᵢ = rand(size(u)...)
# vᵢ = rand(size(v)...)

# uᵢ .-= mean(uᵢ)
# vᵢ .-= mean(vᵢ)

uᵢ = zeros(size(u)...)
vᵢ = zeros(size(v)...)

set!(model, u=uᵢ, v=vᵢ)

cfl_safety = 0.7
sₘ = 0.15
tcfl = cfl_safety*grid.Δxᶠᵃᵃ/sₘ          # [physics].cfl_safety

simulation = Simulation(model, Δt=tcfl, stop_time=50)

conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_change=1.1, max_Δt=0.5)



#Logging
using Printf
function progress_message(sim)
    u = sim.model.velocities.u
    @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s",
                   iteration(sim), time(sim), sim.Δt, maximum(abs, u), prettytime(sim.run_wall_time))
    return nothing
end

add_callback!(simulation, progress_message, IterationInterval(100))


#Output 
ω = ∂x(v) - ∂y(u)

s = sqrt(u^2 + v^2)

filename = "shear-instability-test-p1.jld2"

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, s, u); filename,
                                                schedule = TimeInterval(tcfl*100),
                                                overwrite_files = true)


#Run Simulation
run!(simulation)