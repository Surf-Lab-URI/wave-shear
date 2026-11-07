using Pkg; Pkg.activate(@__DIR__)
using ArgParse

# Command line arguments (run with --help to list them)
function parse_commandline()
    s = ArgParseSettings(description = "Viscous shear flow driven by a perturbed surface velocity",
                         autofix_names = true) # e.g. --stop-time becomes args[:stop_time]

    @add_arg_table! s begin
        "--U"
            help = "mean surface velocity (m/s)"
            arg_type = Float64
            default = 0.1
        "--amplitude"
            help = "amplitude of the surface velocity perturbation (m/s)"
            arg_type = Float64
            default = 0.01
        "--kx"
            help = "perturbation wavenumber in x, as a multiple of 2π/Lx"
            arg_type = Int
            default = 2
        "--ky"
            help = "perturbation wavenumber in y, as a multiple of 2π/Ly (0 means no perturbation)"
            arg_type = Int
            default = 2
        "--tp"
            help = "perturbation onset time (s)"
            arg_type = Float64
            default = 7.0
        "--Nx"
            arg_type = Int
            default = 768
        "--Ny"
            arg_type = Int
            default = 768
        "--Nz"
            arg_type = Int
            default = 512
        "--stop-time"
            help = "simulation stop time (s)"
            arg_type = Float64
            default = 50.0
        "--output-interval"
            help = "time between saved snapshots (s)"
            arg_type = Float64
            default = 0.06
        "--output"
            help = "output file prefix: writes <output>.jld2, or <output>_slices.jld2 with --slices-only"
            default = "shear-instability-test-p1"
        "--slices-only"
            help = "only save u on the top, east and north faces (the faces the plotting script uses)"
            action = :store_true
        "--cpu"
            help = "run on the CPU instead of the GPU (for small tests)"
            action = :store_true
    end

    return parse_args(s; as_symbols = true)
end

const args = parse_commandline()

@info "Parameters:\n" * join(["  $k = $v" for (k, v) in sort(collect(args), by=first)], "\n")
flush(stderr)

using Oceananigans, CUDA
using Random

Random.seed!(404)

# setting up grid
const Nx = args[:Nx]
const Ny = args[:Ny]
const Nz = args[:Nz] #round(Int, Ny/2)

const Ly = 0.1
const Lx = 0.1
const Lz = Ly/2

const refinement = 1.5 # controls spacing near surface (higher means finer spaced)
const stretching = 8   # controls rate of stretching at bottom
h(k) = (k - 1) / Nz
ζ₀(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

arch = args[:cpu] ? CPU() : GPU()

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz),
                                x = (0, Lx),
                                y = (0, Ly),
                                z = k -> Lz * (ζ₀(k) * Σ(k) - 1),
                                topology = (Periodic, Periodic, Bounded))

@info "Built grid:\n$grid"
flush(stderr)

# Surface velocity: U plus a travelling perturbation that ramps up after t = tp.
# max(t - tp, 0) keeps the ramp at zero before tp (and avoids exp overflow for large tp)
uₛ(x, y, t, p) = p.U + p.A * sin(p.kx*x - p.ω*t) * sin(p.ky*y) * (1 - exp(-max(t - p.tp, 0)))

surface_parameters = (U  = args[:U],
                      A  = args[:amplitude],
                      kx = args[:kx] * 2π/Lx,
                      ky = args[:ky] * 2π/Ly,
                      ω  = 0.5,
                      tp = args[:tp])

u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(uₛ, parameters=surface_parameters), bottom = ValueBoundaryCondition(0))
boundary_conditions = (u = u_bcs,);

model = NonhydrostaticModel(grid;
                            boundary_conditions,
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=1e-6))

@info "Built model"
flush(stderr)

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
sₘ = 1.5 * (args[:U] + args[:amplitude]) # estimate of the max speed, for the initial time step
tcfl = cfl_safety*grid.Δxᶠᵃᵃ/sₘ          # [physics].cfl_safety

simulation = Simulation(model, Δt=tcfl, stop_time=args[:stop_time])

conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_change=1.1, max_Δt=0.5)



#Logging
using Printf
function progress_message(sim)
    u = sim.model.velocities.u
    @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s",
                   iteration(sim), time(sim), sim.Δt, maximum(abs, u), prettytime(sim.run_wall_time))
    flush(stderr)
    return nothing
end

add_callback!(simulation, progress_message, IterationInterval(100))


#Output
if args[:slices_only]
    # Just the faces used by plotting/shear-instability-test-plotting.jl
    outputs = (u_top   = view(u, :, :, Nz:Nz),
               u_east  = view(u, Nx:Nx, :, :),
               u_north = view(u, :, Ny:Ny, :))

    filename = args[:output] * "_slices.jld2"
else
    ω = ∂x(v) - ∂y(u)

    s = sqrt(u^2 + v^2)

    outputs = (; ω, s, u)

    filename = args[:output] * ".jld2"
end

# Store the run parameters in the output file, under "parameters/"
function save_parameters(file, model)
    for (k, v) in args
        file["parameters/$k"] = v
    end
end

simulation.output_writers[:fields] = JLD2Writer(model, outputs; filename,
                                                schedule = TimeInterval(args[:output_interval]),
                                                init = save_parameters,
                                                overwrite_files = true)

@info "Saving $(join(keys(outputs), ", ")) every $(args[:output_interval]) s to $(abspath(filename))"
flush(stderr)


#Run Simulation
run!(simulation)

@info "Simulation finished at t = $(time(simulation)) after $(prettytime(simulation.run_wall_time))"
flush(stderr)
