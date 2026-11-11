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
            help = "perturbation mode number in x (dimensionless): the wavenumber is kx⋅2π/Lx rad/m, with Lx = 0.1 m"
            arg_type = Int
            default = 2
        "--ky"
            help = "perturbation mode number in y (dimensionless): the wavenumber is ky⋅2π/Ly rad/m, with Ly = 0.1 m (0 means no perturbation)"
            arg_type = Int
            default = 2
        "--c"
            help = "phase speed of the perturbation in +x (m/s)"
            arg_type = Float64
            default = 0.5 / (2 * 2π / 0.1) # ≈ 0.00398 m/s, reproduces the earlier runs' sin(2⋅2π/Lx⋅x - 0.5t)
        "--tp"
            help = "perturbation onset time (s)"
            arg_type = Float64
            default = 7.0
        "--nu"
            help = "kinematic viscosity (m²/s)"
            arg_type = Float64
            default = 1e-6
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
        "--snapshots-per-period"
            help = "if > 0, save this many snapshots per perturbation period instead of using --output-interval"
            arg_type = Int
            default = 0
        "--output"
            help = "output file prefix: writes <output>.jld2, or <output>_slices.jld2 with --slices-only"
            default = "shear-instability-test-p1"
        "--slices-only"
            help = "only save u on the top, east and north faces (the faces the plotting script uses)"
            action = :store_true
    end

    return parse_args(s; as_symbols = true)
end

const args = parse_commandline()

@info "Parameters:\n" * join(["  $k = $v" for (k, v) in sort(collect(args), by=first)], "\n")
flush(stderr)

# Loading CUDA on a node without a GPU can precompile CUDA.jl's libraries without GPU support,
# which then breaks later runs on GPU nodes, so stop before that can happen
has_gpu = try success(pipeline(`nvidia-smi -L`, devnull)) catch; false end
has_gpu || error("No GPU found. Run this script on a GPU node, e.g. through Slurm with --gpus=1")

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

grid = RectilinearGrid(GPU(); size=(Nx, Ny, Nz),
                                x = (0, Lx),
                                y = (0, Ly),
                                z = k -> Lz * (ζ₀(k) * Σ(k) - 1),
                                topology = (Periodic, Periodic, Bounded))

@info "Built grid:\n$grid"
flush(stderr)

# Surface velocity: U plus a perturbation travelling in +x at phase speed c that ramps up after t = tp.
# max(t - tp, 0) keeps the ramp at zero before tp (and avoids exp overflow for large tp)
uₛ(x, y, t, p) = p.U + p.A * sin(p.kx*(x - p.c*t)) * sin(p.ky*y) * (1 - exp(-max(t - p.tp, 0)))

surface_parameters = (U  = args[:U],            # m/s
                      A  = args[:amplitude],    # m/s
                      kx = args[:kx] * 2π/Lx,   # rad/m
                      ky = args[:ky] * 2π/Ly,   # rad/m
                      c  = args[:c],            # m/s
                      tp = args[:tp])           # s

# Derived quantities, with units in their names. Logged here and saved to the output file under "derived/".
# period_s is the time for one x wavelength to pass a fixed point, wavelength_x_m / |c|
period_s = Lx / args[:kx] / abs(args[:c])
output_interval_s = args[:snapshots_per_period] > 0 ? period_s / args[:snapshots_per_period] : args[:output_interval]

derived = (Lx_m = Lx, Ly_m = Ly, Lz_m = Lz,
           kx_rad_per_m = surface_parameters.kx, ky_rad_per_m = surface_parameters.ky,
           wavelength_x_m = Lx / args[:kx], wavelength_y_m = Ly / args[:ky],
           period_s, output_interval_s)

@info "Derived quantities:\n" * join(["  $k = $v" for (k, v) in pairs(derived)], "\n")
flush(stderr)

# The commit this run used, saved to the output file under "code/". git_dirty = true means the
# script had uncommitted changes, so the commit alone doesn't reproduce the run
git(cmd...) = try readchomp(`git -C $(@__DIR__) $cmd`) catch; "unknown" end
code = (script     = basename(@__FILE__),
        git_commit = git("rev-parse", "HEAD"),
        git_dirty  = !isempty(git("status", "--porcelain", "--untracked-files=no")))

@info "Code: $(code.script) at commit $(code.git_commit)" * (code.git_dirty ? " with uncommitted changes" : "")
flush(stderr)

u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(uₛ, parameters=surface_parameters), bottom = ValueBoundaryCondition(0))
boundary_conditions = (u = u_bcs,);

model = NonhydrostaticModel(grid;
                            boundary_conditions,
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=args[:nu]))

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

# keep at least 20 time steps per perturbation period (period_s is Inf when c = 0)
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_change=1.1, max_Δt=min(0.5, derived.period_s/20))



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

# Store the command line arguments under "parameters/", the derived quantities under "derived/"
# and the git commit under "code/"
function save_parameters(file, model)
    for (k, v) in args
        file["parameters/$k"] = v
    end
    for (k, v) in pairs(derived)
        file["derived/$k"] = v
    end
    for (k, v) in pairs(code)
        file["code/$k"] = v
    end
end

simulation.output_writers[:fields] = JLD2Writer(model, outputs; filename,
                                                schedule = TimeInterval(output_interval_s),
                                                init = save_parameters,
                                                overwrite_files = true)

@info "Saving $(join(keys(outputs), ", ")) every $output_interval_s s to $(abspath(filename))"
flush(stderr)


#Run Simulation
run!(simulation)

@info "Simulation finished at t = $(time(simulation)) after $(prettytime(simulation.run_wall_time))"
flush(stderr)
