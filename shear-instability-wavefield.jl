using Pkg; Pkg.activate(@__DIR__)
using ArgParse

# Like shear-instability-test.jl, but the surface velocity perturbation is a random field of capillary-gravity
# waves (the technique from plotting/surface-velocity-test.jl) instead of a single sin(kx x) sin(ky y) mode:
#
#   u_surface(x, y, t) = U + ramp(t) ⋅ Σᵢ aᵢ cos(kxᵢ x + kyᵢ y - ωᵢ t - ϕᵢ)
#
# Each wave has a random amplitude aᵢ and phase ϕᵢ, drawn from --seed, and travels at the capillary-gravity
# frequency ωᵢ = √(g|k| + (σ/ρ)|k|³) + kxᵢ U_doppler. The amplitudes are scaled together so that the 90th
# percentile of |u′|, the size of the perturbation, is --amplitude.

# Command line arguments (run with --help to list them)
function parse_commandline()
    s = ArgParseSettings(description = "Viscous shear flow driven by a surface velocity perturbed by a random field of capillary-gravity waves",
                         autofix_names = true) # e.g. --stop-time becomes args[:stop_time]

    @add_arg_table! s begin
        "--U"
            help = "mean surface velocity (m/s)"
            arg_type = Float64
            default = 0.15
        "--amplitude"
            help = "size of the surface velocity perturbation u′ (m/s): the 90th percentile of |u′| over the surface and over time"
            arg_type = Float64
            default = 0.02
        "--kx-min"
            help = "smallest x mode number of the waves (dimensionless): the wavenumber is kx⋅2π/Lx rad/m, with Lx = 0.1 m"
            arg_type = Int
            default = 0
        "--kx-max"
            help = "largest x mode number of the waves"
            arg_type = Int
            default = 5
        "--ky-max"
            help = "the waves' y mode numbers run from -ky-max to ky-max (dimensionless): the wavenumber is ky⋅2π/Ly rad/m, with Ly = 0.1 m"
            arg_type = Int
            default = 1
        "--U-doppler"
            help = "current that Doppler-shifts the waves' frequencies, ω = √(g|k| + (σ/ρ)|k|³) + kx⋅U_doppler (m/s)"
            arg_type = Float64
            default = 0.1
        "--seed"
            help = "random seed for the waves' amplitudes and phases: the same seed and modes give the same surface velocity (0 picks a new seed, which is logged and saved)"
            arg_type = Int
            default = 1
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
            default = 30.0
        "--output-interval"
            help = "time between saved snapshots (s), used when --snapshots-per-period is 0"
            arg_type = Float64
            default = 0.06
        "--snapshots-per-period"
            help = "if > 0, save this many snapshots per period of the fastest wave (the shortest period) instead of using --output-interval"
            arg_type = Int
            default = 8
        "--output"
            help = "output file prefix: writes <output>.jld2, or <output>_slices.jld2 with --slices-only"
            default = "shear-instability-wavefield"
        "--slices-only"
            help = "only save u on the top, east and north faces (the faces the plotting script uses)"
            action = :store_true
    end

    return parse_args(s; as_symbols = true)
end

const args = parse_commandline()

@info "Parameters:\n" * join(["  $k = $v" for (k, v) in sort(collect(args), by=first)], "\n")
flush(stderr)

0 ≤ args[:kx_min] ≤ args[:kx_max] || error("--kx-min and --kx-max need 0 ≤ kx-min ≤ kx-max")
args[:ky_max] ≥ 0 || error("--ky-max can't be negative")
args[:kx_max] > 0 || args[:ky_max] > 0 || error("there are no waves: kx-max and ky-max are both 0")

# Loading CUDA on a node without a GPU can precompile CUDA.jl's libraries without GPU support,
# which then breaks later runs on GPU nodes, so stop before that can happen
has_gpu = try success(pipeline(`nvidia-smi -L`, devnull)) catch; false end
has_gpu || error("No GPU found. Run this script on a GPU node, e.g. through Slurm with --gpus=1")

using Oceananigans, CUDA
using Random, StableRNGs, Statistics, Printf

# setting up grid
const Nx = args[:Nx]
const Ny = args[:Ny]
const Nz = args[:Nz]

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

#### The random wave field

const g = 9.81                       # gravity (m/s²)
const surface_tension = 0.073 / 1000 # σ/ρ for clean water (m³/s²)

# Deep-water capillary-gravity waves with wavenumber magnitude κ (rad/m)
intrinsic_frequency(κ) = sqrt(g * κ + surface_tension * κ^3) # rad/s

# Every mode (kx, ky) in the chosen ranges except (0, 0), which would only shift the mean surface velocity
mode_numbers = [(i, j) for i in args[:kx_min]:args[:kx_max] for j in -args[:ky_max]:args[:ky_max] if (i, j) != (0, 0)]
kx_modes = first.(mode_numbers)
ky_modes = last.(mode_numbers)
kxs = kx_modes .* 2π/Lx                                                # rad/m
kys = ky_modes .* 2π/Ly                                                # rad/m
frequencies = intrinsic_frequency.(hypot.(kxs, kys)) .+ kxs .* args[:U_doppler] # rad/s

# StableRNG gives the same numbers for a seed on every Julia version, unlike Julia's default generator
seed = args[:seed] == 0 ? rand(RandomDevice(), 1:typemax(Int32)) : args[:seed]
rng = StableRNG(seed)
relative_amplitudes = rand(rng, Float64, length(mode_numbers)) # uniform in [0, 1), as in surface-velocity-test.jl
phases = 2π .* rand(rng, Float64, length(mode_numbers))        # rad

# Sum of the waves, u′, at the surface point (x, y) and time t. The waves are stored as tuples, which the GPU can use
@inline function wave_perturbation(x, y, t, p)
    u′ = zero(x)
    @inbounds for i in eachindex(p.amplitude)
        u′ += p.amplitude[i] * cos(p.kx[i] * x + p.ky[i] * y - p.frequency[i] * t - p.phase[i])
    end
    return u′
end

waves(amplitudes) = (amplitude = Tuple(amplitudes), kx = Tuple(kxs), ky = Tuple(kys),
                     frequency = Tuple(frequencies), phase = Tuple(phases))

# Scale the amplitudes so the 90th percentile of |u′| is --amplitude. u′ is sampled on a fixed 128 × 128 grid of
# the surface at 200 times over 60 s, so the scale depends only on the waves, not on the resolution or stop time
x_sample = ((1:128) .- 0.5) .* (Lx/128)
y_sample = ((1:128) .- 0.5) .* (Ly/128)
t_sample = range(0, 60, length = 200)
relative_waves = waves(relative_amplitudes)
relative_u′ = [abs(wave_perturbation(x, y, t, relative_waves)) for x in x_sample, y in y_sample, t in t_sample]
amplitude_scale = args[:amplitude] / quantile(vec(relative_u′), 0.9) # m/s
amplitudes = relative_amplitudes .* amplitude_scale                  # m/s
max_perturbation = maximum(relative_u′) * amplitude_scale            # m/s, largest |u′| in the sample

@info "Random wave field: seed $seed, $(length(mode_numbers)) waves, max |u′| ≈ $(@sprintf("%.4f", max_perturbation)) m/s in the sample\n" *
      "   kx  ky   kx (rad/m)  ky (rad/m)  wavelength (m)  amplitude (m/s)  phase (rad)  ω (rad/s)  period (s)\n" *
      join([@sprintf("  %3d %3d   %10.2f  %10.2f  %14.5f  %15.5f  %11.3f  %9.2f  %10.5f",
                     kx_modes[i], ky_modes[i], kxs[i], kys[i], 2π / hypot(kxs[i], kys[i]),
                     amplitudes[i], phases[i], frequencies[i], 2π / abs(frequencies[i])) for i in eachindex(mode_numbers)], "\n")
flush(stderr)

# Surface velocity: U plus the waves, which ramp up after t = tp.
# max(t - tp, 0) keeps the ramp at zero before tp (and avoids exp overflow for large tp)
surface_u(x, y, t, p) = p.U + wave_perturbation(x, y, t, p) * (1 - exp(-max(t - p.tp, 0)))

surface_parameters = merge((U = args[:U],    # m/s
                            tp = args[:tp]), # s
                           waves(amplitudes))

# Derived quantities, with units in their names. Logged here and saved to the output file under "derived/".
# shortest_period_s is the period of the fastest wave, 2π / max|ω|
shortest_period_s = 2π / maximum(abs, frequencies)
output_interval_s = args[:snapshots_per_period] > 0 ? shortest_period_s / args[:snapshots_per_period] : args[:output_interval]

derived = (Lx_m = Lx, Ly_m = Ly, Lz_m = Lz,
           seed, modes = length(mode_numbers),
           amplitude_scale_m_per_s = amplitude_scale, max_perturbation_m_per_s = max_perturbation,
           shortest_period_s, output_interval_s)

@info "Derived quantities:\n" * join(["  $k = $v" for (k, v) in pairs(derived)], "\n")
flush(stderr)

# Every wave, saved to the output file under "surface/", so the surface velocity can be rebuilt without the seed
surface_table = (kx_mode = kx_modes, ky_mode = ky_modes, kx_rad_per_m = kxs, ky_rad_per_m = kys,
                 amplitude_m_per_s = amplitudes, phase_rad = phases, frequency_rad_per_s = frequencies,
                 g_m_per_s2 = g, surface_tension_m3_per_s2 = surface_tension)

# The commit this run used, saved to the output file under "code/". git_dirty = true means the
# scripts had uncommitted changes, so the commit alone doesn't reproduce the run
git(cmd...) = try readchomp(`git -C $(@__DIR__) $cmd`) catch; "unknown" end
code = (script     = basename(@__FILE__),
        git_commit = git("rev-parse", "HEAD"),
        git_dirty  = !isempty(git("status", "--porcelain", "--untracked-files=no")))

@info "Code: $(code.script) at commit $(code.git_commit)" * (code.git_dirty ? " with uncommitted changes" : "")
flush(stderr)

u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(surface_u, parameters=surface_parameters), bottom = ValueBoundaryCondition(0))
boundary_conditions = (u = u_bcs,);

model = NonhydrostaticModel(grid;
                            boundary_conditions,
                            advection = UpwindBiased(order=5),
                            closure = ScalarDiffusivity(ν=args[:nu]))

@info "Built model"
flush(stderr)

u,v,w = model.velocities

set!(model, u=zeros(size(u)...), v=zeros(size(v)...))

cfl_safety = 0.7
sₘ = 1.5 * (args[:U] + max_perturbation) # estimate of the max speed, for the initial time step
tcfl = cfl_safety*grid.Δxᶠᵃᵃ/sₘ

simulation = Simulation(model, Δt=tcfl, stop_time=args[:stop_time])

# keep at least 20 time steps per period of the fastest wave
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.7, max_change=1.1, max_Δt=min(0.5, shortest_period_s/20))


#Logging
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
    ω = ∂x(v) - ∂y(u) # vertical vorticity

    s = sqrt(u^2 + v^2)

    outputs = (; ω, s, u)

    filename = args[:output] * ".jld2"
end

# Store the command line arguments under "parameters/", the derived quantities under "derived/",
# the waves under "surface/" and the script and git commit under "code/"
function save_parameters(file, model)
    for (group, values) in (("parameters", args), ("derived", pairs(derived)),
                            ("surface", pairs(surface_table)), ("code", pairs(code)))
        for (k, v) in values
            file["$group/$k"] = v
        end
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
