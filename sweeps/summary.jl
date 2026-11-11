using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "plotting"))
using JLD2, Printf

# Usage: julia sweeps/summary.jl [directory]
# Collects the parameters of every run under the directory into one table, printed and saved as
# <directory>/summary.csv. The default directory, runs/, covers every sweep; pass runs/<sweep> for one sweep.
# Units are in the column names, and columns that don't apply to any of the runs are left out.
#
# The script column says which simulation made the run, and the amplitude means something different for each:
#   shear-instability-test.jl       the perturbation is amplitude ⋅ sin(kx x) sin(ky y), where kx_mode and ky_mode
#                                   are the mode numbers passed to --kx and --ky: kx_rad_per_m = kx_mode⋅2π/Lx
#   shear-instability-wavefield.jl  random waves with x modes kx_mode_min to kx_mode_max and y modes -ky_mode_max
#                                   to ky_mode_max, scaled so the 90th percentile of |u′| is the amplitude
root = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "..", "runs")

# column name => where it's stored in the output file
stored = ["U_m_per_s"         => "parameters/U",
          "amplitude_m_per_s" => "parameters/amplitude",
          # shear-instability-test.jl
          "kx_mode"           => "parameters/kx",
          "ky_mode"           => "parameters/ky",
          "kx_rad_per_m"      => "derived/kx_rad_per_m",
          "ky_rad_per_m"      => "derived/ky_rad_per_m",
          "wavelength_x_m"    => "derived/wavelength_x_m",
          "wavelength_y_m"    => "derived/wavelength_y_m",
          "c_m_per_s"         => "parameters/c",
          "period_s"          => "derived/period_s",
          # shear-instability-wavefield.jl
          "kx_mode_min"       => "parameters/kx_min",
          "kx_mode_max"       => "parameters/kx_max",
          "ky_mode_max"       => "parameters/ky_max",
          "U_doppler_m_per_s" => "parameters/U_doppler",
          "seed"              => "derived/seed",
          "waves"             => "derived/modes",
          "max_perturbation_m_per_s" => "derived/max_perturbation_m_per_s",
          "shortest_period_s" => "derived/shortest_period_s",
          # both
          "tp_s"              => "parameters/tp",
          "nu_m2_per_s"       => "parameters/nu",
          "stop_time_s"       => "parameters/stop_time",
          "output_interval_s" => "derived/output_interval_s",
          "Nx"                => "parameters/Nx",
          "Ny"                => "parameters/Ny",
          "Nz"                => "parameters/Nz",
          "git_commit"        => "code/git_commit",
          "git_dirty"         => "code/git_dirty"]

# A run is a directory with a simulation log or an output file
isrun(dir) = isfile(joinpath(dir, "simulation.log")) || any(endswith(".jld2"), readdir(dir))

function summarize(dir)
    name = basename(dir)
    row = Dict{String, Any}("name" => name,
                            "sweep" => relpath(dirname(dir), root),
                            "status" => isfile(joinpath(dir, "done")) ? "finished" : "running or failed")
    files = filter(endswith(".jld2"), readdir(dir, join=true))
    isempty(files) && return row
    filename = length(files) == 1 ? only(files) : first(filter(f -> startswith(basename(f), name), files))

    try
        jldopen(filename) do file
            for (column, key) in stored
                row[column] = haskey(file, key) ? file[key] : missing
            end
            # Runs from before the script was saved all used shear-instability-test.jl
            row["script"] = haskey(file, "code/script") ? file["code/script"] : "shear-instability-test.jl"
            iterations = keys(file["timeseries/t"])
            row["snapshots"] = length(iterations)
            row["last_t_s"] = file["timeseries/t/$(maximum(parse.(Int, iterations)))"]
        end
    catch err # e.g. a file that's being written
        @warn "Couldn't read $filename" err
    end
    return row
end

rundirs = sort([dir for (dir, _, _) in walkdir(root) if dir != root && isrun(dir)])
rows = summarize.(rundirs)

# Keep the columns that have a value for at least one run
used(column) = any(row -> !ismissing(get(row, column, missing)), rows)
columns = filter(used, ["name", "sweep", "status", "script", first.(stored)..., "snapshots", "last_t_s"])

csv = joinpath(root, "summary.csv")
open(csv, "w") do io
    println(io, join(columns, ","))
    for row in rows
        println(io, join([coalesce(get(row, c, missing), "") for c in columns], ","))
    end
end

# A narrower view for the terminal
show_columns = filter(in(columns), ["name", "sweep", "status", "script", "amplitude_m_per_s",
                                    "kx_mode", "ky_mode", "kx_rad_per_m", "c_m_per_s", "period_s",
                                    "kx_mode_min", "kx_mode_max", "ky_mode_max", "seed", "shortest_period_s",
                                    "snapshots", "last_t_s"])
format(x::AbstractFloat) = @sprintf("%.4g", x)
format(::Missing) = ""
format(x) = string(x)
table = [show_columns, ([format(get(row, c, missing)) for c in show_columns] for row in rows)...]
widths = [maximum(length(line[j]) for line in table) for j in eachindex(show_columns)]
for line in table
    println(join([rpad(line[j], widths[j]) for j in eachindex(line)], "  "))
end
println("\nAll columns saved to $(abspath(csv))")
