using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "plotting"))
using JLD2, Printf

# Usage: julia sweeps/summary.jl [runs directory]
# Collects the parameters of every run in runs/ into one table, printed and saved as runs/summary.csv.
# Units are in the column names. kx_mode and ky_mode are the dimensionless mode numbers passed to
# --kx and --ky; the wavenumbers are kx_rad_per_m = kx_mode⋅2π/Lx and ky_rad_per_m = ky_mode⋅2π/Ly.
runs = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "..", "runs")

# column name => where it's stored in the output file
stored = ["U_m_per_s"         => "parameters/U",
          "amplitude_m_per_s" => "parameters/amplitude",
          "kx_mode"           => "parameters/kx",
          "ky_mode"           => "parameters/ky",
          "kx_rad_per_m"      => "derived/kx_rad_per_m",
          "ky_rad_per_m"      => "derived/ky_rad_per_m",
          "wavelength_x_m"    => "derived/wavelength_x_m",
          "wavelength_y_m"    => "derived/wavelength_y_m",
          "c_m_per_s"         => "parameters/c",
          "period_s"          => "derived/period_s",
          "tp_s"              => "parameters/tp",
          "nu_m2_per_s"       => "parameters/nu",
          "stop_time_s"       => "parameters/stop_time",
          "output_interval_s" => "derived/output_interval_s",
          "Nx"                => "parameters/Nx",
          "Ny"                => "parameters/Ny",
          "Nz"                => "parameters/Nz",
          "git_commit"        => "code/git_commit",
          "git_dirty"         => "code/git_dirty"]

function summarize(dir)
    name = basename(dir)
    files = filter(endswith(".jld2"), readdir(dir, join=true))
    row = Dict{String, Any}("name" => name,
                            "status" => isfile(joinpath(dir, "done")) ? "finished" : "running or failed")
    isempty(files) && return row

    try
        jldopen(only(files)) do file
            for (column, key) in stored
                row[column] = haskey(file, key) ? file[key] : missing
            end
            iterations = keys(file["timeseries/t"])
            row["snapshots"] = length(iterations)
            row["last_t_s"] = file["timeseries/t/$(maximum(parse.(Int, iterations)))"]
        end
    catch err # e.g. a file that's being written
        @warn "Couldn't read $(only(files))" err
    end
    return row
end

rows = [summarize(dir) for dir in readdir(runs, join=true) if isdir(dir)]
columns = ["name", "status", first.(stored)..., "snapshots", "last_t_s"]

csv = joinpath(runs, "summary.csv")
open(csv, "w") do io
    println(io, join(columns, ","))
    for row in rows
        println(io, join([get(row, c, missing) for c in columns], ","))
    end
end

# A narrower view for the terminal
show_columns = ["name", "status", "amplitude_m_per_s", "kx_mode", "ky_mode", "kx_rad_per_m", "wavelength_x_m",
                "wavelength_y_m", "c_m_per_s", "period_s", "snapshots", "last_t_s"]
format(x::AbstractFloat) = @sprintf("%.4g", x)
format(x) = string(x)
table = [show_columns, ([format(get(row, c, missing)) for c in show_columns] for row in rows)...]
widths = [maximum(length(line[j]) for line in table) for j in eachindex(show_columns)]
for line in table
    println(join([rpad(line[j], widths[j]) for j in eachindex(line)], "  "))
end
println("\nAll columns saved to $(abspath(csv))")
