using Pkg; Pkg.activate(@__DIR__)
using Oceananigans, CairoMakie, JLD2

# Usage: julia shear-instability-test-plotting.jl [output.jld2] [movie.mp4]
# Works with either full-field output (<output>.jld2) or --slices-only output (<output>_slices.jld2)
filename = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "..", "shear-instability-test-p1.jld2")
moviename = length(ARGS) ≥ 2 ? ARGS[2] : "3D-" * replace(basename(filename), ".jld2" => ".mp4")

slices = jldopen(file -> haskey(file, "timeseries/u_top"), filename)

@info "Reading $(slices ? "face slices" : "full fields") from $filename"
flush(stderr)

# Load u on the top (z = z[end]), east (x = x[end]) and north (y = y[end]) faces, as
#   u_top[i, j, n]   at (x[i], y[j], z_top)
#   u_east[j, k, n]  at (x_east, y[j], z[k])
#   u_north[i, k, n] at (x[i], y_north, z[k])
if slices
    top_timeseries   = FieldTimeSeries(filename, "u_top")
    east_timeseries  = FieldTimeSeries(filename, "u_east")
    north_timeseries = FieldTimeSeries(filename, "u_north")

    @info "set up FieldTimeSeries"
    flush(stderr)

    times = top_timeseries.times

    # Each slice knows where it is, so take the face coordinates from the slices themselves
    # (nodes of a single snapshot respect the slice indices; nodes of the whole FieldTimeSeries don't)
    x, y, z_top = nodes(top_timeseries[1])
    x_east, _, z = nodes(east_timeseries[1])
    _, y_north, _ = nodes(north_timeseries[1])
    z_top, x_east, y_north = only(z_top), only(x_east), only(y_north)

    u_top   = interior(top_timeseries)[:, :, 1, :]
    u_east  = interior(east_timeseries)[1, :, :, :]
    u_north = interior(north_timeseries)[:, 1, :, :]
else
    u_timeseries = FieldTimeSeries(filename, "u", backend = OnDisk())

    @info "set up FieldTimeSeries"
    flush(stderr)

    times = u_timeseries.times

    x, y, z = nodes(u_timeseries)
    z_top, x_east, y_north = z[end], x[end], y[end]

    Nt = length(times)

    u_top   = zeros(length(x), length(y), Nt)
    u_east  = zeros(length(y), length(z), Nt)
    u_north = zeros(length(x), length(z), Nt)

    @info "Loading data for plot ..."
    flush(stderr)
    for n in 1:Nt
        uₙ = interior(u_timeseries[n])           # reads one snapshot
        u_top[:, :, n]   .= uₙ[:, :, end]
        u_east[:, :, n]  .= uₙ[end, :, :]
        u_north[:, :, n] .= uₙ[:, end, :]
        GC.gc()
        @info "frame $n"
        flush(stderr)
    end
end

@info "Loaded $(length(times)) snapshots from t = $(first(times)) to $(last(times))"
flush(stderr)

#set up animation
set_theme!(Theme(fontsize = 20))



# Make 3D Plots
Nx, Ny, Nz = length(x), length(y), length(z)

# East face
x_east_face = fill(x_east, Ny, Nz)
y_east_face = repeat(y, 1, Nz)
z_east_face = repeat(z', Ny, 1)

# North face
x_north_face = repeat(x, 1, Nz)
y_north_face = fill(y_north, Nx, Nz)
z_north_face = repeat(z', Nx, 1)

# Top face
z_top_face = fill(z_top, Nx, Ny)

frames = 1:length(times)

n = Observable(1)
# set up 3D Axes. The camera sits at (cos(elevation)cos(azimuth), cos(elevation)sin(azimuth), sin(elevation)),
# i.e. on the +x, +y, +z side for these angles, so the east, north and top faces are the ones in view
fig = Figure(size = (800, 500))
ax = Axis3(fig[1,1]; aspect = :data, azimuth = 0.5, elevation = 0.45)

kw = (colormap = :speed, colorrange = (0,0.15), shading = NoShading)
surface!(ax, x_east_face, y_east_face, z_east_face; color = @lift(u_east[:, :, $n]), kw...)
surface!(ax, x_north_face, y_north_face, z_north_face; color = @lift(u_north[:, :, $n]), kw...)
surface!(ax, x, y, z_top_face; color = @lift(u_top[:, :, $n]), kw...)

fig

# Record animation

@info "Making 3D animation of u, saving to $moviename ..."
flush(stderr)

record(fig, moviename, frames, framerate=24) do i
    n[] = i
    @info "i = $i"
    flush(stderr)
end
