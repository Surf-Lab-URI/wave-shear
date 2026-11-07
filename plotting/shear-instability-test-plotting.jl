using Pkg; Pkg.activate(@__DIR__)
using Oceananigans, CairoMakie

filename = joinpath(@__DIR__, "..", "shear-instability-test-p1.jld2")

ω_timeseries = FieldTimeSeries(filename, "ω", backend = OnDisk())
s_timeseries = FieldTimeSeries(filename, "s", backend = OnDisk())
u_timeseries = FieldTimeSeries(filename, "u", backend = OnDisk())

@info "set up FieldTimeSeries"
flush(stderr)


times = ω_timeseries.times

#set up animation
set_theme!(Theme(fontsize = 20))



# Make 3D Plots
x, y, z = nodes(u_timeseries)
Nx, Ny, Nz = length(x), length(y), length(z)

# East face
x_east = fill(x[end], Ny, Nz)
y_east = repeat(y, 1, Nz)
z_east = repeat(z', Ny, 1)

# North face
x_north = repeat(x, 1, Nz)
y_north = fill(y[end], Nx, Nz)
z_north = repeat(z', Nx, 1)

# Top face
z_top = fill(z[end], Nx, Ny)

# speed value for the faces
# s_east = @lift interior(u_timeseries[$n])[end,:,:]
# s_north = @lift interior(u_timeseries[$n])[:,end,:]
# s_top = @lift interior(u_timeseries[$n])[:,:,end]

frames = 1:length(times)

Nt = length(u_timeseries.times)

s_top   = zeros(length(x), length(y), Nt)
s_east  = zeros(length(y), length(z), Nt)
s_north = zeros(length(x), length(z), Nt)

@info "Loading data for plot ..."
flush(stderr)
for n in frames
    sₙ = interior(u_timeseries[n])           # reads one snapshot
    s_top[:, :, n]   .= sₙ[:, :, end]
    s_east[:, :, n]  .= sₙ[end, :, :]
    s_north[:, :, n] .= sₙ[:, end, :]
    GC.gc()
    @info "frame $n"
    flush(stderr)
end

n = Observable(1)
# set up 3D Axes
fig = Figure(size = (800, 500))
ax = Axis3(fig[1,1]; aspect = :data, azimuth = 0.5, elevation = 0.45)

kw = (colormap = :speed, colorrange = (0,0.15), shading = NoShading)
surface!(ax, x_east, y_east, z_east; color = @lift(s_east[:, :, $n]), kw...)
surface!(ax, x_north, y_north, z_north; color = @lift(s_north[:, :, $n]), kw...)
surface!(ax, x, y, z_top; color = @lift(s_top[:, :, $n]), kw...)

fig

# Record animation

@info "Making 3D animation of speed..."
flush(stderr)

record(fig, "3D-p1.mp4", frames, framerate=24) do i
    n[] = i
    @info "i = $i"
    flush(stderr)
end
