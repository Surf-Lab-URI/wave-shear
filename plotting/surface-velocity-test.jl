using Pkg; Pkg.activate(@__DIR__)
using CairoMakie

Ly = 0.1
Lx = 0.1

M = 768
N = 768
mmax = 3
nmax = 6

a = rand(mmax,nmax)/nmax
k(n) = 2π*(n-1)/Lx
l(m) = 2π*(m-1)/Ly
ϕ = rand(mmax,nmax)*2*π
g = 9.81
T = 0.073/1000
Ud = 0.1
ω(k,l,U) = sqrt(g*sqrt(k^2+l^2) + T*sqrt(k^2+l^2)^3) + k*U

uₛ(x,y,t) = sum(a[m,n]*cos(k(n)*x + l(m-floor(mmax/2+1))*y-ϕ[m,n]-ω(k(n),l(n),Ud)*t) for m in 1:mmax for n in 1:nmax)

x = collect(LinRange(0,Lx,N))
y = collect(LinRange(0,Ly,M))

X = x .+ 0 .* y'
Y = 0 .* x .+ y'

n = Observable(1)
t = collect(0:0.01:0.25)

fig = Figure(size = (800, 500))
ax = Axis(fig[1, 1], aspect = DataAspect(), title = "Surface Velocity", xlabel = "X", ylabel = "Y")
heatmap!(ax,@lift(uₛ.(X,Y,t[$n])),colorrange = (-1, 1))
Colorbar(fig[1,2], colorrange = (-1, 1),label = "u (m/s)")
fig

frames = 1:length(t)
moviename = "us-test.mp4"
@info "Making 3D animation of u, saving to $moviename ..."
flush(stderr)
record(fig, moviename, frames, framerate=1/0.02) do i
    n[] = i
    @info "i = $i"
    flush(stderr)
end