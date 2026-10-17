
# %% [markdown]
# # Design of a Symmetric Broadband Splitter
# 
# Many devices of interest can leverage some form of simulation symmetry to reduce the computational cost and storage requirements. The adjoint solver and its corresponding TO filter toolbox are built to work with these symmetries.
# 
# To demonstrate this, we look at a symmetric, broadband splitter.

# %%
import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa
from autograd import tensor_jacobian_product, grad
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import nlopt

seed = 240
np.random.seed(seed)
mp.quiet(quietval=True)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

# %% [markdown]
# As before, we'll define our geometry, filtering, and wavelength parameters.

# %%
waveguide_width = 0.5
design_region_width = 2.5
design_region_height = 2.5
arm_separation = 1.0
waveguide_length = 0.5
pml_size = 1.0
resolution = 20

minimum_length = 0.09
eta_e = 0.55
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
eta_i = 0.5
eta_d = 1 - eta_e
design_region_resolution = int(5 * resolution)

frequencies = 1 / np.linspace(1.5, 1.6, 10)
# print(1/frequencies)

# %% [markdown]
# We'll also define our simulation domain and set up a broadband source.

# %%
Sx = 2 * pml_size + 2 * waveguide_length + design_region_width
Sy = 2 * pml_size + design_region_height + 0.5
cell_size = mp.Vector3(Sx, Sy)

pml_layers = [mp.PML(pml_size)]

fcen = 1 / 1.56
width = 0.2
fwidth = width * fcen
source_center = [-Sx / 2 + pml_size + waveguide_length / 3, 0, 0]
source_size = mp.Vector3(0, 2, 0)
kpoint = mp.Vector3(1, 0, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [
    mp.EigenModeSource(
        src,
        eig_band=1,
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint,
        size=source_size,
        center=source_center,
    )
]

# %% [markdown]
# Next, we'll define our design region. This time, however, we'll include a symmetry across the Y plane (normal direction of the symmetry plane points in the Y direction).

# %%
Nx = int(design_region_resolution * design_region_width) + 1
Ny = int(design_region_resolution * design_region_height) + 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(design_region_width, design_region_height, 0),
    ),
)

# %% [markdown]
# We'll define a filtering and interpolation function. We need to include the symmetry requirements in the filter too.

# %%
def mapping(x, eta, beta):

    # filter
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        design_region_resolution,
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)

    projected_field = (
        npa.fliplr(projected_field) + projected_field
    ) / 2  # up-down symmetry

    # interpolate to actual materials
    return projected_field.flatten()

# %% [markdown]
# We also need to define a bitmask that describes the boundary conditions of the waveguide and cladding layers.

# %%
# Define spatial arrays used to generate bit masks
x_g = np.linspace(-design_region_width / 2, design_region_width / 2, Nx)
y_g = np.linspace(-design_region_height / 2, design_region_height / 2, Ny)
X_g, Y_g = np.meshgrid(x_g, y_g, sparse=True, indexing="ij")

# Define the core mask
left_wg_mask = (X_g == -design_region_width / 2) & (np.abs(Y_g) <= waveguide_width / 2)
top_right_wg_mask = (X_g == design_region_width / 2) & (
    np.abs(Y_g + arm_separation / 2) <= waveguide_width / 2
)
bottom_right_wg_mask = (X_g == design_region_width / 2) & (
    np.abs(Y_g - arm_separation / 2) <= waveguide_width / 2
)
Si_mask = left_wg_mask | top_right_wg_mask | bottom_right_wg_mask

# Define the cladding mask
border_mask = (
    (X_g == -design_region_width / 2)
    | (X_g == design_region_width / 2)
    | (Y_g == -design_region_height / 2)
    | (Y_g == design_region_height / 2)
)
SiO2_mask = border_mask.copy()
SiO2_mask[Si_mask] = False

# %% [markdown]
# Finally we can formulate our geometry and simulation object.

# %%
geometry = [
    mp.Block(
        center=mp.Vector3(x=-Sx / 4),
        material=Si,
        size=mp.Vector3(Sx / 2 + 1, waveguide_width, 0),
    ),  # left waveguide
    mp.Block(
        center=mp.Vector3(x=Sx / 4, y=arm_separation / 2),
        material=Si,
        size=mp.Vector3(Sx / 2 + 1, waveguide_width, 0),
    ),  # top right waveguide
    mp.Block(
        center=mp.Vector3(x=Sx / 4, y=-arm_separation / 2),
        material=Si,
        size=mp.Vector3(Sx / 2 + 1, waveguide_width, 0),
    ),  # bottom right waveguide
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    # mp.Block(center=design_region.center, size=design_region.size, material=design_variables, e2=mp.Vector3(y=-1))
    #
    # The commented line above impose symmetry by overlapping design region with the same design variable. However,
    # currently there is an issue of doing that; instead, we use an alternative approach to impose symmetry.
    # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    symmetries=[mp.Mirror(direction=mp.Y)],
    default_material=SiO2,
    resolution=resolution,
)

# %% [markdown]
# We can proceed to define our objective function, its corresponding arguments, and the optimization object.

# %%
mode = 1

TE0 = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        center=mp.Vector3(x=-Sx / 2 + pml_size + 2 * waveguide_length / 3),
        size=mp.Vector3(y=1.5),
    ),
    mode,
)
TE_top = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        center=mp.Vector3(
            Sx / 2 - pml_size - 2 * waveguide_length / 3, arm_separation / 2, 0
        ),
        size=mp.Vector3(y=arm_separation),
    ),
    mode,
)
TE_bottom = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        center=mp.Vector3(
            Sx / 2 - pml_size - 2 * waveguide_length / 3, -arm_separation / 2, 0
        ),
        size=mp.Vector3(y=arm_separation),
    ),
    mode,
)
ob_list = [TE0, TE_top, TE_bottom]


def J(source, top, bottom):
    power = npa.abs(top / source) ** 2 + npa.abs(bottom / source) ** 2
    return npa.mean(power)


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decay_by=1e-5,
)

# %% [markdown]
# Let's plot the design and ensure we have symmetry.

# %%
plt.figure()
x0 = mapping(
    np.random.rand(
        Nx * Ny,
    ),
    eta_i,
    128,
)
opt.update_design([x0])
opt.plot2D(True)
plt.show()

# %% [markdown]
# We'll define a simple objective function that returns the gradient. We'll plot the new geometry after each iteration.

# %%
evaluation_history = []
cur_iter = [0]


def f(v, gradient, cur_beta):
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])

    plt.figure()
    ax = plt.gca()
    opt.plot2D(
        False,
        ax=ax,
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=False,
    )
    circ = Circle((2, 2), minimum_length / 2)
    ax.add_patch(circ)
    ax.axis("off")
    plt.show()

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )

    evaluation_history.append(np.max(np.real(f0)))

    cur_iter[0] = cur_iter[0] + 1

    return np.real(f0)

# %% [markdown]
# Finally we'll run the optimizer.

# %%
algorithm = nlopt.LD_MMA
n = Nx * Ny  # number of parameters

# Initial guess
x = np.ones((n,)) * 0.5
x[Si_mask.flatten()] = 1  # set the edges of waveguides to silicon
x[SiO2_mask.flatten()] = 0  # set the other edges to SiO2

# lower and upper bounds
lb = np.zeros((Nx * Ny,))
lb[Si_mask.flatten()] = 1
ub = np.ones((Nx * Ny,))
ub[SiO2_mask.flatten()] = 0

cur_beta = 4
beta_scale = 2
num_betas = 6
update_factor = 12
for iters in range(num_betas):
    print("current beta: ", cur_beta)

    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale

# %% [markdown]
# We can see that the optimizer quickly finds a topology that works well and slowly refines it as we continue to "binarize" the design.

# %%
plt.figure()
plt.plot(10 * np.log10(0.5 * np.array(evaluation_history)), "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Mean Splitting Ratio (dB)")
plt.show()

# %% [markdown]
# We can view the final spectral response to verify that the design performs.

# %%
f0, dJ_du = opt([mapping(x, eta_i, cur_beta)], need_gradient=False)
frequencies = opt.frequencies
source_coef, top_coef, bottom_ceof = opt.get_objective_arguments()

top_profile = np.abs(top_coef / source_coef) ** 2
bottom_profile = np.abs(bottom_ceof / source_coef) ** 2

# %%
plt.figure()
plt.plot(1 / frequencies, top_profile * 100, "-o", label="Top Arm")
plt.plot(1 / frequencies, bottom_profile * 100, "--o", label="Bottom Arm")
plt.legend()
plt.grid(True)
plt.xlabel("Wavelength (microns)")
plt.ylabel("Splitting Ratio (%)")
# plt.ylim(46.5,50)
plt.show()

# %% [markdown]
# And of course we'll visualize the final topology. We'll plot the minimum length scale as a circle in the upper corner.

# %%
opt.update_design([mapping(x, eta_i, cur_beta)])
plt.figure()
ax = plt.gca()
opt.plot2D(
    False,
    ax=ax,
    plot_sources_flag=False,
    plot_monitors_flag=False,
    plot_boundaries_flag=False,
)
circ = Circle((2, 2), minimum_length / 2)
ax.add_patch(circ)
ax.axis("off")
plt.show()


