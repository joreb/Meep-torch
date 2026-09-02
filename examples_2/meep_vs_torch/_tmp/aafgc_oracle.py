
import json, sys
import numpy as np
import meep as mp
import matplotlib
matplotlib.use("Agg")

p = json.load(open(sys.argv[1]))
import h5py
eps = np.load(p["eps_npy"])
with h5py.File(p["eps_h5"], "w") as f:
    f.create_dataset("eps", data=eps)
sim = mp.Simulation(
    cell_size=mp.Vector3(p["LX"], p["LZ"]),
    resolution=p["RES"],
    epsilon_input_file=p["eps_h5"],
    boundary_layers=[mp.PML(p["NPML"] / p["RES"])],
    sources=[mp.EigenModeSource(
        mp.ContinuousSource(p["FCEN"]),
        center=mp.Vector3(p["x_src"] - p["LX"] / 2, p["z_src"] - p["LZ"] / 2),
        size=mp.Vector3(0, p["span"]),
        eig_parity=mp.ODD_Z)],
)
animate = mp.Animate2D(fields=mp.Ez, realtime=False, normalize=True,
                       eps_parameters={"contour": False, "alpha": 0.25},
                       field_parameters={"alpha": 0.9, "cmap": "RdBu",
                                         "interpolation": "none"})
sim.run(mp.at_every(p["snap"], animate), until=p["T"])
animate.to_mp4(p["fps"], p["mp4"])
ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(p["LX"], p["LZ"]),
                   component=mp.Ez)
np.savez(p["out_npz"], ez=ez)
print("MEEP_DONE")
