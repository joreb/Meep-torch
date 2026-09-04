# Basics.md under meep-torch

| section | meep-torch | meep | verdict |
|---|---|---|---|
| A Straight Waveguide | ok | ok | no numbers printed |
| A 90° Bend | raises: TypeError: Invalid shape (334, 160, 160) for image data | ok | - |
| Transmittance Spectrum of a Waveguide Bend | timeout(180s) | ok | - |
| Angular Reflectance Spectrum of a Planar Interface | raises: FileNotFoundError: flux_t0.dat not found. | raises: FileNotFoundError: flux_t0.dat not found. | - |
| Mie Scattering of a Lossless Dielectric Sphere | raises: ModuleNotFoundError: No module named 'PyMieScatt' | raises: ModuleNotFoundError: No module named 'PyMieScatt' | - |
| Differential/Radar Cross Section | raises: ModuleNotFoundError: No module named 'PyMieScatt' | raises: ModuleNotFoundError: No module named 'PyMieScatt' | - |
| Absorbed Power Density Map of a Lossy Cylinder | raises: NotImplementedError: is_integrated=True is not supported: the engine injects soft current sources only (no integrated d(current)/dt path) | ok | - |
| Modes of a Ring Resonator | raises: NameError: name 'mp' is not defined. Did you mean: 'map'? | raises: NameError: name 'mp' is not defined. Did you mean: 'map'? | - |
| Visualizing 3d Structures | raises: AttributeError: plot3D. Did you mean: 'plot2D'? | raises: ModuleNotFoundError: No module named 'vispy' | - |

1/9 runnable sections execute under meep-torch (timeout 180s).
