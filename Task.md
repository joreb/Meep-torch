# Task For presenatation at  examples_2

make a sub folder call examples_2. Using meep-torch

- Using gdsfactory build SIN Ring resonators.
    - a SiN Ring Resonators.
        - [x] Use gplugins to interface with meep-torch. (examples_2/gds_meep_torch.py -- GPL-free adapter playing the gplugins role)
        - [x] Run FDTD Simulation. (examples_2/a_ring_smatrix.py)
        - [x] Get Scattering Matrices. (|S21|^2 spectrum, Q~168, FSR 26 nm)
        - [x] plot response at different detuning, transient + frequency space. (a_ring_smatrix.png)
 
    - b SiN Ring Resonators and Lithium Niobate resonators.
        - [x] Run FDTD Simulation with Chi2 non-linearity. (chi2 added to the engine, a44f0e2; examples_2/b_ring_nonlinear.py)
        - [x] Simulate Kerr effect (resonance red-shift -62 ppm at 4x power)
        - [x] SHG generation (clean line at 2 f0, b_ring_nonlinear.png)
        
    - c 
        - [x] Inverse-design of an edge-coupler. SiN 150 nm / SiO2 (examples_2/c_edge_coupler.py, x1.20)
            - https://www.flexcompute.com/tidy3d/examples/notebooks/EdgeCoupler/
        - [x] Inverse-design of a y splitter. SiN 150 nm / SiO2 (examples_2/c_y_splitter.py, x1.26, imbalance 0.2%)
            - https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd24DigitalSplitter/
    
