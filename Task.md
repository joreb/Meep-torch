# Task For presenatation at  examples_2

make a sub folder call examples_2. Using meep-torch

- Using gdsfactory build SIN Ring resonators.
    - a SiN Ring Resonators.
        - [x] Use gplugins to interface with meep-torch. (examples_2/gds_meep_torch.py -- GPL-free adapter playing the gplugins role)
        - [x] Run FDTD Simulation. (examples_2/a_ring_smatrix.py)
        - [x] Get Scattering Matrices. (|S21|^2 spectrum, Q~168, FSR 26 nm)
        - [x] plot response at different detuning, transient + frequency space. (a_ring_smatrix.png)
        - [x] make video of wave propagating in ring (examples_2/a_ring_propagation.mp4)
        - [x] export gdsii file (examples_2/a_ring_SiN.gds)
 
    - b SiN Ring Resonators and Lithium Niobate resonators.
        - [x] Run FDTD Simulation with Chi2 non-linearity. (chi2 in the engine, a44f0e2)
            - SUPERSEDED: the cited script b_ring_nonlinear.py was DELETED (b682d74).
              It modelled 1550 nm / r=3 with a single flat index for all frequencies,
              so it could not represent phase mismatch at all -- an unpoled ring at
              674/1348 converts ~nothing across 366 coherence lengths.
        - [~] Simulate Kerr effect -- NOT VALIDATED on the real device.
            - The "-62 ppm at 4x power" figure came from that deleted flat-index demo.
            - On the TFLN ring the Kerr run was sized but NEVER RAN; investigation
              halted by USER decision. See examples_2/nonlinear/SHG_METHOD_CHANGE.md.
        - [~] SHG generation -- partially validated on the TFLN ring.
            - PASSED: the P^2 second-order gate, eta(2A)/eta(A) = 3.772 (want 4.0),
              pump scaling amp^2.00 exactly. This is a genuine chi2 signature.
            - NOT MEASURED: any trustworthy absolute conversion efficiency. Requires
              quasi-phase-matched poling (Lambda = 3.43 um, ~183 domains), which is
              implemented but whose steady-state efficiency was never run.
            - All scripts, outputs and the two SHG reports: examples_2/nonlinear/.
        - [x] make video of 3d simulation (examples_2/b_ring_3d.mp4 -- 300 nm TFLN microring, true 3D)
        - [x] export gdsii file for both cases (b_ring_SiN.gds, b_ring_Lithium_Niobate.gds)
            
        
    - c 
        - [x] Inverse-design of an edge-coupler. SiN 150 nm / SiO2 (examples_2/c_edge_coupler.py, x1.20)
            - https://www.flexcompute.com/tidy3d/examples/notebooks/EdgeCoupler/
        - [x] Inverse-design of a y splitter. SiN 150 nm / SiO2 (examples_2/c_y_splitter.py, x1.26, imbalance 0.2%)
            - https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd24DigitalSplitter/
    
        -[x] videos: c_inverse_design_evolution.mp4 + c_splitter_3d.mp4 (3D of final design); gds: c_edge_coupler.gds, c_y_splitter.gds
