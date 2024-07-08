The code used to generate the numerical results in the manuscript "High-order conservative and accurately dissipative numerical integrators via auxiliary variables" by Boris D. Andrews and Patrick E. Farrell.

Packages (located in "avfet_modules/") can be installed by running "pip install ." in the root directory.

----------

The following gives instructions to reproduce each of the figures in the manuscript.
Code marked with a ">>" must be executed through terminal in the root directory; unless otherwise stated, this must be done in a Firedrake virtual environment.

- To generate the plots in Figs. 1 & 2:
    >> python kepler/sp_demos.py --scheme implicit_midpoint
    >> python kepler/sp_demos.py --scheme cohen_hairer
    >> python kepler/sp_demos.py --scheme labudde_greenspan
    >> python kepler/sp_demos.py --scheme andrews_farrell
This does not require a Firedrake virtual environment, however it does require a PETSc installation which can be fetched through Firedrake; a Firedrake virtual environment is therefore sufficient.

- To generate the data in Fig. 3:
    >> python kepler/convergence/convergence.py

- To generate the plots and data in Figs. 4 & 5:
    >> python incompressible_ns/avfet.py
    >> python incompressible_ns/avfet_nohelicity.py
Data and Paraview files will saved in either case to "output/".

- To generate the plots and data in Figs. 6 & 7, run "kovalevskaya.mlx" in MATLAB.
Naturally, this does not require a Firedrake virtual environment.

- To generate the plots and data in Figs. 8, 9, & 10:
    >> python benjamin_bona_mahony/avfet.py
    >> python benjamin_bona_mahony/gauss.py
Data will saved in either case to "output/".
To visualise the data:
    >> python benjamin_bona_mahony/animation.py
To visualise the data through a frame of reference moving at the soliton speed in the continuous case:
    >> python benjamin_bona_mahony/animation.py --cam_speed 1.618034

- To generate the plots and data in Figs. 11 & 12:
    >> python compressible_ns/supersonic/avfet.py
    >> python compressible_ns/supersonic/im.py
Data and Paraview files will saved in either case to "output/".

- To generate the plots and data in Fig. 13:
    >> python compressible_ns/euler/avfet.py
    >> python compressible_ns/euler/im.py
Data and Paraview files will saved in either case to "output/".

----------

Thank you!
