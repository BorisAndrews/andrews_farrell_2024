The code used to generate the numerical results in the manuscript
"High-order conservative and accurately dissipative numerical
integrators via auxiliary variables" by Boris D. Andrews and Patrick
E. Farrell.

Packages (located in "avfet_modules/") can be installed by running
    >> pip install .
in the root directory. They can be subsequently uninstalled by running
    >> pip uninstall avfet_modules
in the root directory.



----------



The following gives instructions to reproduce each of the figures in the
manuscript. Code marked with a ">>" must be executed in a shell in the
root directory; unless otherwise stated, this must be done in a
Firedrake virtual environment. Relevant targets are available in the
included Makefile for convenience. All outputs will be saved to a
created "output/" directory.



- To generate the plots in Figs. 1 & 2 for each scheme:
    >> python kepler/sp_demos.py --scheme implicit_midpoint
    >> python kepler/sp_demos.py --scheme cohen_hairer
    >> python kepler/sp_demos.py --scheme labudde_greenspan
    >> python kepler/sp_demos.py --scheme andrews_farrell

These do not require a Firedrake virtual environment, however it does
require a PETSc installation which can be fetched through Firedrake; a
Firedrake virtual environment is therefore sufficient.



- To generate the data in Fig. 3:
    >> mkdir -p output/kepler_convergence/
    >> python kepler/convergence/convergence.py

Outputs will be saved as text files in "output/kepler_convergence/".



- To generate the plots and data in Figs. 4 & 5, in the
Q_2-preserving case:
    >> mkdir -p output/incompressible_ns/avfet/
    >> python incompressible_ns/avfet.py
and in the non-Q_2-preserving case:
    >> mkdir -p output/incompressible_ns/avfet_nohelicity/
    >> python incompressible_ns/avfet_nohelicity.py

Data and Paraview files will be saved in either case to
"output/incompressible_ns/".



- To generate the plots and data in Figs. 6 & 7:
    >> mkdir -p output/kovalevskaya/im/ output/kovalevskaya/avfet/
    >> matlab -batch "run('kovalevskaya.mlx')"

Images will be saved in "output/kovalevskaya/". Naturally, this does
not require a Firedrake virtual environment.



- To generate the plots and data in Figs. 8, 9, & 10, for the
energy-conserving integrator:
    >> mkdir -p output/benjamin_bona_mahony/avfet/
    >> python benjamin_bona_mahony/avfet.py
and for the Gauss method:
    >> mkdir -p output/benjamin_bona_mahony/gauss/
    >> python benjamin_bona_mahony/gauss.py

Data will saved in either case to "output/benjamin_bona_mahony/".

To produce an animation of the data, for the energy-conserving
integrator:
    >> python benjamin_bona_mahony/animation.py --dir output/benjamin_bona_mahony/avfet/
and for the Gauss method:
    >> python benjamin_bona_mahony/animation.py --dir output/benjamin_bona_mahony/gauss/
In either case, the animation can be viewed through a frame of
reference moving at the exact speed of the continuous soliton by appending:
    >> * --cam_speed 1.618034
e.g.
    >> python benjamin_bona_mahony/animation.py --dir output/benjamin_bona_mahony/avfet/ --cam_speed 1.618034



- To generate the plots and data in Figs. 11 & 12, for the
structure-preserving integrator:
    >> mkdir -p output/compressible_ns/supersonic/avfet/
    >> python compressible_ns/supersonic/avfet.py
and for the implicit midpoint method:
    >> mkdir -p output/compressible_ns/supersonic/im/
    >> python compressible_ns/supersonic/im.py

Data and Paraview files will saved in either case to
"output/compressible_ns/supersonic/".



- To generate the plots and data in Figs. 11 & 12, for the
structure-preserving integrator:
    >> mkdir -p output/compressible_ns/euler/avfet/
    >> python compressible_ns/euler/avfet.py
and for the implicit midpoint method:
    >> mkdir -p output/compressible_ns/euler/im/
    >> python compressible_ns/euler/im.py

Data and Paraview files will saved in either case to
"output/compressible_ns/euler/".



----------



Thank you for your interest!
