install:
	pip install .

uninstall:
	pip uninstall avfet_modules



fig1and2_im:
	python kepler/sp_demos.py --scheme implicit_midpoint
fig1and2_mvdg:
	python kepler/sp_demos.py --scheme cohen_hairer
fig1and2_lbg:
	python kepler/sp_demos.py --scheme labudde_greenspan
fig1and2_avfet:
	python kepler/sp_demos.py --scheme avfet



fig3:
	mkdir -p output/kepler_convergence/
	python kepler/convergence/convergence.py



fig4and5_avfet:
	mkdir -p output/incompressible_ns/avfet/
	python incompressible_ns/avfet.py
fig4and5_avfet_nohelicity:
	mkdir -p output/incompressible_ns/avfet_nohelicity/
	python incompressible_ns/avfet_nohelicity.py

fig4and5: fig4and5_avfet fig4and5_avfet_nohelicity



fig6and7:
	mkdir -p output/kovalevskaya/im/ output/kovalevskaya/avfet/
	matlab -batch "run('kovalevskaya.mlx')"



fig8to10_avfet_simulation:
	mkdir -p output/benjamin_bona_mahony/avfet/
	python benjamin_bona_mahony/avfet.py
fig8to10_avfet_animation:
	python benjamin_bona_mahony/animation.py --dir output/benjamin_bona_mahony/avfet/
fig8to10_avfet: fig8to10_avfet_simulation fig8to10_avfet_animation

fig8to10_gauss_simulation:
	mkdir -p output/benjamin_bona_mahony/gauss/
	python benjamin_bona_mahony/gauss.py
fig8to10_gauss_animation:
	python benjamin_bona_mahony/animation.py --dir output/benjamin_bona_mahony/gauss/
fig8to10_gauss: fig8to10_gauss_simulation fig8to10_gauss_animation

fig8to10_both_simulation: fig8to10_avfet_simulation fig8to10_gauss_simulation



fig11and12_avfet:
	mkdir -p output/compressible_ns/supersonic/avfet/
	python compressible_ns/supersonic/avfet.py
fig11and12_im:
	mkdir -p output/compressible_ns/supersonic/im/
	python compressible_ns/supersonic/im.py

fig11and12: fig11and12_avfet fig11and12_im


fig13_avfet:
	mkdir -p output/compressible_ns/euler/avfet/
	python compressible_ns/euler/avfet.py
fig13_im:
	mkdir -p output/compressible_ns/euler/im/
	python compressible_ns/euler/im.py

fig13: fig13_avfet fig13_im
