import numpy as np
import matplotlib.pyplot as plt

GFLOPS = np.array([2.133, 2.121, 30.552, 25.007, 52.506, 14.407, .011, .011, 18.718, 7.695])

dp1_GFLOPS = GFLOPS[0:2]
dp2_GFLOPS = GFLOPS[2:4]
dp3_GFLOPS = GFLOPS[4:6]
dp4_GFLOPS = GFLOPS[6:8]
dp5_GFLOPS = GFLOPS[8:10]


def plot_roofline(peak_gflops=200, bandwidth_limit=30):
	# arithmetic intensity range (FLOP/byte)
	ai = np.logspace(-3, 3, 400)

	# Roofline: compute bound = peak GFLOP/s, memory bound = bandwidth * AI
	roof_mem = bandwidth_limit * ai  # GB/s * FLOP/byte -> GFLOP/s if units aligned
	roof_compute = np.full_like(ai, peak_gflops)
	roof = np.minimum(roof_mem, roof_compute)

	fig, ax = plt.subplots(figsize=(8,6))
	ax.loglog(ai, roof_mem, '--', label=f'Bandwidth limit ({bandwidth_limit} GB/s)')
	ax.loglog(ai, roof_compute, '-', label=f'Peak FLOPS limit ({peak_gflops} GFLOP/s)')
	# ax.loglog(ai, roof, 'k-', linewidth=2, label='roofline')

	# Example markers: place the measured points at some example AI values
	# Here we distribute example AI values for the five methods
	ai = np.full(5, 0.25)
	groups = [dp1_GFLOPS, dp2_GFLOPS, dp3_GFLOPS, dp4_GFLOPS, dp5_GFLOPS]
	colors = ['C0','C1','C2','C3','C4']
	marker1 = 'o'  # shape for first point in pair (N=1,000,000)
	marker2 = 's'  # shape for second point in pair (N=300,000,000)
	for i, grp in enumerate(groups):
		x_val = ai[i]
		# first point (filled)
		ax.scatter(x_val, grp[0], marker=marker1, label=f'dp{i+1}', facecolors=colors[i], edgecolors=colors[i], s=64)
		# second point (hollow)
		ax.scatter(x_val, grp[1], marker=marker2, label='_nolegend_', facecolors='none', edgecolors=colors[i], s=64)

	# balance point where memory roof meets compute roof: AI* = peak_compute / peak_bandwidth
	threshold = peak_gflops / bandwidth_limit
	ax.axvline(threshold, color='gray', linestyle='--', linewidth=1, label=f'Compute–Bandwidth Threshold')
	# mark a reference arithmetic intensity
	ax.axvline(0.25, color='purple', linestyle=':', linewidth=1, label='Arithmetic Intensity = 0.25 FLOP/byte')

	ax.set_xlabel('Arithmetic intensity (FLOP / byte)  log-scale')
	ax.set_ylabel('Performance (GFLOP/s) log-scale')
	ax.set_title('Microbenchmarks Plotted on Idealized Roofline Model')
	# tighten y-axis to focus on measured points
	ax.set_ylim(-100, 1000)

	ax.grid(True, which='both', ls='--', lw=0.5)
	# add marker legend for N values
	# legend markers for N values (filled vs hollow)
	ax.scatter([], [], marker=marker1, s=64, facecolors='k', edgecolors='k', label='N=1,000,000')
	ax.scatter([], [], marker=marker2, s=64, facecolors='none', edgecolors='k', label='N=300,000,000')
	ax.legend()
	plt.tight_layout()
	out_path = 'roofline.png'
	plt.savefig(out_path, dpi=200, bbox_inches='tight')
	print(f"Saved roofline plot to: {out_path}")
	plt.show()


if __name__ == '__main__':
	plot_roofline(peak_gflops=200.0, bandwidth_limit=30.0)