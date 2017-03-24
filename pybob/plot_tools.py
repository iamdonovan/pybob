import matplotlib.pyplot as plt, os
from matplotlib.pyplot import savefig

def save_results_quicklook(img, raster, com_ext, outfilename, vmin=0, vmax=10, sfact=2, output_directory='.'):
	xmin, ymin, xmax, ymax = com_ext
	ext = [xmin, xmax, ymin, ymax]

	fig = img.overlay(raster, extent=ext, sfact=sfact, vmin=vmin, vmax=vmax, showfig=False)

	savefig(os.path.join(output_directory, outfilename + '.png'), bbox_inches='tight', dpi=200)

def plot_geoimg_sidebyside( img1, img2, com_extent, fig=None, cmap='gray', output_directory='.', filename=None):
	if fig is None:
		fig = plt.figure(facecolor='w', figsize=(16.5, 16.5), dpi=200)
	else:
		fig

	ax1 = plt.subplot(121, axisbg=(0.1, 0.15, 0.15))
	ax1_img = plt.imshow(img1, interpolation='nearest', cmap=cmap, extent=com_extent)
	
	ax2 = plt.subplot(122, axisbg=(0.1, 0.15, 0.15))
	ax2_img = plt.imshow(img2, interpolation='nearest', cmap=cmap, extent=com_extent)


	if filename is not None:
		savefig(os.path.join(output_directory, filename), bbox_inches='tight', dpi=200 )

def plot_nice_histogram(data_values, outfilename, output_directory):
	fig = plt.figure(facecolor='w', figsize=(16.5, 16.5), dpi=200)
	
	outputs = plt.hist(data_values, 25, alpha=0.5, normed=1)

	ylim = plt.gca().get_ylim()
	mu = data_values.mean()
	sigma = data_values.std()

	plt.plot([mu, mu], ylim, 'k--')
	plt.plot([mu+sigma, mu+sigma], ylim, 'r--')
	plt.plot([mu+2*sigma, mu+2*sigma], ylim, 'r--')

	plt.text(mu, np.mean(ylim), r'$\mu$', fontsize=24, color='k')
	plt.text(mu+sigma, np.mean(ylim), r'$\mu + \sigma$', fontsize=24, color='r')
	plt.text(mu+2*sigma, np.mean(ylim), r'$\mu + 2\sigma$', fontsize=24, color='r')
	# just to be sure
	plt.gca().set_ylim(ylim)

	savefig(output_directory + os.path.sep + outfilename + '_vdist.png', bbox_inches='tight', dpi=200)

def plot_chips_corr_matrix(srcimg, destimg, corrmat):
	fig = plt.figure()
	plt.ion()

	ax1 = plt.subplot(131)
	ax1_img = plt.imshow(srcimg, cmap='gray')
	ax1.set_title('source image')

	ax2 = plt.subplot(132)
	ax2_img = plt.imshow(destimg, cmap='gray')
	ax2.set_title('dest. image')

	peak1_loc = cv2.minMaxLoc(corrmat)[3]

	ax3 = plt.subplot(133)
	ax3_img = plt.imshow(corrmat, cmap='jet')
	plt.plot(peak1_loc[0], peak1_loc[1], 'w+')

	plt.show()

	return fig
