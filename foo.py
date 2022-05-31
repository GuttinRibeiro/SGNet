import numpy as np
import matplotlib.pyplot as plt
import imageio

if __name__ == "__main__":
	img = imageio.imread('/media/olorin/Data/PIE/set01/video0001/00001.png')

	# x = np.arange(10)
	# y = 2*x**3
	print(plt.get_backend())
	# plt.ion()
	# plt.plot(x, y)
	plt.imshow(img)
	plt.show()
