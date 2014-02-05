from skimage import data
from skimage import measure

import numpy as np
import matplotlib.pyplot as plt

# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
contours = measure.find_contours(r, 0.8)

# Display the image and plot all contours found
plt.imshow(r, interpolation='nearest')

for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

plt.axis('image')
plt.xticks([])
plt.yticks([])
plt.show()