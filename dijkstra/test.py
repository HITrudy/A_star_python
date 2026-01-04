import numpy as np
import matplotlib.pyplot as plt

data = np.load('sample_000000.npz')
m  = data['map']
cm = data['costmap']
en = data['end']

plt.imshow(m, cmap='Greys', interpolation='nearest')
plt.show()
plt.imshow(cm, cmap='Greys', interpolation='nearest')
plt.show()
