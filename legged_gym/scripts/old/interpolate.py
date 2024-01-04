import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

x = np.array([0, 3, 6])
y = np.array([0, 0.1, 0])

xnew = np.linspace(x.min(), x.max(), 7)
smooth = InterpolatedUnivariateSpline(x, y, k=2)

plt.plot(xnew, smooth(xnew))
plt.show()