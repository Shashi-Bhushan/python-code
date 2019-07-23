import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(10)


x = np.arange(0.0, 50.0, 2.0)
y = x + np.random.rand(x.shape[0]) * 300

plt.scatter(x, y, alpha=0.5)
plt.xlabel('Independent Variable(x)')
plt.ylabel('Dependent Variable(y)')
plt.show()
