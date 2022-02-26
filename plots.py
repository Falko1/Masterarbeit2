from preprocessing import make_positional_sine
import numpy as np
import matplotlib.pyplot as plt

enc = make_positional_sine(1, 128, 100, [100])
plt.plot(np.arange(100), enc[0, :, 16], color="blue", linestyle="dashed")
plt.plot(np.arange(100), enc[0, :, 17], color="blue", )
plt.plot(np.arange(100), enc[0, :, 32], color="orange", linestyle="dashed")
plt.plot(np.arange(100), enc[0, :, 33], color="orange", )
plt.scatter(np.arange(100), enc[0, :, 33], facecolors='none', edgecolors='r')

plt.show()
x = np.arange(1, 2000)
figure = plt.figure()
ax1 = figure.add_subplot()
ax1.plot(x, 0.00003 * np.minimum(x ** (-0.5), x * 400 ** (-1.5)))
ax1.set_xlabel('epoch')
ax1.set_ylabel('learning rate value')
plt.title('Learning Rate with Warm Up')
plt.show()
