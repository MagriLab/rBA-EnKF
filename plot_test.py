import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
t = np.linspace(0, 10, 10)


fig, ax = plt.subplots()


plt.ylim(-1, 1)

def animate(i):
    ax.clear()
    ax.set(ylim=[-1, 1])
    ax.plot(np.sin(x * t[i]))

ani = matplotlib.animation.FuncAnimation(fig, animate,
                frames=len(t), interval=300, repeat=False)

# ani.save('file.gif')
plt.show()
