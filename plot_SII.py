import numpy as np
import matplotlib.pyplot as plt

what_measured = 'distance'

goed = np.array([9, 6, 3, 5, 3])
vergelijkbaar = np.array([1, 2, 2, 3, 1])
distance = np.array([1, 2, 3, 4, 5])

fig, ax = plt.subplots()

ax.errorbar(
    distance,
    (goed + 0.5 * vergelijkbaar) / 10,
    yerr = 0.5 * vergelijkbaar / 10,
    fmt='ok',
    capsize=5
)

ax.set_xticks(distance)
# ax.set_yticks(np.arange(0, 1, 10))

ax.set_xlabel('Distance from drum (m)')
ax.set_ylabel('Intelligibility Score')
ax.set_title('Speech Intelligibility vs. Distance from Drum')
ax.grid()

fig.savefig(f'figures/final_analysis/{what_measured}.png', dpi=300)
fig.savefig(f'figures/final_analysis/{what_measured}.pdf', dpi=300)