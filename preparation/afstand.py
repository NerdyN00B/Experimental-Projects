import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


def delta_pos(distance, separation=0.5, delta_distance = 1e-3):
    """
    Calculate the change in position for the laser due to change in distance.

    Parameters
    ----------
    distance : float, np.ndarray
        The distance from the center of the modules to the window.
    separation : float, np.ndarray, optional
        The separation between the modules, by default 0.5m
    delta_distance : float, optional
        The amplitude of window vibrations, by default 1e-3m
    
    returns
    -------
    delta : float, np.ndarray
        the movement of the laser as a result of the window vibrations.
    """
    b = separation / 2
    tantheta = distance / b
    delta = ((distance + delta_distance) / tantheta - b) * 2
    delta -= ((distance - delta_distance) / tantheta - b) * 2
    return delta


def delta_angle(distance,
                separation=0.5,
                delta_distance=1e-3,
                window_size=0.1):
    """
    Calculate the change in position for the laser due to change in angle.

    Parameters
    ----------
    distance : float, np.ndarray
        The distance from the center of the modules to the surface.
    separation : float, np.ndarray, optional
        The separation between the modules, by default 0.5m
    delta_distance : float, optional
        The amplitude of surface vibrations, by default 1e-3m
    window_size : float, optional
        The width of the window, by default 0.1m
    
    returns
    -------
    delta : float, np.ndarray
        the movement of the laser as a result of the surface vibrations.
    """
    b = separation / 2
    window_angle = np.arctan(delta_distance / (window_size / 2))
    alpha = np.arctan(b / distance)
    delta = distance * np.tan(alpha + window_angle) - b
    delta -= distance * np.tan(alpha - window_angle) - b
    return delta
    


if __name__ == "__main__":
    distances = np.linspace(1, 100, 1000)
    delta_distance = np.linspace(1e-7, 1e-4, 1000)
    xx, yy = np.meshgrid(distances, delta_distance)
    zz = delta_pos(xx, delta_distance=yy)
    zz += delta_angle(xx, delta_distance=yy)
    
    fig, ax = plt.subplots(figsize=(8, 6), layout='tight')
    cs = ax.contourf(
        xx,
        yy * 1e6,
        zz * 1e3,
        levels=30,
        cmap='viridis',
        locator=ticker.LogLocator(subs='auto'),
    )
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    minval = zz.min()
    maxval = zz.max()
    
    cbar = fig.colorbar(cs)
    cbar.set_label(r'Laser Movement ($mm$)')
    cbar.set_ticks([0.01, 0.1, 1, 10, 100])
    
    ax.set_xlabel('Distance from Modules to Surface $d$ ($m$)')
    ax.set_ylabel(r'Window Vibration Amplitude $\delta$ ($\mu m$)')
    ax.set_title('Laser Movement due to Surface Vibrations')
    fig.savefig('figures/final_analysis/theoretical_lasermovement.png', dpi=300)
    fig.savefig('figures/final_analysis/theoretical_lasermovement.pdf', dpi=300)