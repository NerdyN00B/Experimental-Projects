import numpy as np
import matplotlib.pyplot as plt


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
                window_size=1):
    """
    Calculate the change in position for the laser due to change in angle.

    Parameters
    ----------
    distance : float, np.ndarray
        The distance from the center of the modules to the window.
    separation : float, np.ndarray, optional
        The separation between the modules, by default 0.5m
    delta_distance : float, optional
        The amplitude of window vibrations, by default 1e-3m
    window_size : float, optional
        The width of the window, by default 1m
    
    returns
    -------
    delta : float, np.ndarray
        the movement of the laser as a result of the window vibrations.
    """
    b = separation / 2
    window_angle = np.arctan(delta_distance / (window_size / 2))
    alpha = np.arctan(b / distance)
    delta = distance * np.tan(alpha + window_angle) - b
    delta -= distance * np.tan(alpha - window_angle) - b
    return delta
    


if __name__ == "__main__":
    distances = np.linspace(1, 100, 1000)
    delta_amp = delta_pos(distances, separation=2, delta_distance=1e-4)
    fig, ax = plt.subplots(2, 1, layout="tight")
    ax[0].plot(
        distances,
        delta_amp * 1e3,
        label="amplitudal change. Separation 2m, Vibration 1mm",
        c='k'
    )
    ax[0].set(
        xlabel="Distance to window (m)",
        ylabel="Laser movement (mm)",
        title="Laser movement due amplitudal window vibrations of order 100μm",
        xscale="log",
    )
    ax[0].grid()
    
    delta_ang = delta_angle(
        distances,
        separation=2,
        delta_distance=1e-4,
        window_size=1
    )
    ax[1].plot(
        distances,
        delta_ang * 1e3,
        label="angular change. Separation 2m, Vibration 100μm, window size 1m",
        c='k'
    )
    ax[1].set(
        xlabel="Distance to window (m)",
        ylabel="Laser movement (mm)",
        title="Laser movement due angular window vibrations of order 100μm",
        xscale="log",
    )
    ax[1].grid()
    
    fig.savefig("preparation/afstand_log_100um_2m.png", dpi=300)
    fig.savefig("preparation/afstand_log_100um_2m.pdf", dpi=300)

    # distance = 10  # m
    # delta = np.linspace(1e-6, 1e-3, 1000)
    # delta_amp = delta_pos(distance, separation=0.5, delta_distance=delta)
    # delta_ang = delta_angle(
    #     distance,
    #     separation=0.5,
    #     delta_distance=delta,
    #     window_size=1
    # )
    
    # fig, ax = plt.subplots(2, 1, layout="tight")
    # ax[0].plot(
    #     delta * 1e3,
    #     delta_amp * 1e3,
    #     label="amplitudal change. Separation 0.5m, Distance 100m",
    #     c='k'
    # )
    # ax[0].set(
    #     xlabel="Window vibration amplitude (mm)",
    #     ylabel="Laser movement (mm)",
    #     title=f"Laser movement due to amplitudal window vibrations at {distance}m",
    #     xscale="log",
    # )
    # ax[0].grid()
    # ax[1].plot(
    #     delta * 1e3,
    #     delta_ang * 1e3,
    #     label="angular change. Separation 0.5m, Distance 100m, window size 1m",
    #     c='k'
    # )
    # ax[1].set(
    #     xlabel="Window vibration amplitude (mm)",
    #     ylabel="Laser movement (mm)",
    #     title=f"Laser movement due to angular window vibrations at {distance}m",
    #     xscale="log",
    # )
    # ax[1].grid()
    # fig.savefig(f"preparation/vibrationamp_{distance}m_log.png", dpi=300)
    # fig.savefig(f"preparation/vibrationamp_{distance}m_log.pdf", dpi=300)
    # plt.show()