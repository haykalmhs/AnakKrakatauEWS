import numpy as np
from obspy.core import Trace
from scipy.integrate import trapz
from obspy import read

def calc_acoust_power(trace, mode, fac, r, cel, tau, aref, fe=None, att=0, ref_pressure=20e-6, filter_freq=None, rho_air=1.225):
    """
    Calculate acoustic power for different cases using an ObsPy Trace object.

    Parameters:
    - trace: ObsPy Trace object containing the waveform data in Pascals (Pa).
    - mode: Calculation mode ('1', '2b', '2c', '2d').
    - fac: Geometric spreading factor (1, 2, 4, etc.).
    - r: Distance from sensor to source in meters.
    - rho_air: Air density in kg/m^3.
    - cel: Speed of sound in air in m/s.
    - tau: Time window (s) for integration.
    - aref: Reference acoustic pressure (usually 20 microPascals).
    - fe: Sample rate (1/sec), required for mode '1'.
    - att: Attenuation in dB, used in modes '2c' and '2d'.
    - ref_pressure: Reference pressure for dB conversion (default 20 microPascals).

    Returns:
    - Ea: Array of acoustic power in Watts.
    """
    # Read the waveform file into an ObsPy Trace object
    trace = read(trace)[0]
    
    # Apply a low-pass filter if the frequency is specified
    if filter_freq is not None:
        trace.filter('lowpass', freq=filter_freq, corners=2, zerophase=True)


    # Ensure trace.data does not have zero or negative values
    trace.data = np.maximum(trace.data, 1e-10)

    # Calculate geometric spreading factor and other constants
    omega = fac * np.pi * r**2
    int_fac = 1 / (rho_air * cel)

    if mode == '1':
        if fe is None:
            raise ValueError("Sample rate 'fe' must be provided for mode '1'")
        x = trace.data  # Data vector
        window_len = int(tau * fe)  # Window length in samples

        ea = []
        for start in range(0, len(x), window_len):
            end = min(start + window_len, len(x))
            data2 = x[start:end] ** 2
            ea.append(omega * int_fac * (1/fe) * trapz(data2))
        ea = np.array(ea)

    elif mode == '2b':
        # Convert dB to pressure and calculate acoustic power without attenuation factor
        datavec = 20 * np.log10(trace.data / ref_pressure)
        del_p = aref * 10 ** (datavec / 20)
        ea = omega * int_fac * del_p ** 2

    elif mode in ['2c', '2d']:
        # Modes '2c' and '2d' are vector-based with or without attenuation
        t = np.arange(0, len(trace.data)) / trace.stats.sampling_rate
        ti, tf = t[0], t[-1]
        begwind = np.arange(ti, tf, tau)

        ea = []
        for beg in begwind:
            end = min(beg + tau, tf)
            idx_start = int(beg * trace.stats.sampling_rate)
            idx_end = int(end * trace.stats.sampling_rate)
            
            if mode == '2c':
                xint = np.mean(20 * np.log10(trace.data[idx_start:idx_end] / ref_pressure) + att)
            else:  # mode '2d'
                xint = np.mean(20 * np.log10(trace.data[idx_start:idx_end] / ref_pressure))
            
            del_p = aref * 10 ** (xint / 20)
            ea.append(omega * int_fac * del_p ** 2)
        ea = np.array(ea)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return ea

def calc_vel_radiation_pat(ea, rho_air, cel, r_cond, mode='monopole'):
    """
    Calculate the vent gas exit velocity based on the acoustic power at the vent.

    Parameters:
    - ea: Acoustic power in Watts.
    - rho_air: Air density in kg/m^3.
    - cel: Speed of sound in air in m/s.
    - r_cond: Conduit radius in meters.
    - mode: 'monopole' or 'dipole' (default is 'monopole').

    Returns:
    - v_gas: Vent gas exit velocity for the selected mode.
    """

    if mode == 'monopole':
        n, m, K, f = 4, 1, 1, 4
        v_gas = ((ea * cel**m) / (K * f * rho_air * np.pi * r_cond**2))**(1/n)

    elif mode == 'dipole':
        n, m, K, f = 6, 3, 1.3e-2, 1
        v_gas = ((ea * cel**m) / (K * f * rho_air * np.pi * r_cond**2))**(1/n)

    else:
        raise ValueError("Invalid mode. Choose 'monopole' or 'dipole'.")

    return v_gas

def calculate_volumetric_flux(v_gas, R):
    """
    Calculate the volumetric flux based on gas exit velocity and vent radius.

    Parameters:
    - v_gas: Gas exit velocity (m/s).
    - R: Vent radius (m).

    Returns:
    - Q: Volumetric flux (m^3/s).
    """
    Q = np.pi * R**2 * v_gas
    return Q
