import instaseis
import matplotlib.pyplot as plt
import obspy
import numpy as np
import pandas as pd
from obspy.imaging.beachball import beachball
from obspy.imaging.beachball import mt2plane
from obspy.core.event.source import MomentTensor
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.geodetics.base import kilometer2degrees
from geopy.distance import geodesic
from geopy.point import Point
import math

"""
Assumes you have a local Syngine database!
#db = instaseis.open_db("/projects/wg-BayesianFWI/")
"""
def gauss_sliprate(t, mean=5,variance=1.0):
    """Gaussian sliprate function."""
    #dt=db.info.dt
    #npts=1000
    #times=np.linspace(-50, dt*npts, npts)
    sliprate = np.exp(-0.5*((t-mean)/variance)**2)
    sliprate /= np.sum(sliprate)
    return sliprate

def boxcar_sliprate(t, width=15):
    """Boxcar sliprate function."""
    slipratebox = np.ones_like(t)
    slipratebox[t > width] = 0
    return sliprate
    
def triangle_sliprate(t,start=5, midpoint = 15, apex = 25):
    """"triangle sliprate function 
        parameters
        t: array of time values incremented by dt
        start: time in secs at which rate starts
        midpoint: time in secs at which apex is reached
        apex: peak of triangle

        returns: array of normalized sliprates per dt
    """
   
    if start < 0 or midpoint < 0 or apex < 0:
        raise ValueError("All inputs (start, midpoint, apex) must be non-negative")
    if midpoint <= start:
        raise ValueError("The midpoint must be greater than start")
        
    slope1 = (apex-0) / (midpoint-start)
    yint1 = 0 - start*slope1
    y1 = slope1 * t + yint1
    xint = midpoint + (midpoint - start)
    slope2 = (apex - 0) / (midpoint - xint)
    yint2 = 0-slope2*xint

    y1[t >= midpoint] = slope2 * times[t >= midpoint] + yint2

    y1[y1 < 0] = 0
    sliprate = y1 / np.sum(y1)
   # sliprate = y1 / np.sum(y1)
    return sliprate

class Seismogram:
    def __init__(self):
        self._dist_in_degrees = 2
        self.data = None
        self.delta = None
        self.velocity_data = None
        self.accel_data = None
        self.source_lat = 0
        self.source_lon = 0 
        #self.receiver_lat = self.dist_in_degrees
        #self.receiver_lon = 0
        self.receiver_lat = 0
        self.receiver_lon = self.dist_in_degrees
        self.depth_in_m=10000
        
        # Default isotropic moment tensor components
        self.default_m_rr = 1e16
        self.default_m_tt = 1e16
        self.default_m_pp = 1e16
        self.default_m_rt = 0
        self.default_m_rp = 0
        self.default_m_tp = 0

        # Initialize with default moment tensor components
        self.m_rr = self.default_m_rr
        self.m_tt = self.default_m_tt
        self.m_pp = self.default_m_pp
        self.m_rt = self.default_m_rt
        self.m_rp = self.default_m_rp
        self.m_tp = self.default_m_tp
        
        self.sliprate = None
        self.dt = db.info.dt
        #self.starttime = obspy.UTCDateTime("1970-01-01T00:00:00.0")
        #self.endtime = self.starttime + 300
        
        # Initialize moment_tensor (if provided)
        self._moment_tensor = None
 
    @property
    def moment_tensor(self):
        return self._moment_tensor

    @moment_tensor.setter
    def moment_tensor(self, tensor):
        if tensor is None:
            self._moment_tensor = None
            self.m_rr = self.default_m_rr
            self.m_tt = self.default_m_tt
            self.m_pp = self.default_m_pp
            self.m_rt = self.default_m_rt
            self.m_rp = self.default_m_rp
            self.m_tp = self.default_m_tp
            return
            return
            
        # Validate that the tensor is a 3x3 numpy array
        if not isinstance(tensor, np.ndarray) or tensor.shape != (3, 3):
            raise ValueError("Moment tensor must be a 3x3 numpy array.")
       
        # Update the internal attribute
        self._moment_tensor = tensor

        # Update the individual components
        self.m_rr = tensor[0, 0]
        self.m_tt = tensor[1, 1]
        self.m_pp = tensor[2, 2]
        self.m_rt = tensor[0, 1]
        self.m_rp = tensor[0, 2]
        self.m_tp = tensor[1, 2]

    @property
    def dist_in_degrees(self):
        return self._dist_in_degrees

    @dist_in_degrees.setter
    def dist_in_degrees(self, value):
        self._dist_in_degrees = value
        self.receiver_lon = value  # Automatically update source_lon when dist_in_degrees is set

        
    def custom_seismogram(self,**kwargs):
        """
        Generate a seismogram using Instaseis with Gaussian, boxcar, or triangle or optional 
        custom sliprate function.
    
        Parameters:
            source_lat: source event longitude
            source_lon: source event latitude
            receiver_lat: receiver event longitude
            receiver_lon: receiver event latitude
            depth_in_m: depth to even origin in meters
            m_rr,m_tt, m_pp, m_rt, m_rp, m_tp: moments defining moment tensor around radial,
            theta, and phi directions; default is set to an isotropic source moment tensor
            database: Instaseis database object.
            source: Instaseis source object.
            receiver: Instaseis receiver object.
            sliprate: None (use database default), 'Gauss', 'boxcar', or a custom function.
            kwargs--mean: mean for Gaussian sliprate function, default is 5
            kwargs--variance: variance for Gaussian sliprate function, default is 1
            kwargs--width: width for boxcar sliprate function, default is 15
            db: seismic database
            **kwargs: Additional parameters for the sliprate functions, e.g., mean, variance or width.
        """
        sliprate_options = {
        "Gauss": gauss_sliprate,
        "boxcar": boxcar_sliprate,
        "triangle": triangle_sliprate
        }
        #print(self.sliprate)
        # Determine sliprate function
        # if sliprate is None, use default stf from db and return seismogram
        if self.sliprate is None:
            print('No sliprate was specified; using default STF from db')
            sliprate_function = None  # Use database default
            source = instaseis.Source(
                latitude = self.source_lat,
                longitude = self.source_lon,
                depth_in_m= self.depth_in_m,
                m_rr = self.m_rr,
                m_tt = self.m_tt,
                m_pp = self.m_pp,
                m_rt = self.m_rt,
                m_rp = self.m_rp,
                m_tp = self.m_tp,
                sliprate=sliprate_function)
                #dt=self.dt)
            receiver = instaseis.Receiver(
                latitude=self.receiver_lat,
                longitude=self.receiver_lon)
            seismogram=db.get_seismograms(source=source, receiver=receiver)
            self.data = {
            'Z': seismogram.select(component="Z")[0].data,
            'N': seismogram.select(component="N")[0].data,
            'E': seismogram.select(component="E")[0].data
            }
           
            self.delta = seismogram.select(component="Z")[0].stats.delta
            return
            #return self
        # if gauss or boxcar, if neither, throws an error
        elif isinstance(self.sliprate, str):
            if self.sliprate in sliprate_options:
                sliprate_function = lambda t: sliprate_options[self.sliprate](t, **kwargs)
                #print(f'using {self.sliprate} sliprate function')
            else:
                raise ValueError(f"Sliprate option '{self.sliprate}' is not recognized. Available options are: {list(sliprate_options.keys())}.")
        # custom sliprate function
        elif callable(self.sliprate):
            sliprate_function = self.sliprate
            print('using custom STF')
        else:
            raise TypeError("Sliprate must be None, a recognized string, or a callable function.")
    
        # get values of sliprate function by evaluating at discrete time points
        dt=db.info.dt
        npts=1000
        times=np.linspace(-50, dt*npts, npts)  # Example time array; adjust as needed
        sliprate_values = sliprate_function(times)
    
        # Generate and return the seismogram
        source = instaseis.Source(
            latitude = self.source_lat,
            longitude = self.source_lon,
            depth_in_m=self.depth_in_m,
            m_rr = self.m_rr,
            m_tt = self.m_tt,
            m_pp = self.m_pp,
            m_rt = self.m_rt,
            m_rp = self.m_rp,
            m_tp = self.m_tp,
            sliprate=sliprate_values,
            dt=self.dt)
        
        source.set_sliprate(sliprate_values, dt, time_shift=0, normalize=True)
        receiver = instaseis.Receiver(
                latitude=self.receiver_lat,
                longitude=self.receiver_lon)
        seismogram=db.get_seismograms(source=source, receiver=receiver, reconvolve_stf=True, remove_source_shift=False)
        self.data = {
            'Z': seismogram.select(component="Z")[0].data,
            'N': seismogram.select(component="N")[0].data,
            'E': seismogram.select(component="E")[0].data
        }
        self.delta = seismogram.select(component="Z")[0].stats.delta
        #print(self.delta)
        

    def get_response(self, resp = 'VEL'):
        """
        calculate velocity 'VEL' or acceleration 'ACCEL'
        updates velocity_data or accel_data of Seismogram instance
        """
        if self.data is None:
            raise ValueError("Displacement has not been generated yet")
        elif resp == "VEL":
            self.velocity_data = {comp: np.gradient(data, self.delta) for comp, data in self.data.items()}
        elif resp == "ACCEL":
            self.accel_data = {comp: np.gradient(np.gradient(data, self.delta), self.delta) for comp, data in self.data.items()}
        else: 
            raise ValueError("resp value not recognized. Choices are either 'VEL' or 'ACCEL' ")

    def compute_integrated_signal_power(self):
        """Calculate the integrated signal power of the velocity."""
        if self.velocity_data is None:
            raise ValueError("Velocity data has not been generated yet. Call 'get_response(resp=\"VEL\")' first.")
       
        power = 0
        for comp, values in self.velocity_data.items():
            power += np.sum(values**2) * self.delta  # Sum of squared values times the sampling interval
       
        return power

    def compute_maximum_magnitude(self, data_type="DISP"):
        """
        Calculate the maximum magnitude (displacement, velocity, or acceleration).
       
        Parameters:
        - data_type (str): The type of data to compute the magnitude for. Options are "DISP", "VEL", "ACCEL".
       
        Returns:
        - max_magnitude (float): The maximum magnitude observed.
        """
        if data_type == "DISP":
            data = self.data
        elif data_type == "VEL":
            if self.velocity_data is None:
                raise ValueError("Velocity data has not been generated yet. Call 'get_response(resp=\"VEL\")' first.")
            data = self.velocity_data
        elif data_type == "ACCEL":
            if self.accel_data is None:
                raise ValueError("Acceleration data has not been generated yet. Call 'get_response(resp=\"ACCEL\")' first.")
            data = self.accel_data
        else:
            raise ValueError("Invalid data_type. Options are 'DISP', 'VEL', or 'ACCEL'.")
       
        # Compute magnitude at each time step
        magnitudes = np.sqrt(
            data["Z"]**2 + data["N"]**2 + data["E"]**2
        )
       
        # Return the maximum magnitude
        return np.max(magnitudes)


    def compute_total_power(self):
        """Calculate the total power for velocity across all three components."""
        if self.velocity_data is None:
            raise ValueError("Velocity data has not been generated yet. Call 'get_response(resp=\"VEL\")' first.")
       
        total_power = 0
        for comp, values in self.velocity_data.items():
            total_power += np.sum(values**2) * self.delta # Sum of squared values times the sampling interval
       
        return total_power

    def plot(self, data_type="DISP", starttime=None, endtime=None):
        """
        Plot the seismogram data (displacement, velocity, or acceleration) for Z, N, and E components.
       
        Parameters:
        - data_type (str): The type of data to plot. Options are "DISP", "VEL", "ACCEL".
                           "DISP" plots displacement (default),
                           "VEL" plots velocity,
                           "ACCEL" plots acceleration.
        - starttime (float): Start time in seconds for the x-axis (default is the beginning of the data).
        - endtime (float): End time in seconds for the x-axis (default is the end of the data).
        """
        # Determine the data to plot
        if data_type == "DISP":
            data = self.data
            title = "Displacement"
        elif data_type == "VEL":
            if self.velocity_data is None:
                raise ValueError("Velocity data has not been generated yet. Call 'get_response(resp=\"VEL\")' first.")
            data = self.velocity_data
            title = "Velocity"
        elif data_type == "ACCEL":
            if self.accel_data is None:
                raise ValueError("Acceleration data has not been generated yet. Call 'get_response(resp=\"ACCEL\")' first.")
            data = self.accel_data
            title = "Acceleration"
        else:
            raise ValueError("Invalid data_type. Options are 'DISP', 'VEL', or 'ACCEL'.")
    
        # Create time axis based on delta
        npts = len(next(iter(data.values())))  # Get number of samples from any component
        times = np.arange(0, npts * self.delta, self.delta)
    
        # Apply starttime and endtime
        if starttime is None:
            starttime = times[0]
        if endtime is None:
            endtime = times[-1]
    
        # Select the time range for plotting
        mask = (times >= starttime) & (times <= endtime)
        times_zoomed = times[mask]
    
        # Plot each component
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, (comp, values) in zip(axes, data.items()):
            ax.plot(times_zoomed, values[mask], label=f"{comp}-Component")
            ax.set_ylabel(f"{title} ({comp})")
            ax.legend(loc="upper right")
            ax.grid(True)
       
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{title} Seismogram")
        plt.tight_layout()
        plt.show()

# from Modern Global Seismology. Thorne Lay & Terry C. Wallace. Pg 343.
# NOTE: slip=rake
# Example usage
#strike = 45  # degrees
#dip = 30     # degrees
#slip = 90    # degrees
def make_EQ_tensor_EMB(strike, dip, slip):
    # Calculate the tensor terms for an arbitrarily oriented double couple EQ given
    # strike, dip, and slip angles (in degrees)

    # Convert to radians
    delta = np.radians(dip)
    phi = np.radians(strike)
    lambda_ = np.radians(slip)

    # Initialize the tensor matrix
    M = np.zeros((3, 3))

    M[0, 0] = -1 * (np.sin(delta) * np.cos(lambda_) * np.sin(2 * phi) +
                   np.sin(2 * delta) * np.sin(lambda_) * (np.sin(phi) ** 2))
    M[1, 1] = (np.sin(delta) * np.cos(lambda_) * np.sin(2 * phi) -
               np.sin(2 * delta) * np.sin(lambda_) * (np.cos(phi) ** 2))
    M[2, 2] = -(M[0, 0] + M[1, 1])
    M[0, 1] = (np.sin(delta) * np.cos(lambda_) * np.cos(2 * phi) +
               0.5 * np.sin(2 * delta) * np.sin(lambda_) * np.sin(2 * phi))
    M[0, 2] = -(np.cos(delta) * np.cos(lambda_) * np.cos(phi) +
                np.cos(2 * delta) * np.sin(lambda_) * np.sin(phi))
    M[1, 2] = -(np.cos(delta) * np.cos(lambda_) * np.sin(phi) -
                np.cos(2 * delta) * np.sin(lambda_) * np.cos(phi))
    
    # Symmetric entries
    M[1, 0] = M[0, 1]
    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]

    return M


def generate_moment_tensor(strike=None, dip=None, slip=None):
    """
    Generate a synthetic moment tensor based on ISO, CLVD, and DC components.

    Parameters:
    - strike (float, optional): Strike angle in degrees (default: sampled from [0, 360]).
    - dip (float, optional): Dip angle in degrees (default: sampled from [0, 90]).
    - slip (float, optional): Slip (rake) angle in degrees (default: sampled from [-180, 180]).

    Returns:
    - moment_tensor_scaled (np.ndarray): A 3x3 scaled moment tensor.
    """

    # Step 1: Sample ISO, CLVD, DC weights
    weights = np.random.dirichlet([1, 1, 8])  # Adjust alpha parameters for different distributions
    w_iso, w_clvd, w_dc = weights

    # Step 2: Define base components
    iso_component = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]) / 3

    clvd_component = np.array([[2, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])

    # Step 3: Determine strike, dip, and slip
    if strike is None:
        strike = np.random.uniform(0, 360)  # Sample strike if not provided
    if dip is None:
        dip = np.random.uniform(0, 90)  # Sample dip if not provided
    if slip is None:
        slip = np.random.uniform(-180, 180)  # Sample slip if not provided

    # Step 4: Generate DC component based on strike, dip, and slip
    dc_component = make_EQ_tensor_EMB(strike, dip, slip)  # Ensure this function is defined elsewhere

    # Step 5: Combine components
    moment_tensor = (
        w_iso * iso_component +
        w_clvd * clvd_component +
        w_dc * dc_component
    )

    # Use this to scale moment tensor if desired
    # Step 6: Sample moment magnitude (Mw) and compute seismic moment (M0)
    # Mw = np.random.uniform(3.5, 8.5)  # Adjust range as needed
    #Mw = 5
    #M0 = 10 ** (1.5 * Mw + 9.1)

    # Step 7: Scale the moment tensor
    #moment_norm = np.sqrt(0.5 * np.sum(moment_tensor**2))  # Frobenius norm
    #moment_tensor_scaled = moment_tensor * (M0 / moment_norm)

    #return moment_tensor_scaled
    return moment_tensor

