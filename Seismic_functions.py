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
from scipy.stats import dirichlet

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
    '''
    Calculate the tensor terms for an arbitrarily oriented double couple EQ given
    strike, dip, and slip angles (in degrees)

    from Modern Global Seismology. Thorne Lay & Terry C. Wallace. Pg 343.
    NOTE: slip=rake
    
    Parameters
    ----------
    strike, dip, slip: strike,dip, and slip parameters in degrees

    Returns
    ----------
    M : np.array
        3x3 moment tensor
    '''

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


def generate_moment_tensor(strike=None, dip=None, slip=None, MT_components=[1,1,1]):
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
    #weights = dirichlet.rvs(alpha=[1,1,1], size=1)[0]
    weights = dirichlet.rvs(alpha=MT_components, size=1)[0]
    #weights = np.random.dirichlet([1, 1, 8])  # Adjust alpha parameters for different distributions
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

    # Step 6: Sample moment magnitude (Mw) and compute seismic moment (M0)
    # Mw = np.random.uniform(3.5, 8.5)  # Adjust range as needed
    #Mw = 5
    #M0 = 10 ** (1.5 * Mw + 9.1)
    # M0 = 1

    # Step 7: Scale the moment tensor
    #moment_norm = np.sqrt(0.5 * np.sum(moment_tensor**2))  # Frobenius norm
    #moment_tensor_scaled = moment_tensor * (M0 / moment_norm)

    #return moment_tensor_scaled
    return moment_tensor

def generate_latlon_list(lat_start, lat_end, lon_start, lon_end, step):
    """
    Generate a list of (lat, lon) tuples over a regular grid.

    Parameters:
        lat_start, lat_end: float
            Latitude bounds (inclusive)
        lon_start, lon_end: float
            Longitude bounds (inclusive)
        step: float
            Step size in degrees

    Returns:
        List[Tuple[float, float]]: List of (lat, lon) coordinate pairs
    """
    lat_vals = np.arange(lat_start, lat_end + step, step)
    lon_vals = np.arange(lon_start, lon_end + step, step)
    return [(lat, lon) for lat in lat_vals for lon in lon_vals]

def generate_latlon_df(lat_start, lat_end, lon_start, lon_end, step):
    """
    Generate a DataFrame of lat/lon grid points.

    Parameters:
        Same as generate_latlon_list

    Returns:
        pd.DataFrame with columns ['lat', 'lon']
    """
    lat_vals = np.arange(lat_start, lat_end + step, step)
    lon_vals = np.arange(lon_start, lon_end + step, step)
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    return pd.DataFrame({
        'lat': lat_grid.ravel(),
        'lon': lon_grid.ravel()
    })

def repeat_latlon_pairs(latlon_list, n_samples):
    """
    Repeat arbitrary (lat, lon) pairs n_samples times each.

    Parameters:
        latlon_list: list of (lat, lon) tuples
        n_samples: int, number of samples per location

    Returns:
        lat_array, lon_array: np.arrays with length len(latlon_list) * n_samples
    """
    lat_vals = []
    lon_vals = []
    for lat, lon in latlon_list:
        lat_vals.extend([lat] * n_samples)
        lon_vals.extend([lon] * n_samples)
    return np.array(lat_vals), np.array(lon_vals)

def signed_azimuth_difference(source, locA, locB):
    """
    Compute the signed angle from *A to B* with respect to a common source (in degrees).
    Positive = counterclockwise rotation. If either location is the same as the source,
    return 0.0 (no angular difference defined).
    """

    def to_unit_vector(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z])

    src = to_unit_vector(*source)
    vecA = to_unit_vector(*locA) - src
    vecB = to_unit_vector(*locB) - src

    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)

    if normA < 1e-10 or normB < 1e-10:
        return 0.0  # Defined as zero angle if one of the locations is the source

    vecA /= normA
    vecB /= normB

    cross = np.cross(vecA, vecB)
    dot = np.dot(vecA, vecB)
    sign = np.sign(np.dot(src, cross))  # +1 or -1 depending on direction
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    return np.degrees(angle) * sign

def compute_distance_from_source(receiver_latlon, source_latlon=(0, 0)):
    """
    Computes the great-circle distance in degrees between a receiver and a source.

    Parameters:
    - receiver_latlon: (lat, lon) tuple of the receiver
    - source_latlon: (lat, lon) tuple of the source, default is (0, 0)

    Returns:
    - distance in degrees
    """
    distance_km = great_circle(source_latlon, receiver_latlon).kilometers
    distance_deg = distance_km / 111.195  # Convert km to degrees (approx)
    return distance_deg

def generate_dataset_shared_latents_truth(
    fixed_locations,
    depth,
    num_samples=50,
    sliprate_range=(1,3),
    MT_components=[1,1,1]):
    """
    Generates a dataset for GP-based correlation analysis at fixed locations,
    sharing the same set of latent variables (moment tensor + variance)
    across all fixed locations.

    Parameters:
    - fixed_locations: list of tuples [(lat,lon)...]
    - num_samples: number of moment tensor and variance samples per location
    - generate_moment_tensor_func: function to generate a moment tensor
    - depths: list of integers to indicate depth in meters of the seismic event 
    - sliprate_range: tuple (min_var, max_var)
    - MT_components (list of three positive float values) for the relative proportion of iso, clvd, and dc 
    components in the moment tensor. The proportion of each is simulated from a Dirichlet distribution. See 
    SeismicUtils.generate_moment_tensor()

    Returns:
    - DataFrame with input features, max mag and log of max mag
    """   

    # get distance in degrees from location to source, assumed to be at 0,0
    dists = []
    for i in fixed_locations:
        dist = compute_distance_from_source(i)
        dists.append(dist)
    # get azimuth
    angles = compute_azimuths_from_source(fixed_locations)

    # generate latent variables that we want to marignalize over: sliprate variances and moment tensors
    latent_samples = []
    for _ in range(num_samples):
        moment_tensor = generate_moment_tensor(MT_components=MT_components)
        gaussian_variance = np.random.uniform(*sliprate_range)
        #gaussian_variance = 2
        latent_samples.append((moment_tensor, gaussian_variance))
    
    data = []
    for i in range(len(fixed_locations)):
        
        my_azimuth = angles[i]
        theta = np.radians(my_azimuth)
        distance_in_degrees = dists[i]
           
        for moment_tensor, gaussian_variance in latent_samples:
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            M_rotated = R @ moment_tensor @ R.T
            my_moment_tensor = M_rotated

                
            seis_obj = Seismogram()
            seis_obj.depth_in_m = depth
            seis_obj.dist_in_degrees = distance_in_degrees
            seis_obj.sliprate = "Gauss"
            seis_obj.moment_tensor = my_moment_tensor

            seis_obj.custom_seismogram(variance=gaussian_variance)
            seis_obj.get_response(resp="VEL")
            mag = seis_obj.compute_integrated_signal_power()
            #mag = seis_obj.compute_maximum_magnitude(data_type="VEL")
    
            # store input dataset
            data.append({
                "Depth": depth,
                "Distance_in_degrees": distance_in_degrees,
                "Gaussian_variance": gaussian_variance,
                "m_rr": my_moment_tensor[0, 0],
                "m_tt": my_moment_tensor[1, 1],
                "m_pp": my_moment_tensor[2, 2],
                "m_rt": my_moment_tensor[0, 1],
                "m_rp": my_moment_tensor[0, 2],
                "m_tp": my_moment_tensor[1, 2],
                "max_mag": mag,
                "log_max_mag": np.log(mag)
            })
    data = pd.DataFrame(data)
    lat, lon = repeat_latlon_pairs(fixed_locations, num_samples)
    data['lat'] = lat
    data['lon'] = lon
    return pd.DataFrame(data)




def make_augmented_df(num_reps=30, num_azimuths=20, num_distances=10):
    '''
    This function sets up the skeleton for making an augmented GP dataset where moment tensors
    are rotated according to azimuth
    Parameters
    ----------
    num_reps: number of unique depths and Gaussian sliprate variances to sample
    num_azimuths: number of azimuths per fixed location
    num_distances: number of distances per depth

    Returns
    -----------
    a pandas DataFrame with rows equal to num_reps * num_azimuths * num_distances
    '''
    
    # Randomly sample depths
    depths = np.random.uniform(1000, 40000, num_reps)
    
    # Define azimuths
    azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
    
    # Randomly sample distances in degrees to source
    distances = np.linspace(0,3, num_distances)
    
    variances = np.random.uniform(1, 3, num_reps)
    
    # Create a DataFrame with all combinations
    depths_repeated = np.repeat(depths, len(azimuths) * len(distances))
    azimuths_repeated = np.tile(np.repeat(azimuths, len(distances)), len(depths))
    distances_repeated = np.tile(distances, len(depths) * len(azimuths))
    variances_repeated = np.repeat(variances, len(azimuths) * len(distances))
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Depth (m)': depths_repeated,
        'Azimuth (degrees)': azimuths_repeated,
        'Distance (degrees)': distances_repeated,
        'Gaussian_variance': variances_repeated
    })

    return df

def calculate_receiver_location_geopy(lat, lon, azimuth, distance_degrees):
    # Create a point for the source location
    source_point = Point(lat, lon)
    
    # Convert distance from degrees to kilometers
    distance_kilometers = distance_degrees * 111.32  # Convert degrees to kilometers
    
    # Calculate the destination point using geodesic
    destination_point = geodesic(kilometers=distance_kilometers).destination(source_point, azimuth)
    
    return destination_point.latitude, destination_point.longitude