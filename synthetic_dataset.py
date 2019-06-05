import numpy as np 
import random

#########################
### Utility Functions ###
#########################

def _get_samples_per_mode(samples, modes, sample_weights):
    ### Weight number of samples per mode
    if sample_weights:
        sample_weights = [float(w) / sum(sample_weights) for w in sample_weights] # Normalize weights to sum to 1
        samples_per_mode = [int(samples*w) for w in sample_weights]
    else:
        samples_per_mode = [int(np.ceil(samples / modes)) for i in range(modes)] # Equal weighting

    return samples_per_mode

def _get_covariance(variance):
    # Calculate covariance matrix
    if type(variance) is list:
        cov = np.diagflat(variance)
    else:
        cov = np.diagflat([variance, variance])

    return cov

def _sample(samples, means, cov, samples_per_mode, random_state):
    np.random.seed(random_state)
    
    ### Sample at each mode
    data = []
    for i, mean in enumerate(means):
        mode_pts = np.random.multivariate_normal(mean, cov, samples_per_mode[i]).T 
        mode_pts = [[x,y] for x,y in zip(mode_pts[0], mode_pts[1])]
        data.extend(mode_pts)
    
    ### Fix data size from weighted sample rounding error
    if len(data) != samples:
        for i in range(len(data) - samples):
            data.remove(random.choice(data)) # Remove random sample

    return np.array(data)

##########################
### Synthetic Datasets ###
##########################

def GridGaussianDataset(rows=5, cols=5, grid_width=10, grid_height=10, variance=0.0025, samples=10000, sample_weights=None, random_state=None):
    """ Generate a Gaussian Grid dataset.

    Parameters
    ----------
    rows : int, optional (default=5)
    cols : int, optional (default=5)
        The number of rows/cols in the gaussian grid. Must be >1.
        ie. rows=cols=5 produces a grid of 25 evenly spaced 2D gaussian distributions.

    grid_width : int, optional (default=10)
    grid_height : int, optional (default=10)
        The height/width of the data space domain centred around (0,0).
        ie. grid_width=grid_height=10 produces the grid in the domain x∈[-5,5] y∈[-5,5].

    variance : float or list of floats of length 2, optional (default=0.0025)
        The variance of the gaussian distribution.
        If given a single float then both the x and y variance will use that value.
        If given a list of floats then the x and y variance will use the first and second values respectively. 
    
    samples : int, optional (default=10000)
        The total number of samples produced.

    sample_weights : list of floats of length cols*rows or None, optional (default=None)
        The proportion of samples drawn from each gaussian in column-row order (starting with top-left distribution).
        If None then all distributions receive equal weighting. 

    random_state : int or None, optional (default=None)
        Determines the RNG for the data sampling. Use for reproducible outputs.
        If None then output is random each function call.

    Returns
    -------
    data : array of shape [samples, 2]
        The data points.
    """

    # Input exceptions
    if rows<2:
        raise ValueError("Invalid number of rows. Rows must be >1.")
    if cols<2:
        raise ValueError("Invalid number of cols. Cols must be >1.")
    if type(variance) is list and len(variance) != 2:
        raise ValueError("Incorrect variance length. Should be a single scalar or list of length 2.")
    if sample_weights and len(sample_weights) != rows*cols:
        raise ValueError("Incorrect number of sample weights. Should be list of length 'rows*cols'")

    # Calculate grid means
    x_min = 0 - grid_width/2
    x_max = 0 + grid_width/2
    y_min = 0 - grid_height/2
    y_max = 0 + grid_height/2
    x_step = grid_width / (cols-1) 
    y_step = grid_height / (rows-1) 

    means = np.mgrid[x_min:(x_max+0.1):x_step, y_min:(y_max+0.1):y_step].reshape(2,-1).T
    means = sorted(means, key=lambda x: (x[0], -x[1]), reverse=False)

    # Sample at each mode
    samples_per_mode = _get_samples_per_mode(samples, rows*cols, sample_weights)
    cov = _get_covariance(variance)
    data = _sample(samples, means, cov, samples_per_mode, random_state) 

    return data


def CircularGaussianDataSet(modes=8, radius=5, variance=0.0025, samples=10000, sample_weights=None, random_state=None):
    """ Generate a Circular Gaussian dataset.

    Parameters
    ----------
    modes : int, optional (default=8)
        The number of distributions.

    radius : int, optional (default=5)
        The radius of the circle centred around (0,0).

    variance : float or list of floats of length 2, optional (default=0.0025)
        The variance of the gaussian distribution.
        If given a single float then both the x and y variance will use that value.
        If given a list of floats then the x and y variance will use the first and second values respectively. 
    
    samples : int, optional (default=10000)
        The total number of samples produced.

    sample_weights : list of floats of length cols*rows or None, optional (default=None)
        The proportion of samples drawn from each gaussian in counter-clockwise order (starting with the north-most distribution).
        If None then all distributions receive equal weighting. 

    random_state : int or None, optional (default=None)
        Determines the RNG for the data sampling. Use for reproducible outputs.
        If None then output is random each function call.

    Returns
    -------
    data : array of shape [samples, 2]
        The data points.
    """

    # Input exceptions
    if type(variance) is list and len(variance) != 2:
        raise ValueError("Incorrect variance length. Should be a single scalar or list of length 2.")
    if sample_weights and len(sample_weights) != modes:
        raise ValueError("Incorrect number of sample weights. Should be list of length 'modes'")

    # Calculate circle means
    means = []
    theta = (np.pi*2) / modes
    for i in range(modes):
        angle = theta*(i+1)
        x = radius*np.cos(angle)
        y = radius*np.sin(angle)
        means.append([x,y])

    # Sample at each mode
    samples_per_mode = _get_samples_per_mode(samples, modes, sample_weights)
    cov = _get_covariance(variance)
    data = _sample(samples, means, cov, samples_per_mode, random_state)

    return data

def ArchimedeanSpiralDataSet(revolutions=2, scale=1, variance=0.0025, samples=10000, random_state=None):
    """ Generate a Archimedean Spiral dataset.

    Parameters
    ----------
    revolutions : int, optional (default=2)
        The number of revolutions of the spiral.

    scale : float, optional (default=1)
        A scaling factor to control the radius of each revolution.

    variance : float or list of floats of length 2, optional (default=0.0025)
        The variance of the gaussian distribution.
        If given a single float then both the x and y variance will use that value.
        If given a list of floats then the x and y variance will use the first and second values respectively. 
    
    samples : int, optional (default=10000)
        The total number of samples produced.

    random_state : int or None, optional (default=None)
        Determines the RNG for the data sampling. Use for reproducible outputs.
        If None then output is random each function call.

    Returns
    -------
    data : array of shape [samples, 2]
        The data points.
    """

    # Input exceptions
    if type(variance) is list and len(variance) != 2:
        raise ValueError("Incorrect variance length. Should be a single scalar or list of length 2.")

    # Calculate spiral means
    means = []
    pts = 2000
    ls = np.linspace(0,1,pts+1)
    ls = np.delete(ls, 0) # Remhttp://192.168.0.12/ove ls[0]=0
    for i in ls:
        theta = 2*revolutions*np.pi*np.sqrt(i)
        r = (scale/2)*theta 
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        means.append([x,y])

    samples_per_mode = _get_samples_per_mode(samples, pts, None)
    cov = _get_covariance(variance)
    data = _sample(samples, means, cov, samples_per_mode, random_state)

    return data