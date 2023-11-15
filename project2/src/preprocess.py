import autograd.numpy as np 

def center(data, axis=0):
    r'''
    Center the data by subtracting the mean and dividing by the standard
    deviation.

    ``centered_data = (data - np.mean(data)) / np.std(data)``

    Parameters
    ----------
    data : array_like
        Data to be centered. 
    
    axis : int, default: 0
        Which axis to center. If ``data`` is 2x2, ``axis=0`` results in
        the data being centered along the columns.

    Returns
    -------
    array_like :
        The centered data along the given axis.
    '''
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)

def norm_data_zero_one(data):
    r'''
    Normalize the data to range between 0 and 1.

    Parameters
    ----------
    data : array_like
        Data to be normalized.

    Returns
    -------
    array_like :
        Data mapped to range between 0 and 1.
    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))