import numpy as np
from tools import unwrap


def B0Map(pixel_array, echo_list):
    """
    Generates a B0 map from a series of volumes collected
    with 2 different echo times.

    Parameters
    ----------
    pixel_array : 4D/3D array
        A 4D/3D array containing the signal from each voxel at each
        echo time i.e. the dimensions of the array are
        [x, y, TE] or [x, y, z, TE].
    echo_list : list()
        An array of the echo times used for the last dimension of the
        raw data.

    Returns
    -------
    b0 : 2D/3D array
        An array containing the B0 map generated by the method.
        If pixel_array is 4D, then B0 map will be 3D.
        If pixel_array is 3D, then B0 map will be 2D.
    """

    try:
        if len(echo_list) > 1:
            # Is the given array already a Phase Difference or not?
            phase_diff_original = (np.squeeze(pixel_array[..., 0])
                                   - np.squeeze(pixel_array[..., 1]))
            deltaTE = np.absolute(echo_list[1] - echo_list[0]) * 0.001
        else:
            # If it's a Phase Difference / B0 Map, it just unwraps the phase
            # and returns the output
            b0 = unwrap(pixel_array)
            return b0
        # Normalise to -2Pi and +2Pi
        phase_diff = (phase_diff_original / ((1 / (2 * np.pi))
                      * np.amax(phase_diff_original)
                      * np.ones(np.shape(phase_diff_original))))
        # Mathematical expression: B0 = (P1-P2)/(2*pi*deltaTE)
        b0 = (unwrap(phase_diff) / ((2 * np.pi * deltaTE)
                                    * np.ones(np.shape(phase_diff))))
        del phase_diff_original, phase_diff, deltaTE
        return b0
    except Exception as e:
        print('Error in function B0Map.B0Map: ' + str(e))
