import scipy.spatial as spatial
from scipy.signal import fftconvolve
import scipy
import numpy as np

""" Set of functions that are useful for processing movement tracks.

"""

# def fill_in_nans(tracks):
#     """Replace nans in tracks with reasonable numbers 
        
#     Replace nans with closest value if nans at beginning 
#     or end of observation. If in the middle, interpolate 
#     between sandwiching real values.
        
#     return array
#     """
        
#     # As nans are replaced with real values this will be updated
#     val_mask = ~np.isnan(tracks[...,0])
#     # As real value blocks are used they will be set here to False
#     unseen_mask = ~np.isnan(tracks[...,0])

#     tracks = np.copy(tracks)

#     no_nans = np.all(~np.isnan(tracks))

#     while not no_nans:

#         # frame of first nan in every track
#         # (if no nans, then argmin is 0)
#         mininds = np.argmin(val_mask, 1)

#         for track_ind, minval in enumerate(mininds):
#             if minval == 0 and not np.isnan(tracks[track_ind][0, 0]):
#                  # the first frame isn't actually nan, 
#                  # whole track must have real values
#                 minval = None
#             # want to find the first real value after nan block
#             # set all places before first nan to false
#             unseen_mask[track_ind][:minval] = False
#             # frame of first real value after first nan block
#             maxinds = np.argmax(unseen_mask, 1)

#         for track_ind, (minval, maxval) in enumerate(zip(mininds, maxinds)):
#             if minval != maxval:
#                 if minval > maxval:
#                     # nans to the end
#                     # take value from last real value
#                     maxval = None
#                     new_x = tracks[track_ind][minval-1, 0]
#                     new_y = tracks[track_ind][minval-1, 1]
#                 elif minval == 0:
#                     # begins with nans
#                     # take value from first real value
#                     new_x = tracks[track_ind][maxval, 0]
#                     new_y = tracks[track_ind][maxval, 1]
#                 else:
#                     # middle segment of nans
#                     # interpolate between sandwiching real values
#                     new_x = np.linspace(tracks[track_ind][minval-1, 0],
#                                         tracks[track_ind][maxval, 0],
#                                         maxval-minval)
#                     new_y = np.linspace(tracks[track_ind][minval-1, 1],
#                                         tracks[track_ind][maxval, 1],
#                                         maxval-minval)

#                 tracks[track_ind][minval:maxval, 0] = new_x
#                 tracks[track_ind][minval:maxval, 1] = new_y
#                 # update places where nans have been replaced with real values
#                 val_mask[track_ind][minval:maxval] = True
#         # Check if all nans have been replaced yet
#         no_nans = np.all(~np.isnan(tracks))
        
#     return tracks


def fill_in_nans(tracks):
    """Replace nans in tracks with reasonable numbers 
        
    Replace nans with closest value if nans at beginning 
    or end of observation. If in the middle, interpolate 
    between sandwiching real values.
    
    tracks: array of shape (num tracks, num frames)
            or shape (num_tracks, num_frame, num spatial dims)
        
    return array
    """
    is_2d = False
    
    if len(tracks.shape) == 2:
        # Make tracks 3d to make caclulations consistent
        tracks = np.expand_dims(tracks, 2)
        is_2d = True
        
    assert len(tracks.shape) == 3, "Tracks must have 2 or 3 dimensions."
    
    # Tracks that still have nan gaps
    still_nans = np.array([True for _ in range(len(tracks))])
    
    # As nans are replaced with real values this will be updated
    val_mask = ~np.isnan(tracks[still_nans,:,0])
    # As real value blocks are used they will be set here to False
    unseen_mask = ~np.isnan(tracks[still_nans,:,0])

    tracks = np.copy(tracks)
    no_nans = np.all(~np.isnan(tracks))
    
    while not no_nans:    
        
        # frame of first nan in every track
        # (if no nans, then argmin is 0)
        mininds = np.argmin(val_mask[still_nans], 1)

        for track_ind, minval in enumerate(mininds):
            
            if not still_nans[np.argwhere(still_nans)[track_ind][0]]:
                continue
            
            
            if (minval == 0 
                and not np.isnan(
                    tracks[np.argwhere(still_nans)[track_ind][0]][0, 0])):
                 # the first frame isn't actually nan, 
                 # whole track must have real values
                minval = None
                
            # want to find the first real value after nan block
            # set all places before first nan to false
            unseen_mask[np.argwhere(still_nans)[track_ind][0]][:minval] = False
            # frame of first real value after first nan block
        maxinds = np.argmax(unseen_mask[still_nans], 1)
        
        # old still nan can't change in middle of for loop or indexing gets messed up
        next_still_nans = np.copy(still_nans)
            
        for track_ind, (minval, maxval) in enumerate(zip(mininds, maxinds)):
            if not still_nans[np.argwhere(still_nans)[track_ind][0]]:
                continue
            if minval != maxval:
                # values that will replace nans
                new_vals = []
                if minval > maxval:
                    # nans to the end
                    # take value from last real value
                    maxval = None
                    for dim in range(tracks.shape[2]):
                        new_vals.append(tracks[np.argwhere(still_nans)[track_ind][0]][minval-1, dim])
                elif minval == 0:
                    # begins with nans
                    # take value from first real value
                    for dim in range(tracks.shape[2]):
                        new_vals.append(tracks[np.argwhere(still_nans)[track_ind][0]][maxval, dim])
                else:
                    # middle segment of nans
                    # interpolate between sandwiching real values
                    for dim in range(tracks.shape[2]):
                        new_vals.append(np.linspace(tracks[np.argwhere(still_nans)[track_ind][0]][minval-1, dim],
                                        tracks[np.argwhere(still_nans)[track_ind][0]][maxval, dim],
                                        maxval-minval))
                
                for dim in range(tracks.shape[2]):
                    tracks[np.argwhere(still_nans)[track_ind][0]][minval:maxval, dim] = new_vals[dim]

                # update places where nans have been replaced with real values
                val_mask[np.argwhere(still_nans)[track_ind][0]][minval:maxval] = True
            else:
                if np.all(np.isnan(tracks[np.argwhere(still_nans)[track_ind][0]])):
                    # Whole track has nans
                    # Just set all values to 0
                    tracks[np.argwhere(still_nans)[track_ind][0]] = np.zeros_like(
                        tracks[np.argwhere(still_nans)[track_ind][0]])
                
                
                
                next_still_nans[np.argwhere(still_nans)[track_ind][0]] = False

        # Check if all nans have been replaced yet
        no_nans = np.all(~np.isnan(tracks))
        
        still_nans = next_still_nans
        
    if is_2d:
        tracks = np.squeeze(tracks)
        
    return tracks


def smooth_tracks(raw_tracks, kernel_size=31):
    """
    sliding window averaging over tracks. 
        
    pad with first and last value. before filtering replace nans with closest
    value if nans at beginning or end of observation. If in the middle, 
    interpolate between sandwiching real values. Insert nans back after 
    filtering.
    
    raw_tracks: array of shape (num tracks, num steps, num dimensions)
        
    kernel_size: averaging window size (only tested with odd kernels)
    override_tracks: smooth tracks variable or make new set of tracks
    """
        
    if kernel_size % 2 == 0:
        print('please use odd kernel size')
        return False
        

    tracks = np.copy(raw_tracks)
            
    filled_tracks = fill_in_nans(tracks)

    if len(raw_tracks.shape) == 3:
        padded_tracks = np.pad(
            filled_tracks, 
            ((0,0),(int(kernel_size/2), int(kernel_size/2)), (0,0)), 
            'edge')
        
    elif len(raw_tracks.shape) == 2:
        padded_tracks = np.pad(
            filled_tracks, 
            ((0,0),(int(kernel_size/2), int(kernel_size/2))), 
            'edge')
        
    else:
        "Error: Tracks must have dimension 2 or 3."
        return

    if len(raw_tracks.shape) == 3:
        smooth_filter = np.ones((tracks.shape[0], kernel_size, 2)) / kernel_size
        
    if len(raw_tracks.shape) == 2:
        smooth_filter = np.ones((tracks.shape[0], kernel_size)) / kernel_size

    tracks_smooth = fftconvolve(padded_tracks, smooth_filter, mode='same', axes=1)
    # Get rid of padding
    tracks_smooth = tracks_smooth[:, int(kernel_size/2):-int(kernel_size/2)]
        
    assert tracks_smooth.shape == raw_tracks.shape
        
    # put the nans back in tracks
    smooth_tracks = np.where(~np.isnan(tracks), tracks_smooth, np.nan)

    return smooth_tracks
            
def median_filter_tracks(raw_tracks, kernel=[1, 21, 1]):
    """
    median filtering over positions. 
    
    pad with first and last value. before filtering replace nans with closest
    value if nans at beginning or end of observation. If in the middle, 
    interpolate between sandwiching real values. Insert nans back after 
    filtering.
        
    raw_tracks: array of shape (num tracks, num steps, num dimensions)    
        
    kernel_size: filter window size (only tested with odd kernels)
    override_tracks: smooth tracks variable or make new set of tracks
        
    use_positions: if True filter postions instead of tracks
    """
        
    kernel_size = max(kernel)                     
                         
    if kernel_size % 2 == 0:
        print('please use odd kernel size')
        return False


    tracks = np.copy(raw_tracks)
      
    filled_tracks = fill_in_nans(tracks)

    if len(raw_tracks.shape) == 3:
        padded_tracks = np.pad(
            filled_tracks, 
            ((0,0),(int(kernel_size/2), int(kernel_size/2)), (0,0)), 
            'edge')
        
    elif len(raw_tracks.shape) == 2:
        padded_tracks = np.pad(
            filled_tracks, 
            ((0,0),(int(kernel_size/2), int(kernel_size/2))), 
            'edge')
        
    else:
        "Error: Tracks must have dimension 2 or 3."
        return

    # Filter over positions, in seperate x, y, not across individuals
    tracks_filtered = scipy.signal.medfilt(padded_tracks, kernel)
    # Get rid of padding
    tracks_filtered = tracks_filtered[:, int(kernel_size/2):-int(kernel_size/2)]
        
    assert tracks_filtered.shape == raw_tracks.shape

    # put the nans back in tracks
    filtered_tracks = np.where(~np.isnan(tracks), tracks_filtered, np.nan)

    return filtered_tracks