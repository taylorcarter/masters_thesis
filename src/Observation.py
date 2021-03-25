import scipy.spatial as spatial
from scipy.signal import fftconvolve
import scipy
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# import sys
# workstation_functions = '/home/golden/coding/drone-tracking/code/functions'
# sys.path.append(workstation_functions)

import koger_track_functions as ktf


class Observation():
    """
    Contains all the relevant data/info about a single observation
    """
    
    def __init__(self, observation_name, tracks=None, body_points=None, 
                 postures=None, posture_scores=None, tracks_meta=None, 
                 obs_meta=None, target_species=None, focal_species=None, ):
        """
        observation_name: string
        tracks: (num-tracks, num-frames, 2) np array
        body_points: (num tracks, num frames, 3, 2), 
            utm xy of track location, 
            average position of neck fron two shoulders
            average position of top tail, back haunches
        tracks_meta: pandas dataframe with info about each track 
                     col_names: 'raw_track_id', 'track_id', 'species'
        obs_meta: pandas dataframe with info about observation in general
                  observation_name: observation_name
                  first_main_observation_frame: when main group found
                  scare_frame: scare frame
                  big_map: name of big map
        target_species: list of strings of species names to use
        focal_species: species of animal in main group
        """
        
        self.name = observation_name
        # Raw tracks won't be touched
        self.raw_tracks = tracks
        # Tracks will be modified as user wants
        self.tracks = tracks
        self.posititions = tracks
        self.raw_body_points = body_points
        self.body_points = body_points
        self.postures = postures
        self.posture_scores = posture_scores
        self.tracks_meta = tracks_meta
        self.obs_meta = obs_meta 
        self.target_species = target_species
        # list that includes all other  tracks added to focal track
        self.added_tracks = [[] for track_ind in range(len(self.raw_tracks))]
        # maps track index to raw_track index
        self.tracks_to_raw_tracks = np.arange(len(self.raw_tracks))
        
        self.first_frame = self.obs_meta['first_main_observation_frame'][0]
        
        self.total_frames = self.tracks.shape[1]

        self.focal_species = focal_species
        self.scare_frame = obs_meta['scare_frame'].values[0]
        if self.scare_frame == 'none_set':
            self.scare_frame = np.nan


        self._calculate_average_tracks_per_frame()
        
        
    def combine_and_delete_tracks(self):

        """
        using info in track_meta file combine raw_tracks that belong to same individual
        and remove tracks that don't belong to target species

        return False and warning if tracks_meta not present
        return False and warning if target species not present
        """
        
        if self.tracks_meta is None:
            print('Warning no tracks_meta for {}'.format(self.name))
            print('Returning False...')
            return False
        
        if self.target_species is None:
            print('Warning no target_species list for {}'.format(self.name))
            print('Returning False...')
            return False
        
        tracks_to_keep_mask = np.array([True for _ in range(len(self.raw_tracks))])
        for raw_track_id, track_id in enumerate(list(self.tracks_meta['track_id'])):
            if raw_track_id != track_id:
                # This track is part of another track
                # Track_id is the original track, raw_track_id is being added to it
                if self.tracks is not None:
                    self.tracks[track_id] = np.where(np.isnan(self.tracks[track_id]), 
                                                     self.tracks[raw_track_id], 
                                                     self.tracks[track_id])
                if self.body_points is not None:
                    self.body_points[track_id] = np.where(
                        np.isnan(self.body_points[track_id]), 
                        self.body_points[raw_track_id], 
                        self.body_points[track_id])
                if self.postures is not None:
                    self.postures[track_id] = np.where(
                        np.isnan(self.postures[track_id]), 
                        self.postures[raw_track_id], 
                        self.postures[track_id])
                if self.posture_scores is not None:
                    self.posture_scores[track_id] = np.where(
                        np.isnan(self.posture_scores[track_id]), 
                        self.posture_scores[raw_track_id], 
                        self.posture_scores[track_id])
                # Eventually remove the track that is just the extra part
                tracks_to_keep_mask[raw_track_id] = False
                self.added_tracks[track_id].append(raw_track_id)
        # Also want to remove tracks that aren't the right species
        target_species_mask = self.tracks_meta['species'].isin(self.target_species).values
        tracks_to_keep_mask = np.logical_and(tracks_to_keep_mask, target_species_mask)
        self.tracks_to_raw_tracks = self.tracks_to_raw_tracks[tracks_to_keep_mask]

        if self.tracks is not None:
            self.tracks = self.tracks[tracks_to_keep_mask]
        if self.body_points is not None:
            self.body_points = self.body_points[tracks_to_keep_mask]
        if self.postures is not None:
            self.postures = self.postures[tracks_to_keep_mask]
        if self.posture_scores is not None:
            self.posture_scores = self.posture_scores[tracks_to_keep_mask]
        
        return True
    
    def filter_body_points(self, median_filter_size=5, smoothing_filter_size=11):
        """ Apply median filtering and then smoothing to body_points.
        
        median_filter_size: size of median filter kernel
        smoothing_filter_size: size of smoothing filter kernel
         
        """
        self.raw_body_points = np.copy(self.body_points)
        
        body_points = np.transpose(self.body_points, (0, 2, 1, 3))
        body_points = np.concatenate([t for t in body_points])
        body_points_f = ktf.median_filter_tracks(body_points, 
                                                 [1, median_filter_size, 1])
        body_points_f = ktf.smooth_tracks(body_points_f, 
                                          kernel_size=smoothing_filter_size)
        individual_points = []
        for ind in range(int(body_points_f.shape[0]/3)):
            individual_points.append(body_points_f[(ind*3):(ind*3)+3])
        body_points_f = np.stack(individual_points)
        body_points_f = np.transpose(body_points_f, (0,2,1,3))
        
        self.body_points = body_points_f
    
    def get_posture_based_positions(self):
        """ Get positions based on posture where available from tracks where not.

        When available take the center point between front and back of body.
        """
        
        # mean of front and back of animal, nan for ind center when at least one is nan
        posture_center = np.mean(self.body_points[:,:,1:], 2)
        # just use tracks when not enough posture info
        centers = np.where(np.isnan(posture_center), 
                          self.tracks, posture_center)
        self.positions = centers

        return centers
    
    
    def calculate_orientation_angles(self, should_filter=False,
                                     median_filter_size=5,
                                     smoothing_filter_size=11):
        """Get the angle from the back to the front of the animal
            
        Create a object variable named orientation
        
        should_filter: boolean should the calculated heading angle 
                       have meadian filtering applied
        median_filter_size: kernel size of median filter
        smoothing_filter_size: kernel size of smoothing filter
        """
        
        if self.body_points is None:
            self.orientations = None
            return
        
        
        x_diff = self.body_points[:, :, 1, 0] - self.body_points[:, :, 2, 0]
        y_diff = self.body_points[:, :, 1, 1] - self.body_points[:, :, 2, 1]

        self.orientations = np.arctan2(y_diff, x_diff)
        
        if should_filter:
            self.orientations_raw = np.copy(self.orientations)
            self.orientations = ktf.median_filter_tracks(self.orientations, 
                                                     kernel=[1, median_filter_size])
            self.orientations = ktf.smooth_tracks(self.orientations, 
                                              kernel_size=smoothing_filter_size)
        


    def _calculate_average_tracks_per_frame(self):
        """Calculate average number of individuals in across observation.

        Total number of active points across observation divided
        by number of frames in observation.
        """

        # Divide by 2 because counting x and y values
        num_real_value_track_frames = np.sum(~np.isnan(self.tracks[:,self.first_frame:])) / 2
        average_tracks_per_frame = (num_real_value_track_frames / 
                                    self.tracks[self.first_frame:].shape[1])
        self.average_tracks_per_frame = average_tracks_per_frame

        
    def calculate_steps(self, tracks=None):
        """
        Create array steps that contains step vectors from tracks
        same a velocity with units m/(1/30)s
        
        each step is vector pointing from position in current frame to position in next
        
        tracks: Tracks user wants to use to calculate step sizes. 
                If None, just use default tracks
                shape: (num tracks, num steps, dimemsions)
        """
        
        if tracks is None:
            tracks = self.tracks
        
        self.steps = np.zeros_like(tracks)
        #Since step is from current frame to next, last frame will have step (0,0)
        self.steps[:, :-1] = tracks[:, 1:] - tracks[:, :-1]
        
        return self.steps
        
    def calculate_step_sizes(self):
        """
        Create array step_sizes that contains magnitude of steps vector
        
        if self.steps does not exist, create steps vector
        """
        
        if not hasattr(self, 'steps'):
            print('creating steps array')
            self.calculate_steps()
            
        steps_squared = self.steps ** 2
        steps_size_squared = np.sum(steps_squared, 2)
        self.step_sizes = np.sqrt(steps_size_squared)
    
            
            
    def calculate_distance_between_individuals(self):
        """Calculate distance between all pairs of individuals in each frame.
        
        If centers have already been calculated, use those. Otherwise use tracks.
        """
        
        distances = []
        for frame_num in range(self.total_frames):
            distances.append(
                spatial.distance.cdist(
                    self.positions[:, frame_num], self.positions[:, frame_num]))

        self.distances = np.stack(distances, 0)
        
    
    def calculate_local_pair_distance_extremes(self, distance_thresh):
        """Get frames where pairs are local minimum or maximum distance apart.
        
        Stores info in a (num_tracks, num_tracks) shaped array of dicts 
        with keys 'mins' and 'maxs'
        
        distance_thresh: how many meters away must a local min be from 
                         a local max to be considered more than just noise 
        
        """
        
        if not hasattr(self, 'distances'):
            print('creating distances array')
            self.calculate_distance_between_individuals()

        extreme_ds = np.zeros((self.tracks.shape[0], self.tracks.shape[0]), dtype=object)

        for focal_ind in range(self.tracks.shape[0]):
            for other_ind in range(focal_ind, self.tracks.shape[0]):

                # dict to store local maxs and mins for particular pair
                extremes = {'maxs': [], 'mins': []}

                if focal_ind == other_ind:
                    continue


                first_real_valued_frame = np.argmax(
                    ~np.isnan(self.distances[:,focal_ind, other_ind]))

                last_min = self.distances[first_real_valued_frame, focal_ind, other_ind]
                last_max = self.distances[first_real_valued_frame, focal_ind, other_ind]

                curr_min_diff = -distance_thresh
                curr_max_diff = distance_thresh

                curr_max_frame = first_real_valued_frame
                curr_min_frame = first_real_valued_frame

                min_thresh_hit = True
                max_thresh_hit = True

                for frame, distance in enumerate(self.distances[:,focal_ind, other_ind]):
                    if np.isnan(distance):
                        continue
                    difference = distance - last_min

                    if (difference > curr_max_diff):
                        curr_max_diff = difference
                        curr_max_frame = frame
                        max_thresh_hit = True
                        last_max = self.distances[frame, focal_ind, other_ind]
                        if min_thresh_hit:
                            # The last min was a real local min
                            extremes['mins'].append(curr_min_frame)
                            curr_min_diff = -distance_thresh
                            min_thresh_hit = False

                    difference = distance - last_max 

                    if (difference < curr_min_diff):
                        curr_min_diff = difference
                        curr_min_frame = frame
                        min_thresh_hit = True
                        last_min = self.distances[frame, focal_ind, other_ind]
                        if max_thresh_hit:
                            # The last min was a real local min
                            extremes['maxs'].append(curr_max_frame)
                            curr_max_diff = distance_thresh
                            max_thresh_hit = False
                          
                
                if min_thresh_hit:
                    # The last min was a real local min
                    extremes['mins'].append(curr_min_frame)
                
                if max_thresh_hit:    
                    # The last min was a real local min
                    extremes['maxs'].append(curr_max_frame)

                extreme_ds[focal_ind, other_ind] = extremes
                extreme_ds[other_ind, focal_ind] = extremes
        
        self.extreme_ds = extreme_ds
        
    def _calc_distance_between_frames(self, ind, frame0, frame1):
        """ Calculate distance between two frames of one track.
        
        Parameters:
        ind: int, track index of track want to look at
        frame0: int, frame of first point in track want to look at
        frame1: int, frame of the second point in track 
        """
        
        distance_vec = self.tracks[ind][frame1] - self.tracks[ind][frame0]
        distance = np.sqrt(np.sum(distance_vec**2))
        return distance
    
    def calculate_disparity(self, focal_d0, focal_d1, other_d0, other_d1):
        """Calculate disparity of potential pull/anchor event
        
        Parameters:
        focal_d0: distance focal individual moved from min0 to max
        focal_d1: distance focal individual moved from max to min1
        other_d0: distance other individual moved from min0 to max
        other_d1: distance other individual moved from max to min1
        """

        disparity = ((np.abs(focal_d0 - other_d0) * np.abs(focal_d1 - other_d1)) /
                     (np.abs(focal_d0 + other_d0) * np.abs(focal_d1 + other_d1)))

        return disparity
    
    def calculate_strength(self, focal_ind, other_ind, event_frames):
        """ Calculates strength of potential pull/anchor event
        
        Parameters:
        
        focal_ind: track number of focal track
        other_ind: tack number of other track
        event_frames: list, contains frame number of min0, max, min1
        """
        
        distance0 = self.distances[event_frames[0]][focal_ind, other_ind]
        distance1 = self.distances[event_frames[1]][focal_ind, other_ind]
        distance2 = self.distances[event_frames[2]][focal_ind, other_ind]

        strength = (np.abs(distance1 - distance0) * np.abs(distance2 - distance1) / 
                   (np.abs(distance1 + distance0) * np.abs(distance2 + distance1)))

        return strength
    
    def calculate_event_type(self, focal_d0, focal_d1, other_d0, other_d1):
        """Calculate what kind of event possible push/pull is.
        
        if focal moves more in both parts anchor
        if focal moves more in first part, pull
        if other moves more in first part, None
        
        Parameters:
        focal_d0: distance focal individual moved from min0 to max
        focal_d1: distance focal individual moved from max to min1
        other_d0: distance other individual moved from min0 to max
        other_d1: distance other individual moved from max to min1
        """
        
        event_type = 'None'

        if (focal_d0 > other_d0):
            if (focal_d1 > other_d1):
                event_type = 'anchor'
            else:
                event_type = 'pull'

        return event_type
    
    def process_push_anchor_event(self, focal_ind, other_ind, event_frames):
    
        """ See if potential push/anchor event actually is and return results.
    
        Parameters:
        
        focal_ind: track number of focal track
        other_ind: tack number of other track
        event_frames: list, contains frame number of min0, max, min1
        """
    
    
        focal_d0 = self._calc_distance_between_frames(focal_ind, event_frames[0], event_frames[1])
        focal_d1 = self._calc_distance_between_frames(focal_ind, event_frames[1], event_frames[2])

        other_d0 = self._calc_distance_between_frames(other_ind, event_frames[0], event_frames[1])
        other_d1 = self._calc_distance_between_frames(other_ind, event_frames[1], event_frames[2])

        disparity = self.calculate_disparity(focal_d0, focal_d1, other_d0, other_d1)
        strength = self.calculate_strength(focal_ind, other_ind, event_frames)

        event_type = self.calculate_event_type(focal_d0, focal_d1, other_d0, other_d1)

        event = {'leader': focal_ind, 'follower': other_ind, 
             'disparity': disparity, 'strength': strength, 
             'min0': event_frames[0], 'max': event_frames[1], 
             'min1': event_frames[2], 'type': event_type}

        return event


    def combine_pull_anchor_events(self, strength_thresh=.1, disparity_thresh=.1):
        """Combine pull anchor events as described in paper."""

        
        true_event_mask = ((self.raw_pull_anchor_df['strength'] > strength_thresh) & 
                           (self.raw_pull_anchor_df['disparity'] > disparity_thresh) & 
                           ~(self.raw_pull_anchor_df['type'] == 'None'))
        df = self.raw_pull_anchor_df.loc[true_event_mask].copy()
        df['event'] = -1
        df = df.sort_values(by='min0').reset_index()

        event_num = 0
                    
        for row in range(df.shape[0]):
            if df.loc[row, 'event'] == -1:


                # Of events that could be part of event
                # when does follower start moving to one of them
                canidate_mask = ((df['min0'] <= df.loc[row, 'max']) & 
                                 (df['min0'] >= df.loc[row, 'min0']) & 
                                 (df['follower'] == df.loc[row, 'follower']))
                max_event_frame = np.min(df.loc[canidate_mask, 'max'].values)

                event_mask = ((df['min0'] <= max_event_frame) & 
                              (df['min0'] >= df.loc[row, 'min0']) & 
                              (df['follower'] == df.loc[row, 'follower']))
                df.loc[event_mask, 'event'] = event_num

                event_num += 1

        self.pull_anchor_events = df
    
    def find_pull_anchor_events(self):
        """Find all pull/anchor events and store in dataframe."""
        
        events = []
        
        for focal_ind in range(self.tracks.shape[0]):
            for other_ind in range(self.tracks.shape[0]):
                if focal_ind == other_ind:
                    continue
                    
                if not hasattr(self, 'extreme_ds'):
                    print('creating extreme distances array')
                    self.calculate_local_pair_distance_extremes(distance_thresh=3)
                # Frames where there are local min or max distances between given pair
                # of individuals
                extremes = self.extreme_ds[focal_ind, other_ind]
                    
                if len(extremes['mins']) == 1:
                    # only one lcoal minimum: not enough for pull/anchor
                    continue

                for event_num in range(len(extremes['mins']) - 1):
                    # -1: last min can't start new pull/anchor

                    min0 = extremes['mins'][event_num]
                    # first max after min0 frame
                    max0 = extremes['maxs'][np.argmax(np.array(extremes['maxs']) > min0)]
                    min1 = extremes['mins'][event_num + 1]

                    event_frames = [min0, max0, min1]

                    event = self.process_push_anchor_event(focal_ind, other_ind, event_frames)

                    events.append(event)
   
        events_df = pd.DataFrame(events)
        self.raw_pull_anchor_df = events_df
        
        
    def calculate_group_polarization(self):
        """ Calculate group polarizations based on individual orientations.
        
        If no active individuals in frame the result is nan for that frame.
        """
        
        if not hasattr(self, 'orientations'):
            print('"calculate_group_polarizations" failed')
            print('first calculate individual orientations')
            print('use "calculate_orientation_angles."')
            return None
        orientations = self.orientations
        x_orientation = np.nansum(np.cos(orientations), 0) 
        y_orientation = np.nansum(np.sin(orientations), 0)
        orientation_mag = np.sqrt(x_orientation**2 + y_orientation**2)
        num_individuals = np.nansum(~np.isnan(orientations), 0)
        group_polarization = np.full(orientations.shape[1], np.nan, dtype=float)

        group_polarization[num_individuals != 0] = (
            orientation_mag[num_individuals != 0] / 
            num_individuals[num_individuals != 0])
        
        self.group_polarization = group_polarization
        
        return group_polarization
    
    def calculate_dot_products(self):
        """ Calculate dot product between orientations of each individual in frames.
        """
        
        if hasattr(self, 'orientations'):
            diff = (np.expand_dims(self.orientations, 1) - self.orientations)
            dots = np.cos(diff)
            # make array (frames, inds, inds) instead or (inds, inds, frames)
            self.dots = dots.swapaxes(0, 2)
            return self.dots
        
        else:
            print('No orienations. Run ".calculate_orientation_angles"')
            return False
        
    
    def plot_individual(self, individual, obs_ind):
        """ Show given individual in given frame.

        If has body points show front (red) and back (blue).
        Otherwise just show the track position.

        Input:
        individual: number of individual to show
        obs_ind: observation index to show
        """

        if hasattr(self, 'body_points'):
            plt.scatter(self.body_points[individual, obs_ind, 1, 0],
                        self.body_points[individual, obs_ind, 1, 1], c='r')
            plt.scatter(self.body_points[individual, obs_ind, 2, 0],
                        self.body_points[individual, obs_ind, 2, 1], c='b')
        elif hasattr(self, 'tracks'):
            plt.scatter(self.tracks[individual, obs_ind, 1, 0],
                        self.tracks[individual, obs_ind, 1, 1], c='k')
        else:
            print('There are no body points or tracks to show.')

        plt.gca().set_aspect('equal')