#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import pandas as pd
import pickle
import argparse
import utils
from Observation import Observation

def parse_cmd_line(as_dict=False):
    """
    Description:
        Parses the command line arguments for the program
    Parameters:
        as_dict - bool
            Returns the args as a dict. Default=False
    Returns:
        The command line arguments as a dictionary or a Namespace object and the
        parser used to parse the command line.
    """

    defaults = {
        "data_dir" : os.getcwd(),
        "tracks_dir" : os.getcwd(),
        "observations_dir" : os.getcwd()
    }
    
    parser = argparse.ArgumentParser(
        description="""Master's thesis source code""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", 
        type=str,
        default=defaults["data_dir"],
        help="""Directory where data exsits."""
    )
    parser.add_argument("--tracks_dir", 
        type=str,
        default=defaults["tracks_dir"],
        help="""Directory containing tracks files."""
    )
    parser.add_argument("--observations_dir", 
        type=str,
        default=defaults["observations_dir"],
        help="""Directory containing observations."""
    )
        
    args = parser.parse_args()
    if as_dict:
        args = vars(args)

    return args

if __name__ == "__main__":

    args = parse_cmd_line(as_dict=True)

    local_data = args["data_dir"] # '/Users/Taylor/Documents/herdhover-data'

    observations_meta_folder = (
        os.path.join(local_data, 'observations')) 
    observations_meta_file = os.path.join(observations_meta_folder, 'observations-meta.csv')
    observations_meta = pd.read_csv(observations_meta_file)

    # tracks_folder = '/Users/Taylor/Documents/tracks/tracks/utm'
    tracks_folder = args["tracks_dir"] #'/Users/Taylor/Documents/herdhover-data/tracks/utm'


    observation_species = ['Grevy\'s', 'plains'] # Observations to use

    obs_to_use_mask = (observations_meta['species'].isin(observation_species) & 
                    (observations_meta['tracked']=='True')) 

    observations_meta_used = observations_meta.loc[obs_to_use_mask].copy()
    observations_meta_used.sort_values('observation', inplace=True)
    observations_meta_used.reset_index()

    observations_info = observations_meta_used.loc[:,['observation', 'species']].values

    print('Using {} observations'.format(len(observations_info)))

    observations_folder = args["observations_dir"] #'/Users/Taylor/Documents/observations'

    orientation_files = glob.glob(os.path.join(observations_folder, '*/*-body-points.npy'))
    orientation_files.sort()
    observation_names = [os.path.basename(file).split('-')[0] for file in orientation_files]
    orientations_list = [np.load(file) for file in orientation_files]

    # Species to use within observation
    target_species = ['Grevy\'s', 'plains']
    save_folder = None 
    observation_dicts = []
    for observation_info in observations_info:
        if observation_info[0] in observation_names:
            body_points_file = os.path.join(observations_folder, '{}/{}-body-points.npy'.format(
            observation_info[0], observation_info[0]))
            
            postures_file = os.path.join(observations_folder, '{}/{}-postures.npy'.format(
            observation_info[0], observation_info[0]))
            posture_scores_file = os.path.join(observations_folder, '{}/{}-posture-scores.npy'.format(
            observation_info[0], observation_info[0]))
            
            body_points = np.load(body_points_file)
            postures = np.load(postures_file)
            posture_scores = np.load(posture_scores_file)
            
            observation_dict = {'name': observation_info[0],
                                'focal_species': observation_info[1],
                            'tracks_folder': tracks_folder,
                                'body_points': body_points,
                                'postures': postures,
                                'posture_scores': posture_scores,
                            'obs_meta_folder': observations_meta_folder,
                            'target_species': target_species,
                            'save_folder': save_folder}
            observation_dicts.append(observation_dict)


    

    len(observation_dicts)


    

    observations = []
    observations_names = []
    for observation_info in observation_dicts[:]:

        observation_name = observation_info['name']
        focal_species = observation_info['focal_species']
        tracks_folder = observation_info['tracks_folder']
        body_points = observation_info['body_points']
        postures = observation_info['postures']
        posture_scores = observation_info['posture_scores']
        observations_meta_folder = observation_info['obs_meta_folder']
        target_species = observation_info['target_species']
        save_folder = observation_info['save_folder']
        print(observation_name)
        tracks = np.load(os.path.join(tracks_folder, '{}-utm.npy'.format(observation_name))) 
        tracks_meta_file = os.path.join(observations_meta_folder, 
                                            observation_name, 
                                            '{}-tracks-meta.csv'.format(observation_name))
        tracks_meta = pd.read_csv(tracks_meta_file)

        obs_meta_file = os.path.join(observations_meta_folder, 
                                            observation_name, 
                                            '{}-meta.csv'.format(observation_name))
        obs_meta = pd.read_csv(obs_meta_file)

        observation = Observation(observation_name, tracks, body_points, postures, posture_scores,
                                tracks_meta, obs_meta, target_species, focal_species)
        # Use only tracks I want
        observation.combine_and_delete_tracks()
        observation.get_posture_based_positions()
        observations.append(observation)
        observations_names.append(observation_name)


    




    

    positions = []
    positions_names = []
    before_scare_positions = []
    before_scare_positions_names = []
    after_scare_positions = []
    after_scare_positions_names = []

    look_back = 3600

    for obs in range(len(observations)):
        positions.append(observations[obs].positions)
        positions_names.append(observations_names[obs])
        
        if not isinstance(observations[obs].scare_frame, str):
            if observations[obs].positions[:, :observations[obs].scare_frame, :].shape[1] > look_back:
                before_scare = observations[obs].positions[:, :observations[obs].scare_frame, :]
                before_scare_positions.append(before_scare)
                before_scare_positions_names.append(observations_names[obs])

            if observations[obs].positions[:, observations[obs].scare_frame:, :].shape[1] > look_back:
                after_scare = observations[obs].positions[:, observations[obs].scare_frame:, :]
                after_scare_positions.append(after_scare)
                after_scare_positions_names.append(observations_names[obs])


    

    observations[0].name


    

    grevy_positions = []
    grevy_positions_names = []
    grevy_before_scare_positions = []
    grevy_before_scare_positions_names = []
    grevy_after_scare_positions = []
    grevy_after_scare_positions_names = []

    plains_positions = []
    plains_positions_names = []
    plains_before_scare_positions = []
    plains_before_scare_positions_names = []
    plains_after_scare_positions = []
    plains_after_scare_positions_names = []

    look_back = 3600

    for obs in observations:
        if np.any(obs.tracks_meta['species'].values == 'Grevy\'s'):
            if not np.any(obs.tracks_meta['species'].values == 'plains'):
                grevy_positions.append(obs.positions)
                grevy_positions_names.append(obs.name)
                
                if not isinstance(obs.scare_frame, str):
                    if obs.positions[:, :obs.scare_frame, :].shape[1] > look_back:
                        before_scare = obs.positions[:, :obs.scare_frame, :]
                        grevy_before_scare_positions.append(before_scare)
                        grevy_before_scare_positions_names.append(obs.name)
                        
                    if obs.positions[:, obs.scare_frame:, :].shape[1] > look_back:
                        after_scare = obs.positions[:, obs.scare_frame:, :]
                        grevy_after_scare_positions.append(after_scare)
                        grevy_after_scare_positions_names.append(obs.name)
                        
        if np.any(obs.tracks_meta['species'].values == 'plains'):
            if not np.any(obs.tracks_meta['species'].values == 'Grevy\'s'):
                plains_positions.append(obs.positions)
                plains_positions_names.append(obs.name)
                
                if not isinstance(obs.scare_frame, str):
                    if obs.positions[:, :obs.scare_frame, :].shape[1] > look_back:
                        before_scare = obs.positions[:, :obs.scare_frame, :]
                        plains_before_scare_positions.append(before_scare)
                        plains_before_scare_positions_names.append(obs.name)
                        
                    if obs.positions[:, obs.scare_frame:, :].shape[1] > look_back:
                        after_scare = obs.positions[:, obs.scare_frame:, :]
                        plains_after_scare_positions.append(after_scare)
                        plains_after_scare_positions_names.append(obs.name)


    

    data_names = [positions_names, before_scare_positions_names, after_scare_positions_names,
                grevy_positions_names, grevy_before_scare_positions_names, grevy_after_scare_positions_names,
                plains_positions_names, plains_before_scare_positions_names, plains_after_scare_positions_names]
    data_list = [positions, before_scare_positions, after_scare_positions,
                grevy_positions, grevy_before_scare_positions, grevy_after_scare_positions,
                plains_positions, plains_before_scare_positions, plains_after_scare_positions]

    filenames = ['both_species', 'both_species_before_scare', 'both_species_after_scare',
                'grevy', 'grevy_before_scare', 'grevy_after_scare',
                'plains', 'plains_before_scare', 'plains_after_scare', 
                ]


    

    # positions_names


    

    map_obs = ['observation083', 'observation108', 'observation086', 'observation074']

    map_data_names = []
    map_data_list = []
    map_types = []

    for obs_names, all_obs_data in zip(data_names, data_list):
        name_index_data = []
        new_data = []
        for obs_name, obs_data in zip(obs_names, all_obs_data):
            if obs_name in map_obs:
                name_index_data.append(obs_name)
                new_data.append(obs_data)
        map_data_names.append(name_index_data)
        map_data_list.append(new_data)


    

    map_data_names


    

    observation_name_to_big_map_name_file = '/Users/Taylor/Documents/observation_name_to_big_map_name_dict.pkl'
    with open(observation_name_to_big_map_name_file, 'rb') as f:
            observation_name_to_big_map_name_dict = pickle.load(f)
            
    herdhover_folder = '/Users/Taylor/Documents/herdhover-data'

    observation_maps_dict = {}

    for observation_name in map_obs:

        map_name = observation_name_to_big_map_name_dict[observation_name]
        mask_file = os.path.join(herdhover_folder, 'game-trails', 
                                '{}-trails_mask.npy'.format(map_name))
        mask_info_file = os.path.join(herdhover_folder, 'game-trails', 
                                '{}-mask_info.csv'.format(map_name))

        trail_mask = np.load(mask_file)
        mask_info_df = pd.read_csv(mask_info_file)
        
        observation_maps_dict[observation_name] = {'trail_mask': trail_mask,
                                                'mask_info_df': mask_info_df}


    

    vals = []
    for name, utms in zip(map_data_names[0], map_data_list[0]):
        trail_mask = observation_maps_dict[name]['trail_mask']
        mask_info_df = observation_maps_dict[name]['mask_info_df']
        utms = utms
        new_coords_l, spatial, time_index, _ = spatial_dis(utms, 5.0, 1)

        print(np.any(np.isnan(np.concatenate(spatial))))
        
        print([np.any(np.isnan(t)) for t in spatial])
        print([t.shape for t in spatial])
        
    #     for s in spatial:
    #         vals.append(get_trail_values_from_utm(trail_mask, mask_info_df, s))


    

    map_data_names[0]


    

    observation_maps_dict['observation108']['mask_info_df']



    




    




    



    




    




    

    # import sys
    # # To see koger_general_functions


    




    

    data_names = [positions_names, before_scare_positions_names, after_scare_positions_names,
                plains_positions_names, plains_before_scare_positions_names, plains_before_scare_positions_names,
                grevy_positions_names, grevy_before_scare_positions_names, grevy_after_scare_positions_names]

    data_names = ['observation083', 'observation108', 'observation086', 'observation074']
    utm_arrays = [positions[10], positions[14], positions[11], positions[8]]


    #             print(np.nansum(map_vals), np.nansum(vals))


    




    

    name_index


    

    name_index


    

    print(len(data_list_map))


    

    data_list = [positions, before_scare_positions, after_scare_positions,
            plains_positions, plains_before_scare_positions, plains_after_scare_positions,
            grevy_positions, grevy_before_scare_positions, grevy_after_scare_positions]

    data_list = [plains_before_scare_positions]

    for data in range(len(data_list)):
    #     for distance in step_size:
    #         for radius in radius_around_ind:
    #             diff_radius = []
    #             start_ind_id_count = 0  
        for observation in range(len(data_list[data])):
            new_coords_l, spatial, time_index, drop_lst = spatial_dis(data_list[data][observation], distance, resolution)
            alt_steps = random_steps(spatial, distance)

        #     chosen_gtl = []
        #     alt_gtl = []

        #     for ind in range(len(spatial)):
        #         ind_trail_values = get_trail_values_from_utm(trail_mask, mask_info_df, spatial[ind])
        #         alt_step_trail_values = get_trail_values_from_utm(trail_mask, mask_info_df, spatial[ind])

        #         chosen_gtl.append(ind_trail_values)
        #         alt_gtl.append(alt_step_trail_values)


    

    mask_info_df


    # ### Spatial discretization

    

    folder_names = ['all_features_model']

    for folder in folder_names:
        os.makedirs(folder)


    

    print(len(after_scare_positions))
    for obs in after_scare_positions:
        print(obs.shape)


    # #### Model

    

    data_list = [positions, before_scare_positions, after_scare_positions,
            plains_positions, plains_before_scare_positions, plains_after_scare_positions,
            grevy_positions, grevy_before_scare_positions, grevy_after_scare_positions]
    # data_list = [plains_before_scare_positions]

    folder_path = '/Users/Taylor/Documents/all_features_model/'

    filenames = ['df_', 'before_scare_df_', 'after_scare_df_', 
                'plains_df_', 'plains_before_scare_df_', 'plains_after_scare_df_', 
                'grevy_df_', 'grevy_before_scare_df_', 'grevy_after_scare_df_']
    # filenames = ['plains_before_scare_df_']

    step_size = [5,10]
    feature_col_headers = ['group_center', 'social_density', 'recently_used_space']
    resolution = 1
    radius_around_ind = [1, 5, 10, 15, 20]
    look_back = 3600

    for data in range(len(data_list)):
        for distance in step_size:
            for radius in radius_around_ind:
                diff_radius = []
                start_ind_id_count = 0  
                for observation in range(len(data_list[data])):
                    new_coords_l, spatial, time_index, drop_lst = spatial_dis(data_list[data][observation], distance, resolution)
                    alt_steps = random_steps(spatial, distance)
                    
                    #group center
                    cl = group_center(new_coords_l)
                    chosen_gcl, alt_gcl = dc_feature(cl, spatial, distance)
                    
                    
                    if len(chosen_gcl) == len(drop_lst):
                        continue
                        
                    # trails
                    obs_name = data_names[data][observation]
                    trail_mask = observation_maps_dict[obs_name]['trail_mask']
                    mask_info_df = observation_maps_dict[obs_name]['mask_info_df']
                    chosen_trail, alt_trail = trail_feature(alt_steps, spatial, trail_mask, mask_info_df)


                    #social density
                    chosen_sdl, alt_sdl = sd_feature(alt_steps, new_coords_l, radius, fraction=False)
                    
                    #recently used space
                    chosen_rul, alt_rul = ru_feature(alt_steps, data_list[data][observation], radius, time_index, look_back, temporal_data=False, double_count=False)
                    
                    alt_fl = [alt_gcl, alt_sdl, alt_rul]
                    chosen_fl = [chosen_gcl, chosen_sdl, chosen_rul]

                    filename = 'multi_feature_positions' + str(observation)

                    multi_feature_ind_df_list = ind_df_multi_feature(alt_fl, chosen_fl, feature_col_headers, start_ind_id_count)
                    
                    pop_df, drop = save(multi_feature_ind_df_list, filename, distance, drop_lst, feature_col_headers, start_ind_id_count, save_individual_level_data=False, save_poplation_level_data=False)
                    
                    if len(drop)==len(chosen_rul):
                        continue
                        
                    diff_radius.append(pop_df)
                    
                    start_ind_id_count = start_ind_id_count + data_list[data][observation].shape[0]
                
                if len(diff_radius) == 0:
                    continue
                all_observations_one_radius = pd.concat(diff_radius)
            
                all_observations_one_radius.to_csv(folder_path + filenames[data] + str(distance) + 'm_step_size_' + str(radius) + 'm_radius_size.csv', index=False, header=True)
                
        print('dataset in data_list[' + str(data) + '] processed')


    

    ncols=2
    nrows=1
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, constrained_layout=True)
    # fig.delaxes(ax[1,2])
    ax[0].plot(t, np.sin(2 * np.pi * t))
    # ax[0,0].hist(np.arange(5))
    # ax[0,0].plot(t, np.sin(0.5 * np.pi * t))
    # ax[0,1].plot(t, np.sin(0.5 * np.pi * t))
    plt.close()
    # ax[1,0].plot(t, np.cos(2 * np.pi * t))
    # plt.setp(ax2.get_xticklabels(), visible=False)
    # fig.savefig('test')


    




    

    import matplotlib.gridspec as gridspec


    

    filenames = ['df_spatial_model', 'before_scare_df_spatial_model', 'after_scare_df_spatial_model',
                'grevy_df_spatial_model', 'grevy_before_scare_df_spatial_model', 'grevy_after_scare_df_spatial_model',
                'plains_df_spatial_model', 'plains_before_scare_df_spatial_model', 'plains_after_scare_df_spatial_model']

    feature_key = ['gc']

    figure_names = ['both species', 'both species before scare', 'both species after scare',
                'Grevy\'s', 'Grevy\'s' + ' before scare', 'Grevy\'s' + ' after scare',
                'plains', 'plains before scare', 'plains after scare']
    ncols=3
    nrows=3

    step_size = [5,10]
    radius_around_ind = [1, 5, 10, 15, 20]

    for feature in range(len(feature_key)):
        fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, constrained_layout=False, figsize=(15, 10))
        fig.delaxes(ax[2,2])
        fig_row_index = 0
        
        for name in range(len(filenames)):
            if filenames[name]=='plains_after_scare_df_spatial_model':
                continue
            obsv_df = multi_observations_beta_df(step_size, radius_around_ind, ('/Users/Taylor/Desktop/' + filenames[name]))
            group = obsv_df.groupby('step_size')
            fig_col_index = name
            
            if fig_col_index == ncols:
                fig_row_index = fig_row_index + 1
                
            elif fig_col_index == 2*ncols:
                fig_row_index = fig_row_index + 1
            
            if (fig_col_index >= ncols) & (fig_col_index <= (ncols*2)-1):
                fig_col_index = fig_col_index - ncols
                
            elif fig_col_index >= ncols*2:
                fig_col_index = fig_col_index - ncols*2

            for i in range(len(step_size)):
                if step_size[i] == 5:
                    symbol = 'o'
                else:
                    symbol = 'D'
                grp = group.get_group(step_size[i])
                
                if feature_key[feature]=='all_features':
                    
                    ax[fig_row_index, fig_col_index].scatter(grp['radius_size_around_ind'], grp['group_center'], label=str(step_size[i])+'m gc', marker=symbol, c='blue')
                    ax[fig_row_index, fig_col_index].scatter(grp['radius_size_around_ind'], grp['social_density'], label=str(step_size[i])+'m sd', marker=symbol, c="red")
                    ax[fig_row_index, fig_col_index].scatter(grp['radius_size_around_ind'], grp['recently_used_space'], label=str(step_size[i])+'m ru', marker=symbol, c='green')
                    
                elif feature_key[feature]=='sd':
                    ax[fig_row_index, fig_col_index].scatter(grp['radius_size_around_ind'], grp['social_density'], label=str(step_size[i])+'m', marker=symbol)
                elif feature_key[feature]=='ru':
                    ax[fig_row_index, fig_col_index].scatter(grp['radius_size_around_ind'], grp['recently_used_space'], label=str(step_size[i])+'m', marker=symbol)
                else:
                    if step_size[i] == 5:
                        continue
                    else:
                        symbol = 'o' 
                    ax[fig_row_index, fig_col_index].scatter(grp['radius_size_around_ind'].iloc[1:3], grp['group_center'].iloc[1:3], marker=symbol)
                    ax[fig_row_index, fig_col_index].plot(grp['radius_size_around_ind'].iloc[1:3], grp['group_center'].iloc[1:3])
                    
    #             ax[fig_row_index, fig_col_index].legend(loc='upper right', prop={'size': 8})
                ax[fig_row_index, fig_col_index].title.set_text(figure_names[name])

        handles, labels = ax[fig_row_index, fig_col_index].get_legend_handles_labels()
        fig.legend(handles, labels)

        if feature_key[feature]=='gc':
            fig.text(0.47, 0, 'step size (meters)', ha='center', fontsize=12)
        else:
            fig.text(0.47, 0, 'radius size around indivdual (meters)', ha='center', fontsize=12)
            
        fig.tight_layout(rect=[0,0,0.9,1])
    #     fig.tight_layout()

        fig.savefig('/Users/Taylor/Documents/model_figures/' + feature_key[feature] + '_one_legend')
    #     fig.savefig('/Users/Taylor/Documents/model_figures/' + feature_key[feature] + '_multiple_legends')


    

    # obsv_df[['step_size', 'social_density']]
    obsv_df


    # ### Probability

    

    from collections import Counter


    




    

    map_data_names


    

    data_list = [positions, before_scare_positions, after_scare_positions,
                grevy_positions, grevy_before_scare_positions, grevy_after_scare_positions,
            plains_positions, plains_before_scare_positions, plains_after_scare_positions
            ]

    data_list = [positions]


    filenames = ['both_species', 'both_species_before_scare', 'both_species_after_scare',
                'grevy', 'grevy_before_scare', 'grevy_after_scare',
                'plains', 'plains_before_scare', 'plains_after_scare', 
                ]

    folder_paths = ['/Users/Taylor/Documents/recently_used_space/', '/Users/Taylor/Documents/social_density/', 
                    '/Users/Taylor/Documents/group_center/', '/Users/Taylor/Documents/trail_use/']


    step_size = [5, 10]
    resolution = 1
    radius_around_ind = [1, 5, 10, 15, 20]
    look_back = 3600
    fraction = False

    for data in range(3,6):
        for trial in range(50):
        #     ru_keys, ru_probabilities, ru_occurances = calculate_probability(step_size, resolution, data_list[data], 
        #                                                                      radius_around_ind, look_back, fraction=fraction, 
        #                                                                      recently_used_space=True)
        #     sd_keys, sd_probabilities, sd_occurances = calculate_probability(step_size, resolution, data_list[data], 
        #                                                                      radius_around_ind, fraction=fraction, social_density=True)


            trail_keys, trail_probabilities, trail_occurances = calculate_probability(step_size, resolution, map_data_list[data], 
                                                                            radius_around_ind, trails=True, 
                                                                                    list_of_observation_names=map_data_names[data], 
                                                                                    observation_maps_dict=observation_maps_dict)

            #     gc_keys, gc_probabilities, gc_occurances = calculate_probability(step_size, resolution, data_list[data], group_center_feature=True)

        #     for size in range(len(step_size)):
        #         np.save(folder_paths[2] + filenames[data] + '_gc_keys_' + str(step_size[size]) + 'm_step_size', gc_keys[size])
        #         np.save(folder_paths[2] + filenames[data] + '_gc_probabilities_' + str(step_size[size]) + 'm_step_size', gc_probabilities[size])
        #         np.save(folder_paths[2] + filenames[data] + '_gc_occurances_' + str(step_size[size]) + 'm_step_size', gc_occurances[size])

        #     for radius_len in range(len(radius_around_ind)*len(step_size)):
        #         radius_name = radius_len
        #         if radius_len > 4:
        #             radius_name = radius_len - len(radius_around_ind)
        #             size = 10
        #         else:
        #             size = 5
        #         if fraction == False:
        #             np.save(folder_paths[0] + filenames[data] + '_ru_keys_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size'  + '_whole_counts', ru_keys[radius_len])
        #             np.save(folder_paths[0] + filenames[data] + '_ru_probabilities_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size' + '_whole_counts', ru_probabilities[radius_len])
        #             np.save(folder_paths[0] + filenames[data] + '_ru_occurances_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size' + '_whole_counts', ru_occurances[radius_len])

        #             np.save(folder_paths[1] + filenames[data] + '_sd_keys_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size' + '_whole_counts', sd_keys[radius_len])
        #             np.save(folder_paths[1] + filenames[data] + '_sd_probabilities_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size' + '_whole_counts', sd_probabilities[radius_len])
        #             np.save(folder_paths[1] + filenames[data] + '_sd_occurances_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size' + '_whole_counts', sd_occurances[radius_len])
        #         else:
        #             np.save(folder_paths[0] + filenames[data] + '_ru_keys_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size', ru_keys[radius_len])
        #             np.save(folder_paths[0] + filenames[data] + '_ru_probabilities_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size', ru_probabilities[radius_len])
        #             np.save(folder_paths[0] + filenames[data] + '_ru_occurances_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size', ru_occurances[radius_len])

        #             np.save(folder_paths[1] + filenames[data] + '_sd_keys_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size', sd_keys[radius_len])
        #             np.save(folder_paths[1] + filenames[data] + '_sd_probabilities_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size', sd_probabilities[radius_len])
        #             np.save(folder_paths[1] + filenames[data] + '_sd_occurances_' + str(size) + 'm_step_size_' + str(radius_around_ind[radius_name]) 
        #                     + 'm_radius_size', sd_occurances[radius_len])
            for step_num in range(len(step_size)):      
                np.save(folder_paths[3] + filenames[data] + '_trail_keys_' + str(step_size[step_num]) 
                        + 'm_step_size_' + str(0) + 'm_radius_size' + str(trial) + '_trial', trail_keys[step_num])
                np.save(folder_paths[3] + filenames[data] + '_trail_probabilities_' + str(step_size[step_num]) 
                        + 'm_step_size_' + str(0) + 'm_radius_size' + str(trial) + '_trial', trail_probabilities[step_num])
                np.save(folder_paths[3] + filenames[data] + '_trail_occurances_' + str(step_size[step_num]) 
                        + 'm_step_size_' + str(0) + 'm_radius_size' + str(trial) + '_trial', trail_occurances[step_num])


    

    trail_keys


    

    figure_names = ['both species', 'both species before scare', 'both species after scare',
                'Grevy\'s', 'Grevy\'s' + ' before scare', 'Grevy\'s' + ' after scare',
                'plains', 'plains before scare', 'plains after scare']
    # figure_names = ['both species before scare']

    folder_paths = ['/Users/Taylor/Documents/recently_used_space/', 
                    '/Users/Taylor/Documents/social_density/',

                ]
    folder_paths = ['/Users/Taylor/Documents/group_center/']


    file_types = ['both_species', 'both_species_before_scare', 'both_species_after_scare', 
                'grevy', 'grevy_before_scare', 'grevy_after_scare',
                'plains', 'plains_before_scare', 'plains_after_scare']
    # file_types = ['both_species']

    fraction = True
    radius_around_ind_name = [1, 5, 10, 15, 20, 1, 5, 10, 15, 20]
    step_size_l = [5, 10]

    for fraction in [True, False]:

        for folder in folder_paths:

            if 'social_density' in folder:
                feature_key = 'sd'
            if 'recently_used' in folder:
                feature_key = 'ru'
            if 'group_center' in folder:
                feature_key = 'gc'

            for file_type in file_types:
                if fraction==False:
                    keys_files = glob.glob(
                        os.path.join(folder, file_type + '_' + feature_key + '_keys*' + '_whole_counts.npy'))
                    keys_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                int(f.split('m_radius_size')[0].split('_')[-1])))
                    keys = [np.load(file) for file in keys_files]

                    probabilities_files = glob.glob(
                        os.path.join(folder, file_type + '_' + feature_key + '_probabilities*'+ '_whole_counts.npy'))
                    probabilities_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                            int(f.split('m_radius_size')[0].split('_')[-1])))
                    probabilities = [np.load(file) for file in probabilities_files]

                    occurances_files = glob.glob(
                        os.path.join(folder, file_type + '_' + feature_key + '_occurances*' + '_whole_counts.npy'))
                    occurances_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                        int(f.split('m_radius_size')[0].split('_')[-1])))
                    occurances = [np.load(file) for file in occurances_files]

                else:
                    keys_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_keys*radius_size.npy'))
                    keys_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                int(f.split('m_radius_size')[0].split('_')[-1])))
                    keys = [np.load(file) for file in keys_files]

                    probabilities_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_probabilities*radius_size.npy'))
                    probabilities_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                            int(f.split('m_radius_size')[0].split('_')[-1])))
                    probabilities = [np.load(file) for file in probabilities_files]

                    occurances_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_occurances*radius_size.npy'))
                    occurances_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                        int(f.split('m_radius_size')[0].split('_')[-1])))
                    occurances = [np.load(file) for file in occurances_files]


                if feature_key=='gc':
                    
                    keys_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_keys*.npy'))
                    keys_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1])))
                    keys = [np.load(file) for file in keys_files]

                    probabilities_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_probabilities*.npy'))
                    probabilities_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1])))
                    probabilities = [np.load(file) for file in probabilities_files]

                    occurances_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_occurances*.npy'))
                    occurances_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1])))
                    occurances = [np.load(file) for file in occurances_files]
                    
                    
                    nrows = 1
                    ncols = 2
                    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, constrained_layout=False, figsize=(10, 3.5))

                    for size in range(len(keys)):
                        if size == 0:
                            symbol = 'o'
                        else:
                            symbol = 'D'

                        fig_col_index = size

                        ax[fig_col_index].scatter(keys[size], probabilities[size], marker=symbol, label=str(step_size_l[size]) + 'm')
                        ax[fig_col_index].legend()
                        ax[fig_col_index].set_xlabel('chosen location - alternative location')
                        ax[fig_col_index].set_ylabel('probability chosen location')
                        ax[fig_col_index].set_title(file_type.replace('_', ' ') + ' ' + feature_key)

                        ax2 = fig.add_subplot(ax[fig_col_index])        
                        ax2 = ax[fig_col_index].twinx()  # instantiate a second axes that shares the same x-axis

                        bin_values = np.array(sorted(keys[size])) - 0.5
                        ax2.hist(occurances[size], alpha=0.3, bins=bin_values)

                        fig1.tight_layout()

                    fig.savefig('/Users/Taylor/Dropbox/Thesis_materials/probability_figures/' + file_type + '_' + feature_key)
    #                 plt.close()
                else:
                    ncols=5
                    nrows=2

                    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, constrained_layout=False, figsize=(20, 5))
                    fig_row_index = 0

                    for radius_len in range(len(keys)):
                        fig_col_index = radius_len
                        if radius_len <= 4:
                            symbol = 'o'
                            step_size = 5
                        else:
                            symbol = 'D'
                            step_size = 10

                        if fig_col_index == ncols:
                            fig_row_index = fig_row_index + 1

                        if fig_col_index >= ncols:
                            fig_col_index = fig_col_index - ncols

                        label = str(step_size) + 'm, ' + str(radius_around_ind_name[radius_len]) + 'm radius'
                        ax[fig_row_index, fig_col_index].scatter(keys[radius_len], probabilities[radius_len], 
                                                                marker=symbol, label=label)
                        ax[fig_row_index, fig_col_index].legend()
                        ax[fig_row_index, fig_col_index].set_xlabel('chosen location - alternative location')
                        ax[fig_row_index, fig_col_index].set_ylabel('probability chosen location')
                        ax[fig_row_index, fig_col_index].set_title(file_type.replace('_', ' ') + ' ' + feature_key)

                        ax2 = fig.add_subplot(ax[fig_row_index, fig_col_index])        
                        ax2 = ax[fig_row_index, fig_col_index].twinx()  # instantiate a second axes that shares the same x-axis

                        sorted_keys = np.array(sorted(keys[radius_len]))
                        if len(sorted_keys) > 1:
                            bin_values = sorted_keys - (sorted_keys[1] - sorted_keys[0]) / 2
                        else:
                            bin_values = 1
                        ax2.hist(occurances[radius_len], alpha=0.3, bins=bin_values)

                        fig.tight_layout()
                    if feature_key=='sd' or feature_key=='ru':
                        if fraction == True:
                            fig.savefig('/Users/Taylor/Dropbox/Thesis_materials/probability_figures/' 
                                        + file_type + '_' + feature_key + '_fraction')
                        if fraction == False:
                            fig.savefig('/Users/Taylor/Dropbox/Thesis_materials/probability_figures/' + file_type + '_' + feature_key)

    #                 plt.close()


    

    figure_names = [
                'Grevy\'s', 'Grevy\'s' + ' before scare', 'Grevy\'s' + ' after scare',
                ]

    folder = '/Users/Taylor/Documents/trail_use/'

    file_types = [ 
                'grevy', 'grevy_before_scare', 'grevy_after_scare']
    # file_types = ['both_species']

    step_size_l = [5, 10]


    feature_key = 'trail'

    num_trials = 50

    for file_type in file_types[:]:
        
        
        ncols=2
        nrows=1

        fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, constrained_layout=False, figsize=(15, 5))
        fig_row_index = 0
        
        prob_sums = [np.zeros(len(file_types)) for _ in range(len(step_size_l))]
                    
        
        for trial in range(num_trials):

            keys_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_keys*radius_size'+str(trial)+'*.npy'))
            keys_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                        int(f.split('m_radius_size')[0].split('_')[-1])))
            keys = [np.load(file, allow_pickle=True) for file in keys_files]

            probabilities_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_probabilities*radius_size'+str(trial)+'*.npy'))
            probabilities_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                    int(f.split('m_radius_size')[0].split('_')[-1])))
            probabilities = [np.load(file, allow_pickle=True) for file in probabilities_files]

            occurances_files = glob.glob(os.path.join(folder, file_type + '_' + feature_key + '_occurances*radius_size'+str(trial)+'*.npy'))
            occurances_files.sort(key=lambda f: (int(f.split('m_step_size')[0].split('_')[-1]),
                                                int(f.split('m_radius_size')[0].split('_')[-1])))
            occurances = [np.load(file, allow_pickle=True) for file in occurances_files]
            

            if np.any(occurances[0])==None:
                continue       

            for step_ind, step_size in enumerate(step_size_l):
                

                if step_size == 5:
                    symbol = 'o'
                elif step_size == 10:
                    symbol = 'D'
                    
                key_inds = keys[step_ind].argsort()
                keys_s = keys[step_ind][key_inds]
                prob_s = probabilities[step_ind][key_inds]
                prob_sums[step_ind] += prob_s


                fig_col_index = 0


                label = str(step_size) + 'm, ' + str(0) + 'm radius'

                ax[step_ind].scatter(keys_s, prob_s, 
                                    marker=symbol, c='blue', alpha=.2)
    #             ax[fig_row_index].legend()

        for step_ind, step_size in enumerate(step_size_l):
            ax[step_ind].set_xlabel('chosen location - alternative location')
            ax[step_ind].set_ylabel('probability chosen location')
            ax[step_ind].set_title(file_type.replace('_', ' ') + ' ' + feature_key + ' following probability'
                                + ' {} meter step size'.format(step_size))
            
            ax[step_ind].scatter(keys_s, prob_sums[step_ind]/num_trials, 
                                    marker=symbol, label='{} meter step size'.format(step_size), c='red', alpha=1.0)
            
            print(prob_sums[step_ind]/num_trials)

            ax[step_ind].set_ylim(0, 1.0)
            
            ax2 = fig.add_subplot(ax[fig_row_index])        
            ax2 = ax[step_ind].twinx()  # instantiate a second axes that shares the same x-axis

            sorted_keys = sorted(keys[radius_len])
            sorted_keys.append(sorted_keys[-1] + 1)
            sorted_keys = np.array(sorted_keys)
            if len(sorted_keys) > 1:
                bin_values = sorted_keys - (sorted_keys[1] - sorted_keys[0]) / 2

            else:
                bin_values = 1
            ax2.hist(occurances[step_ind], alpha=0.3, bins=bin_values)

        fig.tight_layout()

        fig.savefig('/Users/Taylor/Dropbox/Thesis_materials/probability_figures/' + file_type + '_' + feature_key)

    #     plt.close()


    

    # type(keys[step_ind], prob_sums[step_ind]/num_trials


    

    key_inds = keys[0].argsort()
    keys_s = keys[0][key_inds]
    prob_s = probabilities[0][key_inds]

    print(keys_s, keys[0])
    print(prob_s, probabilities[0])


    

    keys


    

    step_size = [5, 10]
    resolution = 1

    gc_keys, gc_probabilities, gc_occurances = calculate_probability(step_size, resolution, positions, fraction=True, group_center_feature=True)


    

    print(len(keys))
    print(len(probabilities))
    print(len(occurances))


    

    step_size = [5, 10]

    for size in range(len(step_size)):
        if size == 0:
            symbol = 'o'
        else:
            symbol = 'D'
            
        fig, ax1 = plt.subplots()
        
        ax1.scatter(keys[size], probabilities[size], marker=symbol, label=str(step_size[size]) + 'm')
        ax1.legend()
        ax1.set_title('group center')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        bin_values = np.array(sorted(gc_keys[size])) - 0.5

        ax2.hist(gc_occurances[size], alpha=0.3, bins=bin_values)

        ax1.set_xlabel('chosen location - alternative location')
        ax1.set_ylabel('probability chosen location')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


    # #### Ari eLife 2017 figures

    

    frame = 2500
    resolution = 15
    buffer = 20
    individual = 0
    r = 1.5
    radius_size_list = [1, 5, 10, 15, 20, 1, 5, 10, 15, 20]

    fraction = cal_fraction_figure(positions[0], positions[0][individual,frame, :], resolution, buffer, frame, individual, r)

    for radius_size in range(obsv_df.shape[0]):
        preference = obsv_df['social_density'].iloc[radius_size] * fraction
        plt.imshow(preference)
        plt.colorbar()
        if radius_size<=4:
            step_size=5
        else:
            step_size=10
            
        plt.title(str(step_size) + 'm_step_size_' + str(radius_size_list[radius_size]) + 'm_radius_around_ind')
        plt.show()

