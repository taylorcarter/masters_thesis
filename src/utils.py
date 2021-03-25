import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import pandas as pd
import pickle

# import sys
# To see koger_general_functions
# sys.path.append('/home/golden/coding/drone-tracking/code/objects' )

from Observation import Observation

# sys.path.append('/Users/Taylor/Documents/koger_general_functions.py')

# import mapping_functions as kmap

def random_steps(pts_act, r=None, number_of_steps=5):
    """
    Generates possible x,y positions for every individual at every step by pulling random values from 
    calculated step sizes and turning angles
    
    Inputs:
        pts_act: either list of 2D numpy arrays containing the xy coordinates for each individual after they have moved at least r meters(spatial discretization),
                 or a 3d numpy array with raw xy positions of animals in space with shape (individual, frame, 2) (temporal discretization)
                 
        r: int; distance focal individual should travel before a step is recorded
    Ouputs:
        coords_lst: list of 3D numpy arrays with new xy positions for each individual in pts_act with shape (step number, location, 2). coords_lst[0] = xy cooridnates
                    for alternative steps 1-number_of_steps for indvidual 0
        coords_list: list of 3D numpy arrays with new xy positions for each individual in pts_act. coords_list[0] = xy coordinates
                     for 1st alternative step for each individual
        
    """
    
    if isinstance(pts_act, np.ndarray):
        act_diff = np.diff(pts_act, axis=1)
        step_size = np.sqrt((act_diff[:,:,0])**2 + (act_diff[:,:,1])**2)

        act_dir = np.arctan2(act_diff[:,:,1], act_diff[:,:,0])
        d_diff = np.diff(act_dir)

        coords_list = []
        for number in range(number_of_steps):

            coords = np.zeros(pts_act.shape)
            for ind in range(pts_act.shape[0]):

                ss_dropnan = step_size[ind][~np.isnan(step_size[ind])]
                d_diff_dropnan = d_diff[ind][~np.isnan(d_diff[ind])]

                #Once the NaNs are dropped some ind have empty ss_dropnan and d_diff_dropnan. Inds that have this issue have been
                #tagged in time_adj and will be dropped from dataset later, but in order to preserve labeling they need to stay until the final dataframe is made.
                #Thus any ind with this issue will use lst
                if len(d_diff_dropnan)==0:
                    lst = [0]
                    random_step = np.random.choice(lst, pts_act[ind].shape[0])
                    random_turn = np.random.choice(lst, pts_act[ind].shape[0])
                else:
                    random_step = np.random.choice(ss_dropnan, pts_act[ind].shape[0])
                    random_turn = np.random.choice(d_diff_dropnan, pts_act[ind].shape[0])

                x = random_step * (np.cos(random_turn)) + pts_act[ind,:,0]
                y = random_step * (np.sin(random_turn)) + pts_act[ind,:,1]

                coords[ind, :, 0] = x
                coords[ind, :, 1] = y

            coords_list.append(coords)
        return coords_list
    
    elif isinstance(pts_act, list):
        coords_lst = []
        for ind in range(len(pts_act)):
            act_diff = np.diff(pts_act[ind], axis=0)
            step_size = r
            
            act_dir = np.arctan2(act_diff[:,1], act_diff[:,0])
            d_diff = np.diff(act_dir)

            steps = np.zeros((number_of_steps, pts_act[ind].shape[0], 2))
            for number in range(number_of_steps):
                coords = np.zeros(pts_act[ind].shape)

                d_diff_dropnan = d_diff[~np.isnan(d_diff)]

                if len(d_diff_dropnan)==0:
                    lst = [0]
                    random_turn = np.random.choice(lst, pts_act[ind].shape[0])
                else:
                    random_turn = np.random.choice(d_diff_dropnan, pts_act[ind].shape[0])

                x = step_size * (np.cos(random_turn)) + pts_act[ind][:,0]
                y = step_size * (np.sin(random_turn)) + pts_act[ind][:,1]

                coords[:, 0] = x
                coords[:, 1] = y

                steps[number] = coords

            coords_lst.append(steps)

        return coords_lst

def group_center(act_pts): #number_ind=None, subset=False):
    """
    Generates x,y positions for the center of the whole group or the center of k number of closest individuals
    relative to each individual in the dataset
    
    Inputs:
        act_pts: list of 3D numpy arrays containing the xy coordinates for a focal individual after it has moved r meters as well
                 as the xy coordinates of every other individual at that timepoint. The position in the list identifies the focal
                 individual e.g. in new_act_pts_l[0] individual 0 is the focal individual; or a 3d numpy array with raw xy 
                 positions of animals in space with shape (individual, frame, 2)
        
        
        subset: if True, finds center of k number of closest individuals (number_ind) relative to focal
                individual
        number_ind: int 2 or above. Designates number of closest individuals to consider
        
        
    Output:
        center_l: list of 2D numpy arrays with the xy positions for where the center of the group is when using an individual's locations e.g.
                  center_l[0] gives the locations for the center of the group when using individual 0's locations
      
        center: 3D numpy array with x,y positions of the center of the group. Broadcasted for each individual
        
        
        
        If subset==True: (wrong, see comments)
            c_diff: 3d numpy array with x,y positions of the center of the k number of closest individuals
                    relative to focal individual. Has shape (individuals, frames, 2)
            center: 3d numpy array with x,y positions of the center of the k number of closest individuals.
                    Has shape (individuals, frames, 2)
    
    """
    if isinstance(act_pts, np.ndarray):   
        center = np.zeros(act_pts.shape)

#         if subset==True:
#             for ind in range(act_pts.shape[0]):
#                 diff = act_pts - act_pts[ind]
#                 dis = np.sqrt((diff[:,:,0])**2 + (diff[:,:,1])**2)

#                 #wrong here
#                 sort = np.argsort(np.where(dis==0, dis.max(), dis), axis=0)
#                 sort = sort[0:number_ind, :] 

#                 if number_ind<2 or number_ind>=act_pts.shape[0]:
#                     raise ValueError("ValueError: number of individuals needs to be >=2 and < act_pts.shape[0]" 
#                                      + str(len(pts_act)-1))

#                 act_loc = np.zeros((sort.shape[0], sort.shape[1]), 2)

#                 for i in range(sort.shape[0]):
#                     for pt in range(sort.shape[1]):
#                         al = act_pts[sort[i, pt], pt]
#                         act_loc[i, pt ,:] = al

#                 center[ind, :, :] = np.nanmean(act_loc, axis=0)

#         else:
#             center[:,:,:] = np.nanmean(act_pts, axis=0)

        center[:,:,:] = np.nanmean(act_pts, axis=0)
        c_diff = center - act_pts
        
        return center
    
    elif isinstance(act_pts, list):
        center_l = []
        c_diff_l = []

        for ind in range(len(act_pts)):
            center = np.zeros(act_pts[ind][ind].shape)
            center = np.nanmean(act_pts[ind], axis=0)
            center_l.append(center)

            c_diff = center - act_pts[ind][ind]
            c_diff_l.append(c_diff)

        return center_l

def dir_change(pts_inf, pts_act, random_step=None, rand_step=False):
    """
    Finds the change in direction between the vector pointing from pts_act to pts_inf and 
    the vector pointing from pts_act to pts_act
    
    Inputs:
        pts_inf: 3D numpy array with xy positions of social influencers i.e. the center of the group, the 
                 closest individual, the center of a subset of individuals, etc. with shape (individual, frame, 2); OR
                 a list of 2D numpy arrays with xy positions of social influencers len(pts_inf) = number of individuals
                 
        pts_act: 3d numpy array with raw xy positions of animals in space with shape (individual, frame, 2); OR a list of 2D
                 numpy arrays with xy positions of each individual len(pts_act) = number of individuals
        
        If rand_step==True:
            calulate the change in direction direction between the vector pointing from pts_act to pts_inf and 
            the vector pointing from pts_act to random_steps
            random_step: list of 3D numpy arrays with xy positions of alternative steps for each ind in pts_act len(random_steps) = number of steps;
                         OR list of 3D numpy arrays with xy positions of alternative steps for each ind where len(random_steps) = number of individuals
        
    Ouputs:
        dir_diff: 2D numpy array with the direction change between the vector pointing from pts_act to pts_inf and 
                  the vector pointing from pts_act to pts_act
        dir_diff_l: list of 1D numpy arrays with the direction change between the vector pointing from pts_act to pts_inf and 
                    the vector pointing from pts_act to pts_act. The position in the list holds the direciton change values for one individual
        
        If rand_dtep==True:
            dir_diff_l: list of 2D numpy arrays with the direction change between the vector pointing from pts_act to pts_inf and 
                        the vector pointing from pts_act to random_step. 
                        Has shape (individual, frame)
                        If type(pts_act) is a np.ndarray then len(dir_diff_l) = number of steps
                        If type(pts_act) is a list then len(dir_diff_l) = number of individuals
    """
    if isinstance(pts_act, np.ndarray):
        #direction of vector from pts_act to pts_inf
        loc_rel = pts_inf - pts_act
        inf_dir = np.arctan2(loc_rel[:,:,1], loc_rel[:,:,0])

        if rand_step==True:
            #direction of vector from pts_act to random_steps
            dir_diff_l = []
            for step in range(len(random_step)):
                loc_rel = random_step[step] - pts_act
                direct = np.arctan2(loc_rel[:,:,1], loc_rel[:,:,0])
                direct = direct[:,:-1]
            
                dir_diff = inf_dir[:,:-1] - direct
                dir_diff = np.where(dir_diff < -np.pi, (2*np.pi) + dir_diff , dir_diff)
                dir_diff = np.where(dir_diff > np.pi, dir_diff - (2*np.pi), dir_diff)
                
                dir_diff_l.append(dir_diff)
            
            return dir_diff_l

        else:
            #direction of vector from pts_act to pts_act
            act_diff = np.diff(pts_act, axis=1) #loose frame here, correct in inf_dir and direct when rand_step==True
            direct = np.arctan2(act_diff[:,:,1], act_diff[:,:,0])
            
            dir_diff = inf_dir[:,:-1] - direct
            dir_diff = np.where(dir_diff < -np.pi, (2*np.pi) + dir_diff , dir_diff)
            dir_diff = np.where(dir_diff > np.pi, dir_diff - (2*np.pi), dir_diff)
            
            return dir_diff
    
    elif isinstance(pts_act, list):
        
        dir_diff_l = []
        
        for ind in range(len(pts_act)):
            #direction of vector from pts_act to pts_inf
            loc_rel = pts_inf[ind] - pts_act[ind]
            inf_dir = np.arctan2(loc_rel[:,1], loc_rel[:,0])

            if rand_step==True:
                #direction of vector from pts_act to random_steps
                loc_rel = random_step[ind] - pts_act[ind]

                direct = np.arctan2(loc_rel[:,:,1], loc_rel[:,:,0])
                direct = direct[:,:-1]

                dir_diff = inf_dir[:-1] - direct
                dir_diff = np.where(dir_diff < -np.pi, (2*np.pi) + dir_diff , dir_diff)
                dir_diff = np.where(dir_diff > np.pi, dir_diff - (2*np.pi), dir_diff)

            else:
                #direction of vector from pts_act to pts_act
                act_diff = np.diff(pts_act[ind], axis=0) #loose frame here, correct in inf_dir and direct when rand_step==True
                direct = np.arctan2(act_diff[:,1], act_diff[:,0])

                dir_diff = inf_dir[:-1] - direct
                dir_diff = np.where(dir_diff < -np.pi, (2*np.pi) + dir_diff , dir_diff)
                dir_diff = np.where(dir_diff > np.pi, dir_diff - (2*np.pi), dir_diff)
#                 print(dir_diff)
            dir_diff_l.append(np.abs(dir_diff))
#             print(dir_diff_l)

        return dir_diff_l

def ind_df(alt_l, chosen, feature_column_header, start_ind_count_from=0):
    """ 
    Creates a list of pandas dataframes in the correct format for the R function 'mclogit'. Each position in the list is for one individual
    
    Inputs:
        alt_l: list consisting of 2d numpy arrays
               If type(chosen) = np.ndarray then len(alt_l) = number of alternative steps generated 
               If type(chosen) = list then len(alt_l) = number of individuals
        chosen: 2D numpy array OR a list of 1D numpy arrays
        feature_column_header: string, name of column header for feature going into dataframe
        start_ind_count_from: int, used when looking at multiple obeservations together
    Ouputs:
        ind_dfl: a list of pandas dataframes
    
    """
    if isinstance(chosen, list):
        df1_l = []
    
        for ind in range(len(chosen)):
            chosen_step_id = np.arange(chosen[ind].shape[0])
            chosen_individual_id = np.ones(chosen[ind].shape[0]) * (ind + start_ind_count_from)

            chosen_column = chosen[ind]
    
            d1 = {'individual': chosen_individual_id, feature_column_header: chosen_column, 'label': np.ones(chosen_column.shape), 'step': chosen_step_id}
            df1 = pd.DataFrame(data=d1)
            
            df1_l.append(df1)
    
        ind_df_l = []

        for ind in range(len(alt_l)):
            step_id = np.zeros(alt_l[ind].shape)
            individual_id = np.ones(alt_l[ind].flatten().shape[0]) * (ind + start_ind_count_from)

            for f in range(step_id.shape[1]):
                step_id[:,f] = step_id[:,f] + f
            
            step_id = step_id.flatten()
            
            #create feature col
            alt_flatten = alt_l[ind].flatten()
        
            d2 = {'individual': individual_id, feature_column_header: alt_flatten, 'label': np.zeros(alt_flatten.shape), 'step': step_id}
            df2 = pd.DataFrame(data=d2)

            #combine chosen and alternative dataframes together
            df = np.abs(pd.concat([df1_l[ind], df2]))
            drop_nan_df = df.dropna()
            
            ind_df_l.append(df)

    elif isinstance(chosen, np.ndarray):
        df1_l = []
        
        for ind in range(chosen.shape[0]):  
            frame_id = np.arange(chosen[ind].shape[0])
            individual_id = np.ones(chosen[ind].shape[0]) * (ind + start_ind_count_from)
            
            feature_column = chosen[ind]

            d1 = {'individual': individual_id, feature_column_header: feature_column, 'label': np.ones(feature_column.shape), 'frame': frame_id}
            df1 = pd.DataFrame(data=d1)
            
            df1_l.append(df1)
        
        all_steps = np.zeros((len(alt_l), alt_l[0].shape[1]))
        alt_l_concatenated = np.concatenate(alt_l, axis=1)
        
        ind_df_l = []
        
        for ind in range(alt_l_concatenated.shape[0]):
            individual_id = np.ones(alt_l_concatenated[ind].shape[0]) * (ind + start_ind_count_from)
            
            for f in range(all_steps.shape[1]):
                all_steps[:,f] = all_steps[:,f] + f

            frame_id = all_steps.flatten()

            feature_id = alt_l_concatenated[ind]

            d2 = {'individual': individual_id, feature_column_header: feature_id, 'label': np.zeros(feature_id.shape), 'frame': frame_id}
            df2 = pd.DataFrame(data=d2)
            
            df = np.abs(pd.concat([df1_l[ind], df2]))
            drop_nan_df = df.dropna()
            
            ind_df_l.append(drop_nan_df)

    return ind_df_l

def ind_df_multi_feature(alt_fl, chosen_fl, feature_col_header_list, start_ind_count_from=0):
    """ 
    Creates a pandas dataframe in the correct format for the R function 'mclogit' for population level analysis with multiple features
    
    Inputs:
        alt_fl: a list of lists consisting of 2d numpy arrays. len(alt_fl) = number of features. 
                len(alt_fl[0]) = number of individuals
                OR
                len(alt_fl[0]) = number of alternative steps
        
        
        chosen_fl: list consisting of 2d numpy arrays. len(list) = number of features
                   OR
                   list of lists consisting of of 1D numpy arrays
        start_ind_count_from: int, used when looking at multiple obeservations together
        
    Ouputs:
        feature_dfl: a pandas df with columns 'individual', 'feature' * n, 'label', and 'frame'
    
    """
    
    feature_dfl = []
    for feature in range(len(chosen_fl)):
        ind_dfl = ind_df(alt_fl[feature], chosen_fl[feature], feature_col_header_list[feature], start_ind_count_from)
        feature_dfl.append(ind_dfl)

    for feature in range(1, len(feature_col_header_list)):
        for idv_df in range(len(feature_dfl[0])):
            feature_dfl[0][idv_df].insert(feature+1, feature_col_header_list[feature], feature_dfl[feature][idv_df][feature_col_header_list[feature]])
    ind_df_multi_feature_l = feature_dfl[0]
    return ind_df_multi_feature_l




def dc_feature(inf_pts, act_pts, r=None):
    """
    Calculates the change in direction towards a point of influence for collected data and generated data
    
    Inputs:
        inf_pts: 3D numpy array with xy positions of social influencers i.e. the center of the group, the 
                 closest individual, the center of a subset of individuals, etc. with shape (individual, frame, 2); OR
                 a list of 2D numpy arrays with xy positions of social influencers len(pts_inf) = number of individuals
        act_pts: either list of 2D numpy arrays containing the xy coordinates for each individual after they have moved at least r meters(spatial discretization),
                 or a 3d numpy array with raw xy positions of animals in space with shape (individual, frame, 2) (temporal discretization)
        r: int; distance focal individual should travel before a step is recorded
    Outputs:
        If type(pts_act) is a np.ndarray:
            act_dc: 2D numpy array with the direction change between the vector pointing from pts_act to pts_inf and 
                    the vector pointing from pts_act to pts_act
            alt_dcl: list of 2D numpy arrays containing the change in direction towards the point of influence for every individuals alternative steps
                     len(alt_dcl) = number of steps
            
        If type(act_pts) is a list:
            act_dcl: list of 1D numpy arrays with the direction change between the vector pointing from pts_act to pts_inf and 
                     the vector pointing from pts_act to pts_act. Each position in the list holds the direciton change values for each individual
            alt_dcl: list of 2D numpy arrays containing the change in direction towards the point of influence for every individuals alternative steps
                     len(alt_dcl) = number of individuals
    """
    
    if isinstance(act_pts, np.ndarray):
        
        act_dc = dir_change(inf_pts, act_pts)
        
        alt_sl = random_steps(act_pts)
        alt_dcl = dir_change(inf_pts, act_pts, alt_sl, rand_step=True)

        return act_dc, alt_dcl
    
    elif isinstance(act_pts, list):
        act_dcl = dir_change(inf_pts, act_pts)
        
        alt_sl = random_steps(act_pts, r)
        alt_dcl = dir_change(inf_pts, act_pts, alt_sl, rand_step=True)
        
        return act_dcl, alt_dcl




def save(ind_dfl, filename, t, drop_lst, list_feature_names, start_ind_count_from=0, temporal_data=False, save_individual_level_data=True, save_poplation_level_data=True):
    """
    Drops individuals with bad data and saves their dfs. Also returns a population df containing only the individuals with good data
    
    Inputs:
        filename: string
        
        If temporal_data==True:
            t: int desigating the new timescale at witch the data is being analyzed
        If temporal_data==False (i.e. spatial data is entered):
            t: int designating the step size of the individual
            
        drop_lst: list of individuals to be dropped from dataset
        ind_dfl: list of dataframes, one for each individual 
        list_feature_names: list of strings for column headers of each feature
    Output:
        pop_df: dataframe consisting of all individuals in the dataset with good data (individuals with bad data has been dropped)
        drop: list of individuals to be dropped from dataset. Filters individuals with too many NaNs, indviduals with all the same true and false values, and
              individuals with empty dfs
    """
    
    if save_individual_level_data==True:
        for i in range (len(ind_dfl)):
            if temporal_data==True:
                ind_dfl[i].to_csv(r'/Users/Taylor/Documents/' + filename + '_ind' + str(i + start_ind_count_from) + 
                                              'df_t' + str(t) + '.csv', index = False, header=True)
            else:
                ind_dfl[i].to_csv(r'/Users/Taylor/Documents/' + filename + '_ind' + str(i + start_ind_count_from) + 
                                              'df_' + str(t) + 'meter_step_size.csv', index = False, header=True)
      
    #look at each feature column, if any of them have the same exact value for every input in the column then make note to have that ind dropped
    same_num_dropl = []
    
    for idv in range(len(ind_dfl)):
        for n in list_feature_names:
            check = is_same(ind_dfl[idv][n])
            if check==True:
                same_num_dropl.append(idv)
                break
                
    #drop 'bad data' from population df
    drop = np.concatenate((drop_lst, same_num_dropl))
    drop = np.unique(drop)
    drop = drop.astype(int)
    
    drop_copy = np.copy(drop)
    for ind in range(drop_copy.shape[0]):
        if len(drop_copy) > 1:
            if drop_copy[ind] >= drop_copy[1]:
                drop_copy[ind] = drop_copy[ind] - ind

        del ind_dfl[drop_copy[ind]]
    
    if len(ind_dfl) > 0:
        pop_df = pd.concat(ind_dfl)

        if save_poplation_level_data==True:
            if temporal_data==True:
                pop_df.to_csv(r'/Users/Taylor/Documents/' + filename + '_population_df_t' + 
                                 str(t) + '.csv', index = False, header=True)
            else:
                pop_df.to_csv(r'/Users/Taylor/Documents/' + filename + '_population_df_' + 
                                 str(t) + 'meter_step_size.csv', index = False, header=True)


        if save_individual_level_data==True:
            #drop individuals with 'bad data' from saved files
            for ind in drop:
                if temporal_data==True:
                    os.remove(filename + '_ind' + str(ind + start_ind_count_from) + 'df_t' + str(t) + '.csv')
                else:
                    os.remove(filename + '_ind' + str(ind + start_ind_count_from) + 'df_' + str(t) + 'meter_step_size.csv')
    else:
        pop_df = 'empty dataframe'
    return pop_df, drop




def is_same(col):
    """
    Boolean testing whether a column in a pandas dataframe has the same exact value in every row
    
    Input:
        col: column of dataframe
    Output:
        Boolean saying whether all values in the column are the same. True if true, False if false
    
    """
    if col.shape[0]>0:
        col_np = col.to_numpy()
        boolean = (col_np[0] == col_np[1:]).all()
    else:
        boolean = False
    return boolean




def calculate_fraction(act_pts, focal, r, fraction=True):
    """
    Calcualtes fraction of individuals within a given radius of a focal individual
    
    Inputs:
        act_pts: 3D numpy array (ind, frame, xy); OR list of 3D numpy arrays containing the xy coordinates for a focal individual after it has moved r meters as well
                 as the xy coordinates of every other individual at that timepoint.
                 
        focal: 3D numpy array (ind, frame, xy). Focal indivdual. OR list of 3D numpy arrays containing the xy coordinates for a focal individual after it has moved r meters as well
               as the xy coordinates of every other individual at that timepoint.
        r: int; radius around focal individual
        fraction: if False does not calculate fraction of individuals within a givin radius, gives counts of individuals within radius
    Output:
        counts: 2D numpy array containing the fraction of individuals within the given radius of the focal individual. shape (ind, frame)
        counts_l: list of 1D numpy arrays containing the fraction of individuals within the given radius of the focal individual. The focal ind
                  is equal to the position in act_pts list.
        If fraction==False:
            counts: 2D numpy array containing the absolute number of individuals within the given radius of the focal individual. shape (ind, frame)
            counts_l: list of 1D numpy arrays containing the absolute number of individuals within the given radius of the focal individual. The focal ind
                  is equal to the position in act_pts list.
    
    """
    if isinstance(act_pts, np.ndarray): #temporal
        
        counts = np.zeros((act_pts.shape[0], act_pts.shape[1]-1)) #need drop last frame to match len of direction change feature

        for ind in range(act_pts.shape[0]):
            loc = (act_pts[:,:,0] - focal[ind,:,0])**2 + (act_pts[:,:,1] - focal[ind,:,1])**2
            loc = np.where((loc<=r**2), 1, 0)
            loc[ind] = 0 #eliminating focal individual from count
            
            if fraction==True:
                count = np.nansum(loc, axis=0)/(np.sum(~np.isnan(act_pts[:,:,0]), axis=0)-1)
            elif fraction==False:
                count = np.nansum(loc, axis=0)
          
            counts[ind] = count[:-1] #need drop last frame to match len of direction change feature
        return counts
    
    elif isinstance(act_pts, list): #spatial
        
        counts_l = []
        for ind in range(len(act_pts)):
            counts = np.zeros((focal[ind].shape[0], act_pts[ind].shape[1]-1)) #need drop last frame to match len of direction change feature
            
            for i in range(focal[ind].shape[0]):
                loc = (act_pts[ind][:,:,0] - focal[ind][i,:,0])**2 + (act_pts[ind][:,:,1] - focal[ind][i,:,1])**2
                loc = np.where((loc<=r**2), 1, 0)
                loc[ind] = 0 #eliminating focal individual from count
            #changes start
                if fraction==True:
                    count = np.nansum(loc, axis=0)/(np.sum(~np.isnan(act_pts[ind][:,:,0]), axis=0)-1)
                elif fraction==False:
                    count = np.nansum(loc, axis=0)
            #changes end        
                counts[i] = count[:-1] #need drop last frame to match len of direction change feature
                    
            counts_l.append(counts)
                
        return counts_l




def get_trail_values_from_utm(trail_mask, mask_info_df, utms):
    """ Get the trail value for every utm point in trail mask array.
    
    Args:
        trail_mask: 2D np array with values from 0 to 1 with 1 being a trail
        mask_info_df: pandas dataframe with x_origin, y_origin, pixel_width, pixel_height values
        utms: array of shape (steps, 2) or (alt_steps, steps, 2)
        
    Returns:
        array of shape (steps) or (alt_steps, steps)
    
    """
    
    def _get_trail_values_from_track(trail_mask, mask_info_df, utm_track):
        raster_track = kmap.utm_to_raster_track(utm_track, mask_info_df.loc[0, 'x_origin'],
                                                mask_info_df.loc[0, 'y_origin'], 
                                                mask_info_df.loc[0, 'pixel_width'],
                                                mask_info_df.loc[0, 'pixel_height'], 
                                                1.0)
    
        map_vals = np.array([np.nan for _ in range(raster_track.shape[0])])

        map_vals[~np.isnan(raster_track[:,0])] = trail_mask[raster_track[~np.isnan(raster_track[:,0]),1].astype(int), 
                                                    raster_track[~np.isnan(raster_track[:,0]),0].astype(int)]
        
        return map_vals
    
    if len(utms.shape) == 2:
        return _get_trail_values_from_track(trail_mask, mask_info_df, utms)
    
    elif len(utms.shape) == 3:
        map_vals = []
        for utm_track in utms:
            map_vals.append(_get_trail_values_from_track(trail_mask, mask_info_df, utm_track))
            
        map_vals = np.stack(map_vals)
        
        return map_vals
    else:
        print('utms needs to be shape 2 or 3.')
        
    


def trail_feature(alt_pts, act_pts, trail_mask, mask_info_df):
    """
    Calculates if focal indivdiual and its alternative steps are on a trail
    
    Inputs:
        alt_pts: list of 3D numpy arrays with new xy positions of animals in pts_act. alt_pts[0] = xy coordinates
                 of an alternative step for each individual
        act_pts: 3D numpy array with raw xy positions of animals in space with shape (individual, frame, 2)
        trail_mask: 2D numpy array 0 if no trail 1 if trail in that location
        mask_info_df: info about how trail map relates to utm coordinates ('x_origin', 'y_origin', 'pixel_height', 'pixel_width')
    Output:
        individual_specific_counts_l: 1D numpy array containing the fraction of individuals within the given radius of the focal individual. shape (#steps,)
        alt_countsl: list of 2d numpy arrays containing the fraction of individuals within the given radius of a focal individuals alternative step.
                     shape (ind, frame)
    
    """
    chosen_trail_vals = []
    alt_trail_valsl = []
    for act_p, alt_p in zip(act_pts, alt_pts):
        if not np.any(np.isnan(act_p)):
            chosen_trail_vals.append(get_trail_values_from_utm(trail_mask, mask_info_df, act_p))
            alt_trail_valsl.append(get_trail_values_from_utm(trail_mask, mask_info_df, alt_p))

    return chosen_trail_vals, alt_trail_valsl





def sd_feature(alt_pts, act_pts, r, fraction=True):
    """
    Calculates fraction of all group mates within a given radius of a focal indivdiual and its alternative steps
    
    Inputs:
        alt_pts: list of 3D numpy arrays with new xy positions of animals in pts_act. alt_pts[0] = xy coordinates
                 of an alternative step for each individual
        act_pts: 3D numpy array with raw xy positions of animals in space with shape (individual, frame, 2)
        r: int; radius around focal individual
    Output:
        individual_specific_counts_l: 1D numpy array containing the fraction of individuals within the given radius of the focal individual. shape (#steps,)
        alt_countsl: list of 2d numpy arrays containing the fraction of individuals within the given radius of a focal individuals alternative step.
                     shape (ind, frame)
    
    """
    if isinstance(act_pts, np.ndarray):
        counts = calculate_fraction(act_pts, act_pts, r, fraction)
        
        alt_countsl = []
        for step in range(len(alt_pts)):
            alt_counts = calculate_fraction(act_pts, alt_pts[step], r, fraction)
            alt_countsl.append(alt_counts)
        
        return counts, alt_countsl
    
    if isinstance(act_pts, list):
        counts_l = calculate_fraction(act_pts, act_pts, r, fraction)
        
        #counts_l is a 3D np array. code below extracts the focal ind data from that array, other info is not needed
        individual_specific_counts_l = []

        for ind in range(len(act_pts)):
            counts = counts_l[ind][ind]
            individual_specific_counts_l.append(counts)

        alt_countsl = calculate_fraction(act_pts, alt_pts, r, fraction)

        return individual_specific_counts_l, alt_countsl




def count_ind(act_pts, focal, r, time_index, time, temporal_data=False, double_count=True, fraction=False):
    """
    Calcualtes number of individuals that have occupied a potential location within the past time
    
    Inputs:
        act_pts: 3D numpy array (ind, frame, xy)
        focal: 3D numpy array (ind, frame, xy). Focal indivdual
        r: int; radius around focal individual
        time: int. How many frames to look back through
    Output:
        counts: 2D numpy array containing the number of individuals that have occupied a potential location within the past time
        If fraction==True:
            counts: 2D numpy array containing the fraction of individuals that have occupied a poetential location within the past. Indviduals will be
                    counted more than once
        If double_count==False:
            counts: 2D numpy array containing the number of individuals that have occupied a potential location within the past time. Individuals will not
                    be double counted
            If fraction==True:
                counts: 2D numpy array containing the fraction of individuals that have occupied a poetential location within the past. Indviduals will not be
                        counted more than once
    """
    if time > act_pts.shape[1]:
        raise ValueError('time exceeds length of video')
 
    if temporal_data==True:
        counts = np.zeros((act_pts.shape[0], act_pts.shape[1]))
    
        #need to fill in with NaNs to keep indexing intact when making a df
        counts[:,0:time] = counts[:,0:time] * np.NaN

        for ind in range(act_pts.shape[0]):
            for f in range(act_pts.shape[1]-time):
                loc = (act_pts[:,f:time+f,0] - focal[ind,time+f,0])**2 + (act_pts[:,f:time+f,1] - focal[ind,time+f,1])**2
                loc = np.where((loc<=r**2), 1, 0)
                loc[ind] = 0 #eliminating focal individual from count

                if double_count==False:

                    loc = np.any(loc, axis=1)
                    
                    if fraction==False:
                        count = np.sum(loc)
                    elif fraction==True:
                        num_ind_timeframe = np.sum(np.any(~np.isnan(act_pts[:,f:time+f,0]), axis=1))
                        count = np.sum(loc)/(num_ind_timeframe-1)
                else:
                    if fraction==False:
                        count = np.sum(loc)
                    elif fraction==True:
                        count = np.sum(loc)/(np.sum(np.sum(~np.isnan(act_pts[:,f:time+f,0]), axis=0))-1)

                counts[ind,f+time] = count
        return counts[:,:-1]
    
    else: #for spatial
        if isinstance(focal, np.ndarray): #for chosen steps
            counts_l = []
            for ind in range(len(time_index)):
                counts = np.zeros(time_index[ind].shape[0])

                for f in range(time_index[ind].shape[0]):
                    first_frame = time_index[ind][f]-time
                    end_frame = time_index[ind][f]

                    loc = (act_pts[:, first_frame : end_frame ,0] - focal[ind, end_frame ,0])**2 + (act_pts[:, first_frame : end_frame ,1] 
                                                                                                    - focal[ind, end_frame ,1])**2
                    loc = np.where((loc<=r**2), 1, 0)
                    loc[ind] = 0 #eliminating focal individual from count

                    if double_count==False:

                        loc = np.any(loc, axis=1)
                    #only thing I changed about code starts here
                        if fraction==False:
                            count = np.sum(loc) #this was directly from old code (with nice values)
                        elif fraction==True:
                            #divided by the number of individuals that could have potentially been counted throughout the timeframe
                            #i.e. if any individual shows up at all in the timeframe it will be used in the denominator
                            #never exceeds possible number of indiviuals
                            
                            num_ind_timeframe = np.sum(np.any(~np.isnan(act_pts[:,first_frame : end_frame,0]), axis=1))
                            count = np.sum(loc)/(num_ind_timeframe-1)
                    else:
                        if fraction==False:
                            count = np.sum(loc)
                        elif fraction==True:
                            count = np.sum(loc)/(np.sum(np.sum(~np.isnan(act_pts[:,first_frame : end_frame,0]), axis=0))-1)
                    #ends here
                    counts[f] = count
                    
                counts_l.append(counts[:-1])

            return counts_l
        
        if isinstance(focal, list): #for alternative steps
            counts_l = []
            for ind in range(len(time_index)):
                counts = np.zeros((focal[ind].shape[0], focal[ind].shape[1]))
                
                for step in range(focal[ind].shape[0]):
                    for f in range(focal[ind].shape[1]):
                        first_frame = time_index[ind][f]-time
                        end_frame = time_index[ind][f]

                        loc = (act_pts[:, first_frame : end_frame ,0] - focal[ind][step, f ,0])**2 + (act_pts[:, first_frame : end_frame ,1] 
                                                                                                      - focal[ind][step, f ,1])**2
                        loc = np.where((loc<=r**2), 1, 0)
                        loc[ind] = 0 #eliminating focal individual from count

                        if double_count==False:

                            loc = np.any(loc, axis=1)
                        #changes start (same as above)
                            if fraction==False:
                                count = np.sum(loc)
                            elif fraction==True:
                                num_ind_timeframe = np.sum(np.any(~np.isnan(act_pts[:,first_frame : end_frame,0]), axis=1))
                                count = np.sum(loc)/(num_ind_timeframe-1)

                        else:
                            if fraction==False:
                                count = np.sum(loc)
                            elif fraction==True:
                                count = np.sum(loc)/(np.sum(np.sum(~np.isnan(act_pts[:,first_frame : end_frame,0]), axis=0))-1)
                        #changes end
                        counts[step, f] = count
                counts_l.append(counts[:,:-1])

            return counts_l




##still working on this

def cal_ind_figure(act_pts, focal_pt, resolution, buffer, frame, individual, r, fraction=True):
    """
    Calcualtes number of individuals within a given radius of a focal individual
    
    Inputs:
        act_pts: 3D numpy array (ind, frame, xy)
        focal_pt: 1D numpy array containing x coordinate and y coordinate of individual that will be center of figure
        resolution: how many points per 1 increment moved
        buffer: max distance away from center in both the +-x and +-y direction
        frame: int; at which frame the xy positions for each individual are being extracted
        individual: individual being selected as the focal individual
        r: int; radius around focal individual
        num_ind_drop: number of individuals to drop from the data (problem with too many NaNs)
    Output:
        counts: 2D numpy array containing the fraction of individuals within the given radius of the focal individual
    
    """
    counts = np.zeros((int((buffer*resolution)*2+1), int((buffer*resolution)*2+1)))
    column = 0
    row = int((buffer*resolution)*2)
    
    for x in np.linspace(focal_pt[0]-buffer, focal_pt[0]+buffer, int((buffer*resolution)*2+1)):
        for y in np.linspace(focal_pt[1]-buffer, focal_pt[1]+buffer, int((buffer*resolution)*2+1)):
        
            loc = (act_pts[:, 0:frame, 0] - x)**2 + (act_pts[:, 0:frame, 1] - y)**2 #change is instead of looking at all ind locations in 1 frame look at all ind locations up until frame
            loc = np.where((loc<=r**2), 1, 0)
            loc[individual] = 0 #eliminating focal individual from count

            if fraction==True:
                count = np.nansum(loc, axis=0)/(np.sum(~np.isnan(act_pts[:,:,0]), axis=0)-1)
            elif fraction==False:
                count = np.nansum(loc, axis=0)
            
            counts[row, column] = count
            row = row - 1
        column = column + 1
        row = int((buffer*resolution)*2)
    return counts




def ru_feature(alt_pts, act_pts, r, time_index, time, temporal_data=False, double_count=True, fraction=False):
    """
    Calculates number of individuals that have occupied a potential location within the past 4.5min for a focal indivdiual and its alternative steps
    
    Inputs:
        time: int. How many frames to look back through (30 frames = 1sec)
        alt_pts: list of 3D numpy arrays with new xy positions of animals in pts_act. alt_pts[0] = xy coordinates
                 of an alternative step for each individual
        act_pts: 3D numpy array with raw xy positions of animals in space with shape (individual, frame, 2)
        r: int; radius around focal individual
    Output:
        counts: 1D numpy array containing the fraction of individuals within the given radius of the focal individual. shape (ind, frame)
        alt_countsl: list of 2d numpy arrays containing the fraction of individuals within the given radius of a focal individuals alternative step.
                     shape (ind, frame)
    
    """
    if temporal_data==True:
        counts = count_ind(act_pts, act_pts, r, time_index, time, temporal_data, double_count, fraction)
        
        alt_countsl = []
        for step in range(len(alt_pts)):
            alt_counts = count_ind(act_pts, alt_pts[step], r, time_index, time, temporal_data, double_count, fraction)
            alt_countsl.append(alt_counts)
        return counts, alt_countsl
    
    else:
        counts_l = count_ind(act_pts, act_pts, r, time_index, time, temporal_data, double_count, fraction)

        alt_countsl = count_ind(act_pts, alt_pts, r, time_index, time, temporal_data, double_count, fraction)
        
        return counts_l, alt_countsl




def cal_fraction_figure(act_pts, focal_pt, resolution, buffer, frame, individual, r, num_ind_drop=0, fraction=True):
    """
    Calcualtes fraction of individuals within a given radius of a focal individual
    
    Inputs:
        act_pts: 3D numpy array (ind, frame, xy)
        focal_pt: 1D numpy array containing x coordinate and y coordinate of individual that will be center of figure
        resolution: how many points per 1 increment moved
        buffer: max distance away from center in both the +-x and +-y direction
        frame: int; at which frame the xy positions for each individual are being extracted
        individual: individual being selected as the focal individual
        r: int; radius around focal individual
        num_ind_drop: number of individuals to drop from the data (problem with too many NaNs)
    Output:
        counts: 2D numpy array containing the fraction of individuals within the given radius of the focal individual
    
    """
    counts = np.zeros((int((buffer*resolution)*2+1), int((buffer*resolution)*2+1)))
    column = 0
    row = int((buffer*resolution)*2)
    
    for x in np.linspace(focal_pt[0]-buffer, focal_pt[0]+buffer, int((buffer*resolution)*2+1)):
        for y in np.linspace(focal_pt[1]-buffer, focal_pt[1]+buffer, int((buffer*resolution)*2+1)):
        
            loc = (act_pts[:, frame, 0] - x)**2 + (act_pts[:, frame, 1] - y)**2
            loc = np.where((loc<=r**2), 1, 0)
            loc[individual] = 0 #eliminating focal individual from count
            
            if fraction==True:
                count = np.nansum(loc, axis=0)/((act_pts.shape[0]-1) - num_ind_drop)
            if fraction==False:
                count = np.nansum(loc, axis=0)
            
            counts[row, column] = count
            row = row - 1
        column = column + 1
        row = int((buffer*resolution)*2)
    return counts




def spatial_dis(act_pts, r, resolution=1):
    """
    Adjusts the scale from temporal to spatial. Points are now recorded at a constant spatial rate e.g. one point for every r meters moved.
    Locations are stored in spatial. The corresponding frame or timepoint for the location in spatial is stored in time_index
    
    Inputs:
        act_pts: 3D numpy array (ind, frame, xy)
        r: int; distance focal individual should travel before a step is recorded
        resolution: int, at every 'blank' frame, data will be pulled
    Output:
        new_act_pts_l:  list of 3D numpy arrays containing the xy coordinates for a focal individual after it has moved r meters as well
                        as the xy coordinates of every other individual at that timepoint. The position in the list identifies the focal
                        individual e.g. in new_act_pts_l[0] individual 0 is the focal individual
        time_index: list of 1D numpy arrays containing the indexs for the frame at which each individual has moved at least r meters
        spatial: list of 2D numpy arrays containing the xy coordinates for each individual after they have moved at least r meters
        drop_lst: list of individuals to be dropped from dataset. Will be dropped for insufficient amount of data
    
    """
    time_index = []
    spatial = []
    drop_lst = []

    for ind in range(act_pts.shape[0]):
        frame = 0
        temp = []
        spat = []
#         print(ind)
        #should these be here? puts in the first xy coordinate
        temp.append(0)
        spat.append(act_pts[ind,0,:])
        for f in range(0, act_pts.shape[1], resolution):
            loc = (act_pts[ind,f,0] - act_pts[ind,frame,0])**2 + (act_pts[ind,f,1] - act_pts[ind,frame,1])**2
            if loc>=r**2:
                frame = f
                temp.append(f)
                spat.append(act_pts[ind,f,:])
            
        temp = np.asarray(temp)
        
        #Less than equal to 3 because need at least 3 xy locations to do the calc change in direction
        if len(temp)<=3:
            spat = np.asarray(spat)
            drop_lst.append(ind)
        else:
            spat = np.vstack(spat)
            
        time_index.append(temp)
        spatial.append(spat)
    
    new_act_pts_l = []
    
    for ind in range(len(time_index)):
        coords = np.zeros((len(spatial), spatial[ind].shape[0], 2))
        for s in range(time_index[ind].shape[0]):
            new_act_pts = act_pts[:, time_index[ind][s], :]
            coords[:, s, :] = new_act_pts
        new_act_pts_l.append(coords)
        
    return new_act_pts_l, spatial, time_index, drop_lst




def beta_df(act_pts, scale_adjustment, drop_lstl, filename):
    """
    Generates a df consisting of beta values for each individual, for each feature, at every time series
    
    Inputs:
        act_pts: 3D numpy array with raw xy positions of animals in space with shape (individual, frame, 2)
        scale_adjustment: list of time scale adjustments (ints)
        drop_lstl: list of lists containing individuals to drop for each time series in times
        filename: string, name of file from mclogit
    Output:
        beta_df: df consisting of beta values for each individual, for each feature, at every time series
    
    """
    if isinstance(act_pts, np.ndarray):
        ind_idl = []
        t_idl = []

        for time in range(len(scale_adjustment)):

            #create 'individual' and 'timescale' column
            ind_id = np.zeros((1, act_pts.shape[0]+1))
            t_id = np.zeros((1, act_pts.shape[0]+1))

            for i in range(ind_id.shape[1]):
                ind_id[:,i] = ind_id[:,i] + i
            #     ind_id[:,-1] = -1 #bad when plotting

            for i in range(t_id.shape[0]):
                t_id[i] = t_id[i] + scale_adjustment[time]

            #drop individual with 'bad data'
            ind_id = np.delete(ind_id, drop_lstl[time], axis=1)
            t_id = np.delete(t_id, drop_lstl[time], axis=1)

            ind_id = np.concatenate((ind_id), axis=0)
            t_id = np.concatenate((t_id), axis=0)

            ind_idl.append(ind_id)
            t_idl.append(t_id)

        ind_id = np.concatenate(ind_idl)
        t_id = np.concatenate(t_idl)

        #put all columns together in one dataframe
        beta_df = pd.read_csv(filename + "_output.csv")
        beta_df.insert(0, 'individual', ind_id, True)
        beta_df.insert(beta_df.shape[1], 'timescale', t_id, True)

    elif isinstance(act_pts, list):
        
        ind_idl = []
        step_size_idl = []
        
        for scale in range(len(scale_adjustment)):
            ind_id = np.arange(len(act_pts)+1)
            step_size_id = np.ones(len(act_pts)+1) * scale_adjustment[scale]
            
            ind_id = np.delete(ind_id, drop_lstl[scale])
            step_size_id = np.delete(step_size_id, drop_lstl[scale])
            
            ind_idl.append(ind_id)
            step_size_idl.append(step_size_id)
            
        ind_id = np.concatenate(ind_idl)
        step_size_id = np.concatenate(step_size_idl)
        
        #put all columns together in one dataframe
        beta_df = pd.read_csv(filename + "_output.csv")
        beta_df.insert(0, 'individual', ind_id, True)
        beta_df.insert(beta_df.shape[1], 'step_size', step_size_id, True)
        
    return beta_df




def multi_observations_beta_df(scale_adjustment, radius_around_ind, filename):
    """
    Generates a df consisting of beta values for entire observations, for each feature
    
    Inputs:
        scale_adjustment: list of scale adjustments (ints)
        filename: string, name of file from mclogit
    Output:
        beta_df: df consisting of beta values for each observations, for each feature
    
    """
    step_size = []
    radius_id = []
    for scale in range(len(scale_adjustment)):
        step_size_id = np.ones(len(radius_around_ind)) * scale_adjustment[scale]
        radius_around_ind_id = np.zeros(len(radius_around_ind)) + np.asarray(radius_around_ind)
        
        step_size.append(step_size_id)
        radius_id.append(radius_around_ind_id)
    
    radius_id = np.concatenate(radius_id)
    step_size_id = np.concatenate(step_size)
    
    #put all columns together in one dataframe
    beta_df = pd.read_csv(filename + "_output.csv")
    beta_df.insert(0, 'step_size', step_size_id, True)
    beta_df.insert(beta_df.shape[1], 'radius_size_around_ind', radius_id, True)
    
    
    return beta_df

def calculate_probability(step_size, resolution, list_of_observations, radius_around_ind=None, 
                          look_back=None, fraction=True, group_center_feature=False, social_density=False, 
                          recently_used_space=False, temporal_data=False, double_count=False,
                         trails=False, list_of_observation_names=None, observation_maps_dict=None):
    """
    Inputs:
        step_size: list of ints, how far animal has travelled each step
        resolution: int, at every 'blank' frame, data will be pulled
        list_of_observations: list of 3D numpy arrays containing tracks for each individual in each observation
        radius_around_ind: list of ints; radius around focal individual
        look_back: int. How many frames to look back through (30 frames = 1sec)
        trails: True if calculate trail probabilities
        list_of_observation_names: name of each observation in list_of_observations
        observation_maps_dict: if calculcating trail probabilities: has trail_masks and info
        
    Output:
        probabilities: list of probabilities associated with 
        keys:
        event_counts: 
    """
    
    #chosen_l: list of 1D arrays; value for every step/individual; len(chosen_l)=number of individuals; chosen_l[0] corresponds to ind0
    #alt_l: list of 2D arrays; values for each alternative step for each ind; len(alt_l)=num of individuals; alt_l[0].shape= (5,#steps)
    
    
    probabilities = []
    keys = []
    event_counts = []
    
    if group_center_feature==True:
    
        for distance in step_size:
            all_observations_one_step_size = []
            for obsv in list_of_observations:
                new_coords_l, spatial, _, _ = spatial_dis(obsv, distance, resolution)
                alt_steps = random_steps(spatial, distance)
                cl = group_center(new_coords_l)
                
                chosen_l, alt_l = dc_feature(cl, spatial, distance)

                diff_l = []
                for ind in range(len(chosen_l)):
                    step = np.random.randint(0, len(alt_l[ind]))
                    diff = chosen_l[ind] - alt_l[ind][step]
                    diff_l.append(diff)
                diff_array = np.concatenate(diff_l)
                diff_array = np.around(diff_array, decimals=1)
                    
                all_observations_one_step_size.append(diff_array)
            all_observations = np.concatenate(all_observations_one_step_size)

            outcome_count = Counter(all_observations)
            event_count = Counter(np.abs(all_observations))

            probabilities_list = []
            keys_list = []
            for key, value in outcome_count.items():
                if key == 0:
                    probability = 0.5
                else:
                    probability = value/event_count[np.abs(key)]
                probabilities_list.append(probability)
                keys_list.append(key)

            probabilities.append(np.asarray(probabilities_list))
            keys.append(np.asarray(keys_list))
            event_counts.append(all_observations)
            
    elif trails:
        for distance in step_size:
            diff_all_obsv_one_step = [] #all observations for a single radius and step size
            for obsv, obs_name in zip(list_of_observations, list_of_observation_names):
                new_coords_l, spatial, time_index, _ = spatial_dis(obsv, distance, resolution)
                alt_steps = random_steps(spatial, distance)
                

                trail_mask = observation_maps_dict[obs_name]['trail_mask']
                mask_info_df = observation_maps_dict[obs_name]['mask_info_df']
                chosen_l, alt_l = trail_feature(alt_steps, spatial, trail_mask, mask_info_df)
                


                diff_l = []
                for ind in range(len(chosen_l)):
                    step = np.random.randint(0, len(alt_l[ind]))
                    diff = chosen_l[ind] - alt_l[ind][step]
                    diff_l.append(diff)
                diff_array = np.concatenate(diff_l)
                diff_all_obsv_one_step.append(diff_array)
            if not diff_all_obsv_one_step:
                probabilities.append(None)
                keys.append(None)
                event_counts.append(None)
                continue
            diff_all_obsv_one_step_concatenated = np.concatenate(diff_all_obsv_one_step)

            outcome_count = Counter(diff_all_obsv_one_step_concatenated)
            event_count = Counter(np.abs(diff_all_obsv_one_step_concatenated))
            
#             print(outcome_count)

            probabilities_list = []
            keys_list = []
            for key, value in outcome_count.items():
                if key == 0:
                    probability = 0.5
                elif event_count[np.abs(key)] == 0:
                    probability = 0
                else:
                    probability = value/event_count[np.abs(key)]
                probabilities_list.append(probability)
                keys_list.append(key)

            probabilities.append(np.asarray(probabilities_list))
            keys.append(np.asarray(keys_list))
            event_counts.append(diff_all_obsv_one_step_concatenated)
        
        
    else:
        for distance in step_size:
            for radius in radius_around_ind:
                diff_all_obsv_one_radius = [] #all observations for a single radius and step size
                for obsv in list_of_observations:
                    new_coords_l, spatial, time_index, _ = spatial_dis(obsv, distance, resolution)
                    alt_steps = random_steps(spatial, distance)

                    if social_density==True:
                        chosen_l, alt_l = sd_feature(alt_steps, new_coords_l, radius, fraction)
    
                    elif recently_used_space==True:
                        chosen_l, alt_l = ru_feature(alt_steps, obsv, radius, time_index, look_back, temporal_data, double_count, fraction)

                    diff_l = []
                    for ind in range(len(chosen_l)):
                        step = np.random.randint(0, len(alt_l[ind]))
                        diff = chosen_l[ind] - alt_l[ind][step]
                        diff_l.append(diff)
                    diff_array = np.concatenate(diff_l)
                    
                    if fraction==True:
                        diff_array = np.around(diff_array, decimals=1)
                    diff_all_obsv_one_radius.append(diff_array)
                diff_all_obsv_one_radius_concatenated = np.concatenate(diff_all_obsv_one_radius)

                outcome_count = Counter(diff_all_obsv_one_radius_concatenated)
                event_count = Counter(np.abs(diff_all_obsv_one_radius_concatenated))

                probabilities_list = []
                keys_list = []
                for key, value in outcome_count.items():
                    if key == 0:
                        probability = 0.5
                    else:
                        probability = value/event_count[np.abs(key)]
                    probabilities_list.append(probability)
                    keys_list.append(key)

                probabilities.append(np.asarray(probabilities_list))
                keys.append(np.asarray(keys_list))
                event_counts.append(diff_all_obsv_one_radius_concatenated)
    
    #add in missing zero values for when probability is 1
    indexes = []
    for radius in range(len(probabilities)):
        indexes.append(np.argwhere(probabilities[radius]==1))

    for indx in range(len(indexes)):
        for diff in range(indexes[indx].shape[0]):
            missing_zero_values_figure = keys[indx][indexes[indx][diff,0]] * -1

            keys[indx] = np.append(keys[indx], missing_zero_values_figure)
            probabilities[indx] = np.append(probabilities[indx], 0)
    
    return keys, probabilities, event_counts