import numpy as np
import datetime as dt
import scipy.io as sio
import pandas as pd

"""
Convert the data from matlab format to python format of the Cosentino DB
"""


def mat2df(mat_path, name):
    """
    Read the .mat files and convert them to a pandas data frame
    name: HQ or LQ
    """
    # Read the data and convert it to DataFrame
    mat = sio.loadmat(mat_path)['LogModel' + name]
    metadata = mat.dtype

    df = pd.DataFrame()

    df['id_mat'] = mat['id'][0].astype(np.uint)
    df = df.set_index('id_mat')
    dates = []
    waves = []
    for i in np.arange(len(mat['date'][0])):
        dates.append(mat['date'][0][i][0][0])
        waves.append(mat['wave'][0][i].astype(np.float))
    df['datetime'] = pd.to_datetime(np.array(dates)-719529, unit='D')
    df['wave'] = waves
    df['P'] = mat['P'][0].astype(np.uint)

    return df


def mat2df_test(mat_path):
    """
    Read the .mat files and convert them to a pandas data frame
    """
    # Read the data and convert it to DataFrame
    mat = sio.loadmat(mat_path)['TestingData']
    metadata = mat.dtype

    df = pd.DataFrame()

    df['id_mat'] = mat['id'][0].astype(np.uint)
    df = df.set_index('id_mat')
    dates = []
    waves = []
    for i in np.arange(len(mat['realdate'][0])):
        dates.append(mat['realdate'][0][i][0])
        waves.append(mat['wave'][0][i].astype(np.float))
    df['datetime'] = pd.to_datetime(dates)
    df['startSample'] = mat['startSample'][0].astype(np.uint)
    df['wave'] = waves
    df['duration'] = mat['duration'][0].astype(np.uint)
    df['CF'] = mat['CF'][0].astype(np.double)
    df['BW'] = mat['BW'][0].astype(np.double)
    df['Q'] = mat['Q'][0].astype(np.double)
    df['XC'] = mat['XC'][0].astype(np.double)
    df['ratio'] = mat['ratio'][0].astype(np.double)
    df['ManualAsign'] = mat['ManualAsign'][0].astype(np.uint)
    df['ClassifiedAs'] = mat['ClassifiedAs'][0].astype(np.uint)

    return df


def all_mat2pkl(path_dict):
    """
    Save the data of all the files to a pkl file for easier reading 
    """
    df_list = []
    for name, mat_path in path_dict.items():
        if name == 'test':
            df = self.mat2df_test(mat_path, name)
        else:
            df = self.mat2df(mat_path, name)
        
        df.to_pickle(mat_path.replace('.mat', '.pkl'))   
        df_list.append(df)   
    
    return df_list


def load_pickle_data(path_list):
    """
    Load all the data
    """
    df_list = []
    for path in path_list:
        df = pd.read_pickle(path)
        df_list.append(df)

    return df_list
    
