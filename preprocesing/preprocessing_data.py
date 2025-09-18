import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def standarize(df:pd.DataFrame):
    """
    Standardizes the input DataFrame by converting units and filtering rows.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least the columns 'range' and 't'.

    Returns:
        pd.DataFrame: The standardized DataFrame with:
            - 'range' column converted from original units to kilometers (divided by 1000).
            - Rows filtered to keep only those with 'range' > 0 and 't' < 1181241184.
            - 't' column converted from microseconds to milliseconds (multiplied by 1e-6).
    """
    df['range'] /= 1000  # Convertir a metros
    df = df[(df['range'] > 0.) & (df['t'] < 1181241184)]
    df['t'] *= 1e-6  # Convertir a milisegundos
    return df

def round_coords(df:pd.DataFrame, precision:int=2):
    """
    Rounds the coordinates in the DataFrame to a specified precision.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'x', 'y', and 'z' columns.
        precision (int): Number of decimal places to round to. Default is 2.

    Returns:
        pd.DataFrame: The DataFrame with rounded coordinates.
    """
    df['x'] = df['x'].round(precision)
    df['y'] = df['y'].round(precision)
    df['z'] = df['z'].round(precision)
    return df

def get_road_mask(road_df:pd.DataFrame, df:pd.DataFrame, threshold:float=0.4):
    """
    Identifies points in a DataFrame that are within a specified distance threshold from any point in a road DataFrame.
    Args:
        road_df (pd.DataFrame): DataFrame containing road points with columns 'x', 'y', and 'z'.
        df (pd.DataFrame): DataFrame containing points to be checked, with columns 'x', 'y', and 'z'.
        threshold (float, optional): Maximum distance to consider a point as being on the road. Defaults to 0.4.
    Returns:
        np.ndarray: Boolean mask array indicating which points in `df` are within the threshold distance from any road point.
    """

    road_points = road_df[['x', 'y', 'z']].to_numpy()
    tree = KDTree(road_points)

    points = df[['x', 'y', 'z']].to_numpy()
    distances, _ = tree.query(points)

    mask = distances < threshold
    return mask

