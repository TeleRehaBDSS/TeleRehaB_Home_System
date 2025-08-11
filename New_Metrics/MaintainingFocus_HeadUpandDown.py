import pandas as pd
import numpy as np
import os
from datetime import datetime
import statistics 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json
import scipy


def process_imu_data(imu_data_lists, fs, plotdiagrams=False):
    # Ensure lists are not empty and convert to DataFrames
    dataframes = []
    c = 0;
    for imu_data in imu_data_lists:
        if imu_data:
            columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
            df = pd.DataFrame(imu_data, columns=columns)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df.sort_values(by='Timestamp')
            df.set_index('Timestamp', inplace=True)
            dataframes.append(df)
            c = c + 1

    if not dataframes:
        return None  # No data to process

    # Find the common time range
    max_start_time = max(df.index[0] for df in dataframes)
    min_end_time = min(df.index[-1] for df in dataframes)

    # Filter dataframes to the common time range
    dataframes = [df[max_start_time:min_end_time] for df in dataframes]

    # Determine the maximum number of samples across lists
    max_samples = max(len(df) for df in dataframes)

    # Resample and interpolate dataframes to have the same number of samples
    resampled_dataframes = []
    for df in dataframes:
        df_resampled = df.resample(f'{1000//fs}ms').mean()  # Resampling to match the sampling frequency
        df_interpolated = df_resampled.interpolate(method='time')
        df_interpolated = df_interpolated.dropna().head(max_samples)
        resampled_dataframes.append(df_interpolated)

    if plotdiagrams:
        for idx, df in enumerate(resampled_dataframes):
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['W(number)'], label='W')
            plt.plot(df.index, df['X(number)'], label='X')
            plt.plot(df.index, df['Y (number)'], label='Y')
            plt.plot(df.index, df['Z (number)'], label='Z')
            plt.xlabel('Timestamp')
            plt.ylabel('Quaternion Components')
            plt.title(f'IMU {idx+1} Quaternion Components (W, X, Y, Z) over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'quaternion_components_plot_{idx+1}.png')
            # plt.show()

    # Convert the processed DataFrames back to lists
    resampled_lists = [df.reset_index().values.tolist() for df in resampled_dataframes]

    return resampled_lists, c


def reformat_sensor_data(sensor_data_list):
    if not sensor_data_list:
        return []

    # Get the reference timestamp
    reference_timestamp = sensor_data_list[0].timestamp

    reformatted_data = []

    # Iterate through the sensor data list
    for data in sensor_data_list:
        timestamp = data.timestamp
        elapsed_time = timestamp - reference_timestamp
        reformatted_entry = [timestamp, elapsed_time, data.w, data.x, data.y, data.z]
        reformatted_data.append(reformatted_entry)

    return reformatted_data

def plotIMUDATA(Limu, x, filename):

    time = [row[0] for row in Limu]
    w = [row[x] for row in Limu]

    plt.figure(figsize=(10, 6))  
    plt.plot(time, w, marker='o', linestyle='-', color='b')  
    plt.title('Time vs W Component')
    plt.xlabel('Time (sec)')
    plt.ylabel('W component of quaternion')
    plt.grid(True)  


def interpolate_imu_data(imu_data, starttime, endtime, N):
    """
    Interpolate IMU data (w, x, y, z) between starttime and endtime into N samples.

    Parameters:
    imu_data (list of lists): The IMU data in format [time, w, x, y, z, _, _].
    starttime (float): The start time for interpolation.
    endtime (float): The end time for interpolation.
    N (int): Number of samples to interpolate.

    Returns:
    list of lists: Interpolated IMU data with N entries.
    """
    
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     nyq = 0.5 * fs  
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs  
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Check if data length is greater than default padlen, adjust if necessary
    default_padlen = 2 * max(len(b), len(a)) - 1
    if len(data) <= default_padlen:
        padlen = len(data) - 1
    else:
        padlen = default_padlen

    y = filtfilt(b, a, data, padlen=padlen)
    return y


def striplist(L):
    A = []
    for item in L:
        t = item[1:-1]
        if ',' in t:
            t = t.split(',')
        else:
            t = t.split(' ')
        if "(number" not in t:
            A.append([t[-7],t[-5],t[-4],t[-3],t[-2],t[-1]])
    return A

def get_metrics(imu1,imu2,imu3,imu4, counter):
    Limu1 = reformat_sensor_data(imu1)
    Limu2 = reformat_sensor_data(imu2)
    Limu3 = reformat_sensor_data(imu3)   
    Limu4 = reformat_sensor_data(imu4)

    imu_data_lists = [Limu1, Limu2, Limu3, Limu4]
    processed_dataframes, c = process_imu_data(imu_data_lists, 50, True)
    Limu1 = processed_dataframes[0]
    if (c >= 2):
        Limu2 = processed_dataframes[1]
    if (c >= 3):
        Limu3 = processed_dataframes[2]
    if (c >= 4):
        Limu4 = processed_dataframes[3]

    if(len(Limu1) > 0):
        returnedJson = getMetricsSittingOld02(Limu1, False) 
        return returnedJson

def getMetricsSittingOld02(Limu1, plotdiagrams):
   
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y(number)', 'Z(number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['Timestamp'] = pd.to_datetime(df_Limu1['Timestamp'])
    df_Limu1 = df_Limu1.sort_values(by='Timestamp')
    df_Limu1.set_index('Timestamp', inplace=True)
    

    if (plotdiagrams):
        plt.figure(figsize=(10, 6))
        plt.xlabel('Timestamp')
        plt.ylabel('Quaternion Components')
        plt.title('Quaternion Components (W, X, Y, Z) over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig('quaternion_components_plot!!!!!!!.png')
        #plt.show()
    
    quaternions = df_Limu1[['X(number)', 'Y(number)', 'Z(number)', 'W(number)']].to_numpy()
    rotations = R.from_quat(quaternions)
    euler_angles = rotations.as_euler('xyz', degrees=False)
    euler_df = pd.DataFrame(euler_angles, columns=['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)'])
    euler_angles_degrees = rotations.as_euler('xyz', degrees=True)
    euler_df_degrees = pd.DataFrame(euler_angles_degrees, columns=['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)'])
   
    
    if (plotdiagrams):
        plt.figure(figsize=(12, 8))
        plt.plot(euler_df_degrees.index, euler_df_degrees['Roll (degrees)'], label='Roll', linewidth=1)
        plt.plot(euler_df_degrees.index, euler_df_degrees['Pitch (degrees)'], label='Pitch', linewidth=1)
        plt.plot(euler_df_degrees.index, euler_df_degrees['Yaw (degrees)'], label='Yaw', linewidth=1)

        plt.xlabel('Timestamp')
        plt.ylabel('Euler Angles (degrees)')
        plt.title('Euler Angles (Roll, Pitch, Yaw) over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()  
        plt.show()

    fs = 50
    cutoff = 0.75

    pitch_filtered = butter_lowpass_filter(euler_df_degrees['Pitch (degrees)'], cutoff, fs, order=5)
    if (plotdiagrams):
        plt.figure(figsize=(12, 6))
        plt.plot(euler_df_degrees.index, euler_df_degrees['Pitch (degrees)'], label='Original Pitch', linewidth=1, alpha=0.5)
        plt.plot(euler_df_degrees.index, pitch_filtered, label='Filtered Yaw', linewidth=2)
        plt.xlabel('Timestamp')
        plt.ylabel('Pitch (degrees)')
        plt.title('Pitch Signal Filtering')
        plt.legend()
        plt.show()

    peaks, _ = find_peaks(pitch_filtered)

    valleys, _ = find_peaks(-pitch_filtered)

    # print("peaks ", peaks)
    # print("valleys ", valleys)
    if(len(peaks) == 0):
        return 0
    if(len(valleys) == 0):
        return 0

    if valleys[0] > peaks[0]:
        peaks = peaks[1:]  
    if peaks[-1] < valleys[-1]:
        valleys = valleys[:-1]  
        
    movement_pairs = []

    for i in range(min(len(peaks), len(valleys))):
        movement_pairs.append((valleys[i], peaks[i]))

    #print("Movement pairs (as index positions):", movement_pairs)

    if (plotdiagrams):
        plt.figure(figsize=(12, 6))
        plt.plot(pitch_filtered, label='Filtered Pitch', linewidth=1)
        plt.plot(peaks, pitch_filtered[peaks], "x", label='Maxima')
        plt.plot(valleys, pitch_filtered[valleys], "o", label='Minima')
        plt.xlabel('Sample index')
        plt.ylabel('Pitch (degrees)')
        plt.title('Pitch Signal with Detected Movements')
        plt.legend()
        plt.show()

    movement_ranges = []

    for valley, peak in movement_pairs:
        movement_range = pitch_filtered[peak] - pitch_filtered[valley]
        movement_ranges.append(movement_range)

    # for i, movement_range in enumerate(movement_ranges):
    #     print(f"Movement {i+1}: Range = {movement_range:.2f} degrees")
    significant_movements = [(pair, mrange) for pair, mrange in zip(movement_pairs, movement_ranges) if mrange >= 5]

    filtered_pairs = [pair for pair, range in significant_movements]
    filtered_ranges = [mrange for pair, mrange in significant_movements]

    # for i, (pair, mrange) in enumerate(significant_movements):
    #     print(f"Significant Movement {i+1}: Pair = {pair}, Range = {mrange:.2f} degrees")

    movement_durations = []

    for start, end in filtered_pairs:
        start_time = df_Limu1.iloc[start].name
        end_time = df_Limu1.iloc[end].name
        duration = (end_time - start_time).total_seconds()
        movement_durations.append(duration)

    total_duration_seconds = (df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds()

    if (total_duration_seconds > 0):
        pace = len(filtered_pairs) / total_duration_seconds  
    else:
        pace = -1;

    if (len(filtered_ranges) > 0):
        mean_range = np.mean(filtered_ranges)
        std_range = np.std(filtered_ranges, ddof=1) 
    else:
        mean_range = -1;
        std_range = -1;

    if (len(movement_durations) > 0):
        mean_duration = np.mean(movement_durations)
        std_duration = np.std(movement_durations, ddof=1)
    else:
        mean_duration = -1
        std_duration = -1        



    metrics_data = {
        "total_metrics": {
            "number_of_movements": int(len(filtered_pairs)),
            "pace_movements_per_second": float(pace),
            "mean_range_degrees": float(mean_range),
            "std_range_degrees": float(std_range),
            "mean_duration_seconds": float(mean_duration),
            "std_duration_seconds": float(std_duration),
            "exercise_duration_seconds":total_duration_seconds
        }
    }
    print (metrics_data)

    return json.dumps(metrics_data, indent=4)

