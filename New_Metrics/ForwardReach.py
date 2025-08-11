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

def detect_rotation_point(y_data):
    # Detect the rotation point based on the largest change in y-axis data
    rotation_index = np.argmax(np.abs(np.diff(y_data)))
    return rotation_index

def detect_movements(data, prominence, is_negative=False):
    if is_negative:
        return find_peaks(-data, prominence=prominence)[0]
    else:
        return find_peaks(data, prominence=prominence)[0]

def calculate_metrics(timestamps, y_smooth, peaks, valleys):
    # Directly index into the DatetimeIndex for peaks and valleys, then calculate time differences in seconds
    all_movements_timestamps = np.sort(np.concatenate((timestamps[peaks].values, timestamps[valleys].values)))
    movement_durations = np.diff(all_movements_timestamps) / np.timedelta64(1, 's')  # Convert to seconds

    movement_ranges = [
        y_smooth[peaks[i]:peaks[i+1]].max() - y_smooth[peaks[i]:peaks[i+1]].min()
        for i in range(len(peaks) - 1)
    ] + [
        y_smooth[valleys[i]:valleys[i+1]].max() - y_smooth[valleys[i]:valleys[i+1]].min()
        for i in range(len(valleys) - 1)
    ]

    number_of_movements = len(peaks) + len(valleys)
    mean_duration = np.mean(movement_durations) if movement_durations.size > 0 else 0
    std_duration = np.std(movement_durations) if movement_durations.size > 0 else 0
    mean_combined_range_degrees = np.mean(movement_ranges) if movement_ranges else 0
    std_combined_range_degrees = np.std(movement_ranges) if movement_ranges else 0

    return {
        "number_of_movements": number_of_movements,
        "movement_duration": movement_durations.tolist(),
        "mean_duration_seconds": mean_duration,
        "std_duration_seconds": std_duration,
        "mean_combined_range_degrees": mean_combined_range_degrees,
        "std_combined_range_degrees": std_combined_range_degrees
    }

def process_imu_data(imu_data_lists, fs, plotdiagrams=True):
    # Ensure lists are not empty and convert to DataFrames
    dataframes = []
    initial_empty_lists = [len(imu_data) == 0 for imu_data in imu_data_lists]  # Track initially empty lists

    c = 0;
    for imu_data in imu_data_lists:
        if imu_data:
            #print('imu_data = ', imu_data)
            columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y(number)', 'Z(number)']
            df = pd.DataFrame(imu_data, columns=columns)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df.sort_values(by='Timestamp')
            df.set_index('Timestamp', inplace=True)
            dataframes.append(df)
            c = c + 1


    if not dataframes:
        print('no data to process')
        return None  # No data to process
    else:
        print(dataframes)

    # Find the common time range
    max_start_time = max(df.index[0] for df in dataframes)
    min_end_time = min(df.index[-1] for df in dataframes)

    print('max_start_time = ', max_start_time)
    print('min_end_time = ', min_end_time)

    # Filter dataframes to the common time range
    #dataframes = [df[max_start_time:min_end_time] for df in dataframes]

    # Determine the maximum number of samples across lists
    #max_samples = max(len(df) for df in dataframes)

    # Resample and interpolate dataframes to have the same number of samples
    resampled_dataframes = []
    for df in dataframes:
        #df_resampled = df.resample(f'{1000//fs}ms').mean()  # Resampling to match the sampling frequency
        #df_interpolated = df_resampled.interpolate(method='time')
        #df_interpolated = df_interpolated.dropna().head(max_samples)
        #resampled_dataframes.append(df_interpolated)
        resampled_dataframes.append(df)

    if plotdiagrams:
        for idx, df in enumerate(resampled_dataframes):
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['W(number)'], label='W')
            plt.plot(df.index, df['X(number)'], label='X')
            plt.plot(df.index, df['Y(number)'], label='Y')
            plt.plot(df.index, df['Z(number)'], label='Z')
            plt.xlabel('Timestamp')
            plt.ylabel('Quaternion Components')
            plt.title(f'IMU {idx+1} Quaternion Components (W, X, Y, Z) over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'quaternion_components_plot_{idx+1}.png')
            # plt.show()

    # Convert the processed DataFrames back to lists
    #resampled_lists = [df.reset_index().values.tolist() for df in resampled_dataframes]

    #20241013
    resampled_lists = []

    data_idx = 0
    for is_empty in initial_empty_lists:
        if is_empty:
            resampled_lists.append([])  # Keep the list empty if it was initially empty
        else:
            resampled_lists.append(resampled_dataframes[data_idx].reset_index().values.tolist())  # Add processed data
            data_idx += 1

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


# def interpolate_imu_data(imu_data, starttime, endtime, N):
      
#     Interpolate IMU data (w, x, y, z) between starttime and endtime into N samples.

#     Parameters:
#     imu_data (list of lists): The IMU data in format [time, w, x, y, z, _, _].
#     starttime (float): The start time for interpolation.
#     endtime (float): The end time for interpolation.
#     N (int): Number of samples to interpolate.

#     Returns:
#     list of lists: Interpolated IMU data with N entries.
      
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

    imu_data_lists1 = [Limu1, Limu2]
    imu_data_lists2 = [Limu3, Limu4]

    processed_dataframes1, c1 = process_imu_data(imu_data_lists1, 50, True)
    processed_dataframes2, c2 = process_imu_data(imu_data_lists2, 100, True)


    Limu1 = processed_dataframes1[0]
    Limu2 = processed_dataframes1[1]
    Limu3 = processed_dataframes2[0]
    Limu4 = processed_dataframes2[1]

    if(len(Limu2) > 0 and len(Limu3) > 0 and len(Limu4) > 0 ):
        print('procceding to metrics...')
        returnedJson = getMetricsForwardReach(Limu2, Limu3, Limu4 ,False) 
        return returnedJson

def getMetricsForwardReach(Limu1, Limu2, Limu3, plotdiagrams):
   
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y(number)', 'Z(number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['elapsed(time)'] = pd.to_datetime(df_Limu1['elapsed(time)'], unit='ms')
    df_Limu1 = df_Limu1.sort_values(by='elapsed(time)')
    df_Limu1.set_index('elapsed(time)', inplace=True)
    
    #Limu2
    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['elapsed(time)'] = pd.to_datetime(df_Limu2['elapsed(time)'], unit='ms')
    df_Limu2 = df_Limu2.sort_values(by='elapsed(time)')
    df_Limu2.set_index('elapsed(time)', inplace=True)

    #Limu3
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    df_Limu3['elapsed(time)'] = pd.to_datetime(df_Limu3['elapsed(time)'], unit='ms')
    df_Limu3 = df_Limu3.sort_values(by='elapsed(time)')
    df_Limu3.set_index('elapsed(time)', inplace=True)


    quaternions_df1 = df_Limu1;
    quaternions_df2 = df_Limu2;
    quaternions_df3 = df_Limu3;

    fs = 50
    cutoff = 0.5

    y_filtered_pelvis = butter_lowpass_filter(df_Limu1['Y(number)'], cutoff, fs, order=5)
    y_filtered_leftfoot = butter_lowpass_filter(df_Limu2['Y(number)'], cutoff, fs, order=5)
    
    # Detect rotation point from the filtered left foot data
    rotation_index = detect_rotation_point(y_filtered_leftfoot)
    rotation_timestamp = df_Limu2.index[rotation_index]

    # Split Pelvis data into two phases
    phase1_data_pelvis = df_Limu1[df_Limu1.index < rotation_timestamp]
    phase2_data_pelvis = df_Limu1[df_Limu1.index >= rotation_timestamp]

    # Detect movements in each phase
    phase1_peaks = detect_movements(phase1_data_pelvis['Y(number)'], prominence=0.05)
    phase2_valleys = detect_movements(phase2_data_pelvis['Y(number)'], prominence=0.1, is_negative=True)

    # Calculate metrics for each phase
    metrics_phase1 = calculate_metrics(phase1_data_pelvis.index, phase1_data_pelvis['Y(number)'], phase1_peaks, [])
    metrics_phase2 = calculate_metrics(phase2_data_pelvis.index, phase2_data_pelvis['Y(number)'], [], phase2_valleys)

    # Aggregate the metrics
    total_movements = metrics_phase1['number_of_movements'] + metrics_phase2['number_of_movements']
    mean_duration = (metrics_phase1['mean_duration_seconds'] + metrics_phase2['mean_duration_seconds']) / 2
    std_duration = (metrics_phase1['std_duration_seconds'] + metrics_phase2['std_duration_seconds']) / 2
    mean_combined_range_degrees = (metrics_phase1['mean_combined_range_degrees'] + metrics_phase2['mean_combined_range_degrees']) / 2
    std_combined_range_degrees = (metrics_phase1['std_combined_range_degrees'] + metrics_phase2['std_combined_range_degrees']) / 2
    exercise_duration = (df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds()
    pace_movements_per_second = total_movements / exercise_duration if exercise_duration > 0 else 0
    symmetry = min(len(phase1_peaks), len(phase2_valleys)) / max(len(phase1_peaks), len(phase2_valleys)) if max(len(phase1_peaks), len(phase2_valleys)) > 0 else 0

    # Compile metrics into dictionary
    metrics_data = {
        "total_metrics": {
                "number_of_movements": total_movements,
                "pace_movements_per_second": pace_movements_per_second,
                "mean_combined_range_degrees": mean_combined_range_degrees,
                "std_combined_range_degrees": std_combined_range_degrees,
                "mean_duration_seconds": mean_duration,
                "std_duration_seconds": std_duration,
                "Exercise_duration": exercise_duration,
                "symmetry": symmetry
            }
        }
    print(metrics_data)
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_ForwardReach_metrics.txt"

    # Save the metrics to a file
    #save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)

    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Standing Metrics Data"
    sub_directory = "ForwardReach Metrics Data"

    directory = os.path.join(main_directory, sub_directory)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
   
    full_path = os.path.join(directory, file_path)

   
    with open(full_path, 'w') as file:
        for main_key, main_value in metrics.items():
            file.write(f"{main_key}:\n")
            for key, value in main_value.items():
                file.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    file.write(f"    {sub_key}: {sub_value}\n")
                file.write("\n") 
         
