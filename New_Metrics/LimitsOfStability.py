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
from scipy.signal import savgol_filter
from collections import deque
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(w, x, y, z):
    """
    Converts quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    Returns angles in degrees.
    """
    rotation = R.from_quat([x, y, z, w])  # Quaternion order: [x, y, z, w]
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)  # 'xyz' corresponds to roll, pitch, yaw
    return roll, pitch, yaw

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
#     """
#     Interpolate IMU data (w, x, y, z) between starttime and endtime into N samples.

#     Parameters:
#     imu_data (list of lists): The IMU data in format [time, w, x, y, z, _, _].
#     starttime (float): The start time for interpolation.
#     endtime (float): The end time for interpolation.
#     N (int): Number of samples to interpolate.

#     Returns:
#     list of lists: Interpolated IMU data with N entries.
#     """
    

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

    
    Limu2 = processed_dataframes[1]

    if(len(Limu2) > 0):
        returnedJson = getMetricsStandingNew02(Limu2, False) 
        return returnedJson

def getMetricsStandingNew02(Limu2, plotdiagrams):
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['elapsed(time)'] = pd.to_datetime(df_Limu2['elapsed(time)'], unit='ms')
    df_Limu2 = df_Limu2.sort_values(by='elapsed(time)')
    df_Limu2.set_index('elapsed(time)', inplace=True)

    # Extract quaternions and calculate Euler angles
    df_Limu2[['Roll', 'Pitch', 'Yaw']] = df_Limu2.apply(
        lambda row: quaternion_to_euler(row['W(number)'], row['X(number)'], row['Y (number)'], row['Z (number)']),
        axis=1, result_type='expand'
    )

    timestamps = df_Limu2['Timestamp']

    y_signal = df_Limu2['Y (number)']

    # Smooth the y-axis signal
    smoothed_y = savgol_filter(y_signal, window_length=15, polyorder=2)

    # Detect peaks (maxima) and troughs (minima)
    peaks, _ = find_peaks(smoothed_y, prominence=0.02)
    troughs, _ = find_peaks(-smoothed_y, prominence=0.02)

    movements = []
    for peak_idx in peaks:
        movements.append(('peak', peak_idx))
    for trough_idx in troughs:
        movements.append(('trough', trough_idx))

    # Sort movements by their time index
    movements = sorted(movements, key=lambda x: x[1])

    # Process each pair of consecutive movements
    significant_movements = []
    durations = []
    amplitudes = []
    
    exercise_duration = (df_Limu2.index[-1] - df_Limu2.index[0]).total_seconds()

    for i in range(1, len(movements)):
        start_type, start_idx = movements[i - 1]
        end_type, end_idx = movements[i]

        # Separate movements based on type
        if (start_type == 'trough' and end_type == 'peak') or (start_type == 'peak' and end_type == 'trough'):
            time_diff = (df_Limu2.index[end_idx] - df_Limu2.index[start_idx]).total_seconds()
            amplitude_diff = abs(smoothed_y[end_idx] - smoothed_y[start_idx])
            significant_movements.append({
                'Movement': f'Movement {len(significant_movements) + 1}',
                'Time Difference (s)': time_diff,
                'Amplitude Difference': amplitude_diff,
            })
            durations.append(time_diff)
            amplitudes.append(amplitude_diff)

    # Plot detected movements if requested
    if plotdiagrams:
        plt.figure(figsize=(12, 6))
        plt.plot(df_Limu2.index, smoothed_y, label='Smoothed Y Signal', linestyle='--')
        plt.plot(df_Limu2.index[peaks], smoothed_y[peaks], "ro", label="Detected Peaks")
        plt.plot(df_Limu2.index[troughs], smoothed_y[troughs], "go", label="Detected Troughs")
        plt.title("Smoothed Signal with Detected Movements (Peaks and Troughs)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    # Calculate additional metrics
    num_movements = len(significant_movements)
    pace_movements_per_second = num_movements / exercise_duration if exercise_duration > 0 else 0
    mean_combined_range_degrees = np.mean(amplitudes) if amplitudes else 0
    std_combined_range_degrees = np.std(amplitudes) if amplitudes else 0
    mean_duration_seconds = np.mean(durations) if durations else 0
    std_duration_seconds = np.std(durations) if durations else 0

    # Output metrics
    metrics_data = {
        "total_metrics": {
            "number_of_movements": num_movements,
            "pace_movements_per_second": pace_movements_per_second,
            "mean_combined_range_degrees": mean_combined_range_degrees,
            "std_combined_range_degrees": std_combined_range_degrees,
            "mean_duration_seconds": mean_duration_seconds,
            "std_duration_seconds": std_duration_seconds,
            "exercise_duration_seconds": exercise_duration
        }
    }

    print(metrics_data)
    # for movement in significant_movements:
    #     print(f"{movement['Movement']}: Time Difference = {movement['Time Difference (s)']} s, "
    #           f"Amplitude Difference = {movement['Amplitude Difference']}")

    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_AnteroposteriorDirection_metrics.txt"
    save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)

def save_metrics_to_txt(metrics, file_path):
    main_directory = "Standing Metrics Data"
    sub_directory = "AnteroposteriorDirection Metrics Data"

    directory = os.path.join(main_directory, sub_directory)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
   
    full_path = os.path.join(directory, file_path)

   
    with open(full_path, 'w') as file:
        for key, value in metrics.items():
            if isinstance(value, dict):  
                file.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    file.write(f"  {sub_key}: {sub_value}\n")
            else:
                file.write(f"{key}: {value}\n")
            file.write("\n")