import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json
import matplotlib.pyplot as plt

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

    # Resample and interpolate dataframes to have the same number of samples
    resampled_dataframes = []
    for df in dataframes:
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

def get_metrics(imu1, imu2, imu3, imu4, counter):
    
    Limu1 = reformat_sensor_data(imu1)
    Limu2 = reformat_sensor_data(imu2)
    Limu3 = reformat_sensor_data(imu3)   
    Limu4 = reformat_sensor_data(imu4)

    imu_data_lists = [Limu1, Limu2, Limu3, Limu4]

    processed_dataframes, c = process_imu_data(imu_data_lists, 50, True)

    
    Limu2 = processed_dataframes[1]

    if len(Limu2) > 0:
        returnedJson = getMetricsStandingNew01(Limu2, False) 
        return returnedJson

def getMetricsStandingNew01(Limu2, plotdiagrams):
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['elapsed(time)'] = pd.to_datetime(df_Limu2['elapsed(time)'], unit='ms')
    df_Limu2 = df_Limu2.sort_values(by='elapsed(time)')
    df_Limu2.set_index('elapsed(time)', inplace=True)
    

    quaternions2 = df_Limu2[['X(number)', 'Y (number)', 'Z (number)', 'W(number)']].to_numpy()
    rotations2 = R.from_quat(quaternions2)
    euler_angles2 = rotations2.as_euler('xyz', degrees=False)
    euler_df2 = pd.DataFrame(euler_angles2, columns=['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)'])
    euler_angles_degrees2 = rotations2.as_euler('xyz', degrees=True)
    euler_df_degrees2 = pd.DataFrame(euler_angles_degrees2, columns=['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)'])

    start_time = df_Limu2.index.min()
    end_time = df_Limu2.index.max()
    interval_length = pd.Timedelta(seconds=5)
    

    quaternions_df2 = df_Limu2

    fs = 50
    cutoff = 0.5

    W_filtered = butter_lowpass_filter(quaternions_df2['W(number)'], cutoff, fs, order=5)
    Y_filtered = butter_lowpass_filter(quaternions_df2['Y (number)'], cutoff, fs, order=5)

    movement_magnitude = np.sqrt(np.square(W_filtered) + np.square(Y_filtered))

    yaw_filtered2 = butter_lowpass_filter(euler_df_degrees2['Yaw (degrees)'], cutoff, fs, order=5)

    peaks, _ = find_peaks(movement_magnitude)
    valleys, _ = find_peaks(-movement_magnitude)

    #print("peaks ", peaks)
    #print("valleys ", valleys)
    if len(peaks) == 0:
        return 0
    if len(valleys) == 0:
        return 0

    if valleys[0] > peaks[0]:
        peaks = peaks[1:]  
    if peaks[-1] < valleys[-1]:
        valleys = valleys[:-1]  
    
    movement_pairs = []
    for i in range(min(len(peaks), len(valleys))):
        movement_pairs.append((valleys[i], peaks[i]))

    #print("Movement pairs (as index positions):", movement_pairs)

    movement_ranges_yaw = []
    movement_ranges_roll = []

    for valley, peak in movement_pairs:
        yaw_range = abs(W_filtered[peak] - W_filtered[valley])
        movement_ranges_yaw.append(yaw_range)
        
        roll_range = abs(Y_filtered[peak] - Y_filtered[valley])
        movement_ranges_roll.append(roll_range)

    combined_movement_ranges = [np.sqrt(yaw**2 + roll**2) for yaw, roll in zip(movement_ranges_yaw, movement_ranges_roll)]

    for i, (yaw_range, roll_range) in enumerate(zip(movement_ranges_yaw, movement_ranges_roll)):
        combined_range = np.sqrt(yaw_range**2 + roll_range**2)
        print(f"Movement {i+1}: Yaw Range = {yaw_range:.2f} degrees, Roll Range = {roll_range:.2f} degrees, Combined Range = {combined_range:.2f} degrees")

    significant_movements = [(pair, yaw, roll, np.sqrt(yaw**2 + roll**2)) for pair, yaw, roll in zip(movement_pairs, movement_ranges_yaw, movement_ranges_roll) if np.sqrt(yaw**2 + roll**2) >= 0.01]

    filtered_pairs = [item[0] for item in significant_movements]
    filtered_combined_ranges = [item[3] for item in significant_movements]

    #for i, (_, _, _, combined_range) in enumerate(significant_movements):
        #print(f"Significant Movement {i+1}: Combined Range = {combined_range:.2f} degrees")

    movement_durations = []

    for start, end in filtered_pairs:
        start_time = df_Limu2.iloc[start].name
        end_time = df_Limu2.iloc[end].name
        duration = (end_time - start_time).total_seconds()
        movement_durations.append(duration)

    total_duration_seconds = (df_Limu2.index[-1] - df_Limu2.index[0]).total_seconds()
    pace = len(filtered_pairs) / total_duration_seconds  # Movements per second

    mean_combined_range = np.mean(filtered_combined_ranges)
    std_combined_range = np.std(filtered_combined_ranges, ddof=1)  # ddof=1 for sample standard deviation

    mean_duration = np.mean(movement_durations)
    std_duration = np.std(movement_durations, ddof=1)  # ddof=1 for sample standard deviation    

    # sway_range = max(combined_movement_ranges) - min(combined_movement_ranges) if combined_movement_ranges >= 4 else -1

    metrics_data = {
        "total_metrics": {
            "number_of_movements": int(len(filtered_pairs)),
            "pace_movements_per_second": float(pace),
            "mean_range_degrees": float(mean_combined_range),
            "std_range_degrees": float(std_combined_range),
            "mean_duration_seconds": float(mean_duration),
            "std_duration_seconds": float(std_duration),
            "exersice_duration_seconds" : total_duration_seconds
        }
    }
    print(metrics_data)
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_StandingBalanceFoam_metrics.txt"

    save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)

def save_metrics_to_txt(metrics, file_path):
    main_directory = "Standing Metrics Data"
    sub_directory = "StandingBalanceFoam Metrics Data"

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
