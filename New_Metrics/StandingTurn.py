import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt, correlate
import json


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
            # plt.show()

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

    if(len(Limu2) > 0 and len(Limu3)>0 and len(Limu4)>0 ):
        returnedJson = getMetricsStandingOld04(Limu2,Limu3,Limu4, False) 
        return returnedJson

def getMetricsStandingOld04(Limu2, Limu3, Limu4, plotdiagrams):
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    df_Limu3['elapsed(time)'] = pd.to_datetime(df_Limu3['elapsed(time)'], unit='ms')
    df_Limu3 = df_Limu3.sort_values(by='elapsed(time)')
    df_Limu3.set_index('elapsed(time)', inplace=True)
    
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['elapsed(time)'] = pd.to_datetime(df_Limu2['elapsed(time)'], unit='ms')
    df_Limu2 = df_Limu2.sort_values(by='elapsed(time)')
    df_Limu2.set_index('elapsed(time)', inplace=True)

    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu4 = pd.DataFrame(Limu4, columns=columns)
    df_Limu4['elapsed(time)'] = pd.to_datetime(df_Limu4['elapsed(time)'], unit='ms')
    df_Limu4 = df_Limu4.sort_values(by='elapsed(time)')
    df_Limu4.set_index('elapsed(time)', inplace=True)


    # Extract W components
    w_pelvis = df_Limu2['W(number)']
    w_ankle_left = df_Limu3['W(number)']
    w_ankle_right = df_Limu4['W(number)']

    # Movement detection and metric calculation
    metrics_data = {}
    autocorr = correlate(w_pelvis - np.mean(w_pelvis), w_pelvis - np.mean(w_pelvis), mode='full')
    autocorr = autocorr[autocorr.size // 2:]

    # Estimate periodic movement distance using autocorrelation
    peak_lags, _ = find_peaks(autocorr)
    estimated_distance = int(np.mean(np.diff(peak_lags)) / 2) if len(peak_lags) > 1 else None

    if estimated_distance:
        peaks, _ = find_peaks(w_pelvis, distance=estimated_distance)
        peak_values = w_pelvis[peaks]
        mean_peak_value = np.mean(peak_values)
        std_peak_value = np.std(peak_values)
        filtered_peaks = peaks[(peak_values >= mean_peak_value - std_peak_value) & 
                               (peak_values <= mean_peak_value + std_peak_value)]
        
        minima = []
        for i in range(len(filtered_peaks) - 1):
            start, end = filtered_peaks[i], filtered_peaks[i + 1]
            local_min = start + np.argmin(w_pelvis[start:end])
            minima.append(local_min)

        movements = []
        max_to_min_count = 0
        min_to_max_count = 0
        for i in range(1, len(filtered_peaks)):
            if i - 1 < len(minima) and minima[i - 1] > filtered_peaks[i - 1] and minima[i - 1] < filtered_peaks[i]:
                movements.append((filtered_peaks[i - 1], minima[i - 1], 'max_to_min'))
                max_to_min_count += 1
            if i < len(minima) and minima[i - 1] < filtered_peaks[i]:
                movements.append((minima[i - 1], filtered_peaks[i], 'min_to_max'))
                min_to_max_count += 1

        # Calculate movement metrics
        durations = [abs(end - start) for start, end, _ in movements]
        num_movements = len(movements)
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        left_right_durations = [abs(end - start) for start, end, direction in movements if direction == 'max_to_min']
        right_left_durations = [abs(end - start) for start, end, direction in movements if direction == 'min_to_max']
        
        mean_duration_left_right = np.mean(left_right_durations) if left_right_durations else None
        mean_duration_right_left = np.mean(right_left_durations) if right_left_durations else None
        symmetry = mean_duration_left_right / mean_duration_right_left if mean_duration_left_right and mean_duration_right_left else None
        total_duration_seconds = (df_Limu2.index[-1] - df_Limu2.index[0]).total_seconds()
        # Store metrics

        metrics_data = {
            "total_metrics":{
            "Total Movements": int(num_movements),
            "Right-to-Left (Max-to-Min) Movements": max_to_min_count,
            "Left-to-Right (Min-to-Max) Movements": min_to_max_count,
            "Mean Duration": float(mean_duration),
            "Duration Standard Deviation": float(std_duration),
            "Mean Duration Left-to-Right (Max-to-Min)": mean_duration_left_right,
            "Mean Duration Right-to-Left (Min-to-Max)": mean_duration_right_left,
            "Symmetry Ratio (Left-to-Right / Right-to-Left)": symmetry,
            "exercise_duration_seconds":total_duration_seconds
        }
        }
    else:
        metrics_data = {
            "total_metrics":{
            "Total Movements": 0,
            }}
        print(metrics_data);
        if plotdiagrams:
            plt.plot(w_pelvis, label="Pelvis W Component")
            plt.plot(filtered_peaks, w_pelvis[filtered_peaks], "go", label="Filtered Peaks")
            plt.plot(minima, w_pelvis[minima], "bo", label="Detected Minima")
            plt.title("Real-Time Pelvis Movements")
            plt.xlabel("Time")
            plt.ylabel("W Component")
            plt.legend()
            plt.show()

    # Saving the metrics to file
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_StandingTurn_metrics.txt"
    save_metrics_to_txt(metrics_data, filename)
    
    return json.dumps(metrics_data, indent=4)

    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Standing Metrics Data"
    sub_directory = "StandingTurn Metrics Data"

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