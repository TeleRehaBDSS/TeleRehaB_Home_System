
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
from scipy.signal import butter, filtfilt, correlate
import json
import pywt

def preprocess_signal_enhanced(signal, smoothing_window=5, spike_threshold=0.02):
    smoothed_signal = signal.rolling(window=smoothing_window, center=True).mean()
    smoothed_signal = pywt.threshold(smoothed_signal.fillna(method='bfill').fillna(method='ffill'), spike_threshold, mode='soft')
    return smoothed_signal

def calculate_metrics(peaks, signal, sampling_rate):
    num_movements = len(peaks)
    intervals = np.diff(peaks) / sampling_rate
    pace_movements_per_second = num_movements / (peaks[-1] / sampling_rate) if num_movements > 1 else 0
    ranges = [abs(signal.iloc[p] - signal.iloc[p-1]) if p > 0 else 0 for p in peaks]
    mean_combined_range_degrees = np.mean(ranges)
    std_combined_range_degrees = np.std(ranges)
    mean_duration_seconds = np.mean(intervals) if len(intervals) > 0 else 0
    std_duration_seconds = np.std(intervals) if len(intervals) > 0 else 0
    exercise_duration = peaks[-1] / sampling_rate if num_movements > 1 else 0

    metrics = {
        "number_of_movements": num_movements,
        "pace_movements_per_second": pace_movements_per_second,
        "mean_combined_range_degrees": mean_combined_range_degrees,
        "std_combined_range_degrees": std_combined_range_degrees,
        "mean_duration_seconds": mean_duration_seconds,
        "std_duration_seconds": std_duration_seconds,
        "exercise_duration_seconds": exercise_duration,
        "movement_duration": intervals.tolist()  
    }
    return metrics



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
            print("DF")
            print(df)
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

def estimate_period(signal, sampling_rate):
    """
    Estimate the dominant period of a signal (in samples) using its autocorrelation.
    Returns the lag (in samples) of the first peak after lag 0.
    """
    # Remove the mean to avoid a huge peak at lag 0
    signal = signal - np.mean(signal)
    # Compute full autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    # Use only the second half (non-negative lags)
    autocorr = autocorr[autocorr.size // 2:]
    # Zero out the lag-0 value so that we can find the next peak
    autocorr[0] = 0
    # Find peaks in the autocorrelation function
    peaks, _ = find_peaks(autocorr)
    if len(peaks) == 0:
        return None
    # The first detected peak corresponds to the dominant period (in samples)
    period = peaks[1]
    return period*0.5

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
            A.append([t[-6],t[-4],t[-3],t[-2],t[-1]])
    return A

def get_metrics(imu1,imu2,imu3,imu4, counter):

    Limu1 = reformat_sensor_data(imu1)
    Limu2 = reformat_sensor_data(imu2)
    Limu3 = reformat_sensor_data(imu3)   
    Limu4 = reformat_sensor_data(imu4)
    imu_data_lists = [Limu1, Limu2, Limu3, Limu4]
    
    processed_dataframes, c = process_imu_data(imu_data_lists, 50, True)
    Limu1 = processed_dataframes[0]
    #print('Limu1 = ', Limu1)
    if (c >= 2):
        Limu2 = processed_dataframes[1]
    if (c >= 3):
        Limu3 = processed_dataframes[2]
    if (c >= 4):
        Limu4 = processed_dataframes[3]

    dt1 = 0 
    dt2 = 0
    dt3 = 0
    dt4 = 0
    
    if(len(Limu1) > 0 ):
        dt1 = float(Limu1[-1][1]) - float(Limu1[0][1]);
    if(len(Limu2) > 0 ):
        dt2 = float(Limu2[-1][1]) - float(Limu2[0][1]);
    if(len(Limu3) > 0 ):
        dt3 = float(Limu3[-1][1]) - float(Limu3[0][1]);
    if(len(Limu4) > 0 ):
        dt4 = float(Limu4[-1][1]) - float(Limu4[0][1]);

    mean = statistics.mean([dt1, dt2, dt3, dt4])
    std = statistics.stdev([dt1, dt2, dt3, dt4])

    Limu1 = [[float(item) for item in sublist] for sublist in Limu1]

    Limu2 = [[float(item) for item in sublist] for sublist in Limu2]
    Limu3 = [[float(item) for item in sublist] for sublist in Limu3]
    Limu4 = [[float(item) for item in sublist] for sublist in Limu4]
    
    if len(Limu3) > 0 and len(Limu4) > 0:
        print('procceding to metrics...')
        returnedJson = getMetricsSittingNew02(Limu3, Limu4,False) 
        return returnedJson

def getMetricsSittingNew02(Limu3, Limu4, plotdiagrams):
    # Create DataFrames from the provided data
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    df_Limu3 = df_Limu3.reset_index(drop=True)
    df_Limu3['elapsed(time)'] = pd.to_datetime(df_Limu3['elapsed(time)'], unit='ms')
    df_Limu3 = df_Limu3.sort_values(by=['elapsed(time)'])
    df_Limu3.set_index('elapsed(time)', inplace=True)
        
    df_Limu4 = pd.DataFrame(Limu4, columns=columns)
    df_Limu4 = df_Limu4.reset_index(drop=True)
    df_Limu4['elapsed(time)'] = pd.to_datetime(df_Limu4['elapsed(time)'], unit='ms')
    df_Limu4 = df_Limu4.sort_values(by=['elapsed(time)'])
    df_Limu4.set_index('elapsed(time)', inplace=True)
        
    # Preprocess the    X-axis data
    df_Limu3['z_smoothed'] = preprocess_signal_enhanced(df_Limu3['Z (number)'])
    df_Limu4['z_smoothed'] = preprocess_signal_enhanced(df_Limu4['Z (number)'])
    
    # Detection parameters
    sampling_rate = 50  # Hz
    default_distance = 100  # Fallback value if autocorrelation fails

    # Use autocorrelation to estimate the period (in samples) of the toe raises
    left_period = estimate_period(df_Limu3['z_smoothed'].values, sampling_rate)
    right_period = estimate_period(df_Limu4['z_smoothed'].values, sampling_rate)
    distance_left = int(left_period) if left_period is not None else default_distance
    distance_right = int(right_period) if right_period is not None else default_distance

    print(f"Estimated left period (samples): {distance_left}")
    print(f"Estimated right period (samples): {distance_right}")

    # Detect movements using the autocorrelation-based distance constraints
    left_valleys, _ = find_peaks(df_Limu3['z_smoothed'],prominence=0.05, distance=distance_left)
    right_peaks, _ = find_peaks(df_Limu4['z_smoothed'],prominence=0.05, distance=distance_right)

    # Calculate metrics for each leg
    left_metrics = calculate_metrics(left_valleys, df_Limu3['z_smoothed'], sampling_rate)
    right_metrics = calculate_metrics(right_peaks, df_Limu4['z_smoothed'], sampling_rate)

    # Organize metrics data
    metrics_data = {
        "total_metrics": {
            "RIGHT LEG": right_metrics,
            "LEFT LEG": left_metrics
        }
    }

    print(metrics_data)

    # Optionally plot diagrams
    if plotdiagrams:
        plt.figure(figsize=(12, 6))
        plt.plot(df_Limu3['z_smoothed'].values, label='Left Foot Z-Axis (Smoothed)')
        plt.plot(left_valleys, df_Limu3['z_smoothed'].values[left_valleys], "rx", label="Detected Movements - Left Foot")
        plt.legend()
        plt.title("Detected Movements in Left Foot (Z-Axis)")
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(df_Limu4['z_smoothed'].values, label='Right Foot Z-Axis (Smoothed)')
        plt.plot(right_peaks, df_Limu4['z_smoothed'].values[right_peaks], "rx", label="Detected Movements - Right Foot")
        plt.legend()
        plt.title("Detected Movements in Right Foot (Z-Axis)")
        plt.show()
    
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_ToeRaises_metrics.txt"

    # Save the metrics to a file
    save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)

    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Sitting Metrics Data"
    sub_directory = "ToeRaises Metrics Data"

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