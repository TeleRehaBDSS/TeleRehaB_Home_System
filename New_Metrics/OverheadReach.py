import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json

def moving_average_filter(signal, window_size=25):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Keep only the smallest minimum between each pair of maxima
def filter_minima(signal, maxima, minima):
    filtered_minima = []
    if isinstance(signal, np.ndarray):
        signal = pd.Series(signal)
    for i in range(len(maxima) - 1):
        # Get minima between two maxima
        start, end = maxima[i], maxima[i + 1]
        interval_minima = [m for m in minima if start < m < end]
        if interval_minima:
            # Find the smallest minimum in the interval
            smallest_minimum = min(interval_minima, key=lambda m: signal.iloc[m])
            filtered_minima.append(smallest_minimum)
    return filtered_minima

# Filter maxima based on one standard deviation
def filter_maxima(signal, peaks):
    if isinstance(signal, np.ndarray):
        signal = pd.Series(signal)
    max_val = signal.max()
    std_dev = 1.5 * signal.std()
    return peaks[signal.iloc[peaks] >= (max_val - std_dev)]

# Calculate autocorrelation to determine the distance parameter
def calculate_autocorrelation(data):
    autocorr = correlate(data, data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Use only positive lags
    lags = range(len(autocorr))
    return autocorr, lags

def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Normalize the 'w' components to 0-1 range
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

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
    #print('Limu1 = ', Limu1)
    if (c >= 2):
        Limu2 = processed_dataframes[1]
    if (c >= 3):
        Limu3 = processed_dataframes[2]
    if (c >= 4):
        Limu4 = processed_dataframes[3]

    if len(Limu1) > 0 and len(Limu2) > 0:
        returnedJson = getMetricsStandingOld03(Limu1, Limu2, False) 
        return returnedJson

def getMetricsStandingOld03(Limu1, Limu2, plotdiagrams):
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['elapsed(time)'] = pd.to_datetime(df_Limu1['elapsed(time)'], unit='ms')
    df_Limu1 = df_Limu1.sort_values(by='elapsed(time)')
    df_Limu1.set_index('elapsed(time)', inplace=True)
    
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['elapsed(time)'] = pd.to_datetime(df_Limu2['elapsed(time)'], unit='ms')
    df_Limu2 = df_Limu2.sort_values(by='elapsed(time)')
    df_Limu2.set_index('elapsed(time)', inplace=True)

    # Convert to Euler angles
    quat1 = df_Limu1[['X(number)', 'Y (number)', 'Z (number)', 'W(number)']].to_numpy()
    quat2 = df_Limu2[['X(number)', 'Y (number)', 'Z (number)', 'W(number)']].to_numpy()
    
    euler1 = R.from_quat(quat1).as_euler('xyz', degrees=True)
    euler2 = R.from_quat(quat2).as_euler('xyz', degrees=True)

    df_Limu1['Pitch (deg)'] = euler1[:, 1]
    df_Limu2['Pitch (deg)'] = euler2[:, 1]

    # Smooth signals
    fs = 50
    interval = 1 / fs
    time_interval = interval  # used for duration

    head_pitch = normalize(df_Limu1['Pitch (deg)'])
    pelvis_pitch = normalize(df_Limu2['Pitch (deg)'])

    head_pitch = moving_average_filter(head_pitch, window_size=50)
    pelvis_pitch = moving_average_filter(pelvis_pitch, window_size=50)

    # Autocorrelation-based distance estimate
    def calc_distance(signal):
        autocorr = correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        return next(i for i in range(1, len(autocorr)) if autocorr[i] < autocorr[0] * 0.90)

    distance = calc_distance(df_Limu1['W(number)'])

    # Detect head movements
    head_maxima, _ = find_peaks(df_Limu1['W(number)'],prominence=0.07)
    head_minima, _ = find_peaks(-df_Limu1['W(number)'],prominence=0.07)

    # Filter minima to pair with maxima
    head_minima_filtered = head_minima

    durations = []
    ranges_head = []
    ranges_pelvis = []

    for i in range(min(len(head_maxima), len(head_minima_filtered))):
        start = min(head_maxima[i], head_minima_filtered[i])
        end = max(head_maxima[i], head_minima_filtered[i])
        durations.append((end - start) * time_interval)
        ranges_head.append(abs(df_Limu1['Pitch (deg)'].iloc[end] - df_Limu1['Pitch (deg)'].iloc[start]))
        ranges_pelvis.append(abs(df_Limu2['Pitch (deg)'].iloc[end] - df_Limu2['Pitch (deg)'].iloc[start]))

    total_duration = (df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds()
    pace = len(durations) / total_duration if total_duration > 0 else 0

    metrics_data = {
        "total_metrics": {
            "number_of_movements": len(durations),
            "pace_movements_per_second": pace,
            "mean_duration_seconds": np.mean(durations) if durations else 0,
            "std_duration_seconds": np.std(durations) if durations else 0,
            "mean_range_degrees_head": np.mean(ranges_head) if ranges_head else 0,
            "std_range_degrees_head": np.std(ranges_head) if ranges_head else 0,
            "mean_range_degrees_pelvis": np.mean(ranges_pelvis) if ranges_pelvis else 0,
            "std_range_degrees_pelvis": np.std(ranges_pelvis) if ranges_pelvis else 0,
            "exercise_duration": total_duration
        }
    }

    print(metrics_data)

    # Save to file
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_OverheadReach_metrics.txt"
    save_metrics_to_txt(metrics_data, filename)

    print(json.dumps(metrics_data, indent=4))
    return json.dumps(metrics_data, indent=4)

    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Standing Metrics Data"
    sub_directory = "StandingBendingOver Metrics Data"

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