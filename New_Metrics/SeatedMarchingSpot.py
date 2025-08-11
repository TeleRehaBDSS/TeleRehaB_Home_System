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
from scipy.ndimage import gaussian_filter1d
import pywt

def estimate_peak_distance(signal, sampling_rate):
    """
    Estimate the minimum distance between peaks using autocorrelation.
    """
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Take the positive lags
    peaks, _ = find_peaks(autocorr)
    if len(peaks) > 1:
        peak_distances = np.diff(peaks)
        avg_distance = np.mean(peak_distances)
        return int(avg_distance)
    else:
        return int(sampling_rate / 2)  # Default to half the sampling rate

def estimate_prominence(signal, factor=0.3):
    """
    Estimate the prominence based on the standard deviation of the signal.
    """
    return factor * np.std(signal)

def estimate_cutoff_frequency(signal, sampling_rate):
    """
    Estimate the cutoff frequency for lowpass filtering based on the power spectral density.
    """
    freqs, psd = plt.psd(signal, Fs=sampling_rate)
    dominant_freq = freqs[np.argmax(psd)]
    return dominant_freq * 1.5  # Slightly above the dominant frequency



# Filter peaks by ensuring significant distance between them
def filter_peaks(peaks, previous_peaks, min_distance=150):
    all_peaks = np.unique(np.concatenate((previous_peaks,peaks)))
    filtered_peaks = []
    last_peak = -min_distance  # Initial large negative to ensure first peak is included
    for peak in peaks:
        if peak - last_peak >= min_distance:
            filtered_peaks.append(peak)
            last_peak = peak
    return np.array(filtered_peaks, dtype=int)  # Ensure integer type for indexing
    
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

    imu_data_lists1 = [Limu1, Limu2]
    imu_data_lists2 = [Limu3, Limu4]

    processed_dataframes1, c1 = process_imu_data(imu_data_lists1, 50, True)
    processed_dataframes2, c2 = process_imu_data(imu_data_lists2, 100, True)


    Limu1 = processed_dataframes1[0]
    Limu2 = processed_dataframes1[1]
    Limu3 = processed_dataframes2[0]
    Limu4 = processed_dataframes2[1]

    if(len(Limu2) > 0 and len(Limu3) >0 and len(Limu4) > 0 ):
        print('procceding to metrics...')
        returnedJson = getMetricsSittingNew04(Limu2, Limu3, Limu4 ,False) 
        return returnedJson


def getMetricsSittingNew04(Limu0, Limu1, Limu2, plotdiagrams):
   
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['Timestamp'] = pd.to_datetime(df_Limu1['Timestamp'])
    df_Limu1 = df_Limu1.sort_values(by='Timestamp')
    df_Limu1.set_index('Timestamp', inplace=True)

    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['Timestamp'] = pd.to_datetime(df_Limu2['Timestamp'])
    df_Limu2 = df_Limu2.sort_values(by='Timestamp')
    df_Limu2.set_index('Timestamp', inplace=True)

    # Sampling rate and time interval
    sampling_rate = 100  # 100 Hz
    time_interval = 1 / sampling_rate  # seconds

    # Initialize previous peaks
    previous_left_peaks = np.array([], dtype=int)
    previous_right_peaks = np.array([], dtype=int)

    # Extract and smooth the z-axis signal
    left_foot_z = gaussian_filter1d(df_Limu1['Z (number)'].values, sigma=2)
    right_foot_z = gaussian_filter1d(df_Limu2['Z (number)'].values, sigma=2)

    # Convert timestamps to elapsed time in seconds
    timestamps_left = (df_Limu1.index - df_Limu1.index[0]).total_seconds().values
    timestamps_right = (df_Limu2.index - df_Limu2.index[0]).total_seconds().values

    # Lists to store cumulative values
    left_movement_counts = []
    right_movement_counts = []
    left_durations_list = []
    right_durations_list = []
    left_ranges_list = []
    right_ranges_list = []

    # Dynamic parameter estimation
    left_distance = estimate_peak_distance(left_foot_z, sampling_rate)
    right_distance = estimate_peak_distance(right_foot_z, sampling_rate)

    left_prominence = estimate_prominence(left_foot_z)
    right_prominence = estimate_prominence(right_foot_z)

    # Peak detection
    left_peaks, _ = find_peaks(-left_foot_z, prominence=left_prominence, distance=left_distance)
    right_peaks, _ = find_peaks(right_foot_z, prominence=right_prominence, distance=right_distance)

    # Append detected movements to lists
    left_movement_counts.append(len(left_peaks))
    right_movement_counts.append(len(right_peaks))

    # Compute durations and append them
    if len(left_peaks) > 1:
        left_durations_list.extend(np.diff(timestamps_left[left_peaks]))
    if len(right_peaks) > 1:
        right_durations_list.extend(np.diff(timestamps_right[right_peaks]))

    # Compute ranges and append them
    if len(left_peaks) > 0:
        left_ranges_list.append(np.ptp(left_foot_z[left_peaks]))
    if len(right_peaks) > 0:
        right_ranges_list.append(np.ptp(right_foot_z[right_peaks]))

    # Compute cumulative values
    total_left_movements = sum(left_movement_counts)
    total_right_movements = sum(right_movement_counts)
    total_left_duration = sum(left_durations_list) if len(left_durations_list) > 0 else 0
    total_right_duration = sum(right_durations_list) if len(right_durations_list) > 0 else 0
    total_left_range = sum(left_ranges_list) / len(left_ranges_list) if len(left_ranges_list) > 0 else 0
    total_right_range = sum(right_ranges_list) / len(right_ranges_list) if len(right_ranges_list) > 0 else 0

    # Compile final metrics
    metrics_data = {
        "total_metrics": {
            "LEFT LEG": {
                "number_of_movements": total_left_movements,
                "pace_movements_per_second": total_left_movements / (len(left_foot_z) * time_interval),
                "mean_combined_range_degrees": total_left_range,
                "mean_duration_seconds": total_left_duration / total_left_movements if total_left_movements > 0 else 0,
                "exercise_duration_seconds": len(left_foot_z) * time_interval,
                "movement_duration": total_left_duration,
            },
            "RIGHT LEG": {
                "number_of_movements": total_right_movements,
                "pace_movements_per_second": total_right_movements / (len(right_foot_z) * time_interval),
                "mean_combined_range_degrees": total_right_range,
                "mean_duration_seconds": total_right_duration / total_right_movements if total_right_movements > 0 else 0,
                "exercise_duration_seconds": len(right_foot_z) * time_interval,
                "movement_duration": total_right_duration,
            }
        }
    }

    print(metrics_data)


    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_SeatedMarchingSpot_metrics.txt"

    # Save the metrics to a file
    save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)
 
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Sitting Metrics Data"
    sub_directory = "SeatedMarchingSpot Metrics Data"

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
         
