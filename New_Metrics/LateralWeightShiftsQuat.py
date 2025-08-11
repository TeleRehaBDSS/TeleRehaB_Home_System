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
from scipy.signal import correlate

def calculate_autocorrelation_distance(data_y, sampling_rate, detect_peaks=True):
    """
    Calculate the dominant autocorrelation-based distance in a signal.
    
    Parameters:
    - data_y: The signal to analyze.
    - sampling_rate: Sampling rate of the signal in Hz.
    - detect_peaks: Boolean indicating whether to calculate distance for peaks (True) or valleys (False).

    Returns:
    - distance: The estimated periodicity in samples.
    """
    # Normalize the signal for autocorrelation
    data_y_normalized = (data_y - np.mean(data_y)) / np.std(data_y)

    # For peaks, use the original signal; for valleys, invert it
    if not detect_peaks:
        data_y_normalized = -data_y_normalized

    # Calculate the autocorrelation of the signal
    autocorr = correlate(data_y_normalized, data_y_normalized, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Keep only the positive lags

    # Find peaks in the autocorrelation to identify periodicity
    peaks, _ = find_peaks(autocorr, height=0)  # Only consider positive peaks
    if len(peaks) > 1:
        # Calculate the first significant lag
        dominant_lag = peaks[1]  # Exclude lag=0
        distance = dominant_lag
    else:
        # Default distance if no periodicity detected
        distance = int(sampling_rate * 0.5)  # Assume half a second interval

    return distance


# Adaptive smoothing function
def adaptive_smooth(data_y, sigma=2):
    return gaussian_filter1d(data_y, sigma=sigma)

# Baseline correction
def baseline_correction(data_y):
    median_baseline = data_y.median()
    return data_y - median_baseline

def detect_major_valleys(data_y, sampling_rate, prominence=0.025, distance = 100):
    """
    Detect major valleys dynamically using autocorrelation for distance estimation.

    Parameters:
    - data_y: The signal to analyze.
    - sampling_rate: Sampling rate of the signal in Hz.
    - prominence: Prominence threshold for detecting valleys.

    Returns:
    - major_valleys: Indices of detected valleys.
    """

    #print(f"Autocorrelation-based distance: {distance} samples")

    # Detect valleys with dynamically determined distance
    valleys, properties = find_peaks(-data_y, distance=distance)
    return valleys  # Sorted indices of valleys


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
        returnedJson = getMetricsStandingNew01(Limu2, Limu3, Limu4 ,False) 
        return returnedJson

# Main function for real-time movement detection
def getMetricsStandingNew01(Limu2, Limu3, Limu4, plotdiagrams):

    left_timestamps = []
    right_timestamps = []
    left_durations = []
    right_durations = []
    
    columns = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    df_Limu3['elapsed(time)'] = pd.to_datetime(df_Limu3['elapsed(time)'], unit='ms')
    df_Limu3 = df_Limu3.sort_values(by='elapsed(time)')
    df_Limu3.set_index('elapsed(time)', inplace=True)
    df_Limu3['w_diff'] = df_Limu3['W(number)'].diff()

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
    df_Limu4['w_diff'] = df_Limu4['W(number)'].diff()
    # Apply Gaussian smoothing for noise reduction
    sigma = 2  # Smoothing factor
    df_Limu3['w_smooth'] = gaussian_filter1d(df_Limu3['w_diff'].fillna(0), sigma=sigma)
    df_Limu4['w_smooth'] = gaussian_filter1d(df_Limu4['w_diff'].fillna(0), sigma=sigma)


    # Preprocess each foot’s data
    # Adaptive peak detection for movement identification
    def detect_refined_peaks(signal):
        """Detects movement peaks with adaptive thresholds."""
        max_amp = np.max(np.abs(signal))
        std_dev = np.std(signal)
        height_threshold = max_amp * 0.2
        prominence_threshold = std_dev * 0.4
        min_distance = int(len(signal) / 12)
        peaks, _ = find_peaks(signal, height=height_threshold, prominence=prominence_threshold, distance=min_distance)
        return peaks
    
    # Detect movements
    left_movements = detect_refined_peaks(df_Limu3['w_smooth'])
    right_movements = detect_refined_peaks(df_Limu4['w_smooth'])

    # Convert indices to timestamps
    left_timestamps = df_Limu3['Timestamp'].iloc[left_movements].tolist()
    right_timestamps = df_Limu4['Timestamp'].iloc[right_movements].tolist()

    # Compute movement durations (right to next left & left to next right)
    left_durations = []
    right_durations = []

    for rt in right_timestamps:
        next_left = next((lt for lt in left_timestamps if lt > rt), None)
        if next_left:
            right_durations.append((next_left - rt).total_seconds())

    for lt in left_timestamps:
        next_right = next((rt for rt in right_timestamps if rt > lt), None)
        if next_right:
            left_durations.append((next_right - lt).total_seconds())

    # Compute updated metrics separately for left and right foot
    num_left_movements = len(left_durations)
    num_right_movements = len(right_durations)

    total_time = (df_Limu2.index[-1] - df_Limu2.index[0]).total_seconds()
    
    pace_left = num_left_movements / total_time if total_time > 0 else 0
    pace_right = num_right_movements / total_time if total_time > 0 else 0

    mean_duration_left = np.mean(left_durations) if left_durations else 0
    std_duration_left = np.std(left_durations) if left_durations else 0

    mean_duration_right = np.mean(right_durations) if right_durations else 0
    std_duration_right = np.std(right_durations) if right_durations else 0
    total_duration_seconds = (df_Limu3.index[-1] - df_Limu3.index[0]).total_seconds()
    # Calculate metrics separately for each foot
    metrics_data = {
        "total_metrics": {
            "LEFT LEG": {
                "number_of_movements": num_left_movements,
                "pace_movements_per_second": pace_left,
                "mean_duration_seconds": mean_duration_left,
                "std_duration_seconds": std_duration_left,
                "exercise_duration_seconds": total_duration_seconds
            },
            "RIGHT LEG": {
                "number_of_movements": num_right_movements,
                "pace_movements_per_second": pace_right,
                "mean_duration_seconds": mean_duration_right,
                "std_duration_seconds": std_duration_right,
                "exercise_duration_seconds": total_duration_seconds
            }
        }
    }

    print(metrics_data)
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_LateralWeightShifts_metrics.txt"
    # Save the metrics to a file
    save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)

    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Standing Metrics Data"
    sub_directory = "LateralWeightShifts Metrics Data"

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
        