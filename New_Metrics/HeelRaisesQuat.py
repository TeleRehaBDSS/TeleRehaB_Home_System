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
from scipy.signal import butter, filtfilt, medfilt, find_peaks, correlate
from scipy.ndimage import gaussian_filter1d


def preprocess_signal(y_data):
    return gaussian_filter1d(y_data, sigma=3)

def estimate_peak_distance(signal, sampling_rate=100):
    """
    Estimate the dominant period of the signal using autocorrelation.
    The peak distance is set as a fraction of the dominant period.
    """
    signal = signal - np.mean(signal)  # Remove DC component
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Take positive lags only
    peaks, _ = find_peaks(autocorr)
    
    if len(peaks) > 1:
        dominant_period = peaks[1]  # First peak gives the period
    else:
        dominant_period = sampling_rate  # Fallback to sampling rate
    
    return max(int(dominant_period * 0.5), 1)  # Use half the period as peak distance

def detect_valleys(y_data, sampling_rate=100, prominence=0.002):
    """
    Detect valleys (negative peaks) with dynamic distance calculation.
    """
    y_data_smooth = preprocess_signal(y_data)
    dynamic_distance = estimate_peak_distance(y_data_smooth, sampling_rate)
    valleys, _ = find_peaks(y_data_smooth, prominence=prominence, distance=dynamic_distance)
    return valleys

def calculate_metrics_with_sampling(valleys, timestamps, y_data, sampling_frequency=100):
    sampling_period = 1 / sampling_frequency
    num_movements = len(valleys)
    durations = []
    ranges = []

    for i in range(len(valleys) - 1):
        start = valleys[i]
        end = valleys[i + 1]
        duration = (end - start) * sampling_period
        range_deg = abs(y_data[start] - y_data[end]) * 180 / np.pi
        durations.append(duration)
        ranges.append(range_deg)
    
    metrics = {
        "number_of_movements": num_movements,
        "pace_movements_per_second": num_movements / (len(y_data) * sampling_period),
        "mean_combined_range_degrees": np.mean(ranges) if ranges else 0,
        "std_combined_range_degrees": np.std(ranges) if ranges else 0,
        "mean_duration_seconds": np.mean(durations) if durations else 0,
        "std_duration_seconds": np.std(durations) if durations else 0,
        "exercise_duration_seconds": len(y_data) * sampling_period,
        "movement_duration": durations
    }
    return metrics

# Function to apply a Butterworth low-pass filter
def low_pass_filter(signal, cutoff_freq, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Function to remove spikes with median-based threshold
def remove_spikes(signal, threshold_factor=3.0):
    median = np.median(signal)
    std_dev = np.std(signal)
    spike_threshold_upper = median + threshold_factor * std_dev
    spike_threshold_lower = median - threshold_factor * std_dev
    cleaned_signal = np.where((signal > spike_threshold_upper) | (signal < spike_threshold_lower), median, signal)
    return cleaned_signal

# Load and preprocess the signal
def load_and_preprocess_signal(file_path, column_name='z'):
    data = pd.read_csv(file_path)
    signal = data[column_name].values
    signal = remove_spikes(signal, threshold_factor=3.0)
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    median_filtered_signal = medfilt(normalized_signal, kernel_size=5)
    return low_pass_filter(median_filtered_signal, cutoff_freq=2, sampling_rate=100)

# Detect segments based on minima and maxima
def detect_segments(signal, sampling_rate=100):
    minima_indices = find_peaks(-signal, prominence=0.1)[0]
    maxima_indices = find_peaks(signal, prominence=0.1)[0]

    # Adjust start and end to capture full signal range
    first_max_idx = maxima_indices[0] if len(maxima_indices) > 0 else None
    last_max_idx = maxima_indices[-1] if len(maxima_indices) > 0 else None

    if first_max_idx is not None:
        first_minima = minima_indices[minima_indices < first_max_idx]
        if len(first_minima) > 0:
            minima_indices = np.insert(minima_indices, 0, first_minima[-1])

    if last_max_idx is not None:
        last_minima = minima_indices[minima_indices > last_max_idx]
        if len(last_minima) > 0 and last_minima[0] > minima_indices[-1]:  # Append last minima after the last maximum
            minima_indices = np.append(minima_indices, last_minima[0])

    # Segment the signal into "up" and "down" sequences
    segments = []
    for i in range(len(minima_indices) - 1):
        start_min_idx = minima_indices[i]
        end_min_idx = minima_indices[i + 1]
        
        # Find the maximum in between these minima
        max_between = [idx for idx in maxima_indices if start_min_idx < idx < end_min_idx]
        if max_between:
            max_idx = max_between[0]
            duration_up = (max_idx - start_min_idx) / sampling_rate
            duration_down = (end_min_idx - max_idx) / sampling_rate
            amplitude_up = signal[max_idx] - signal[start_min_idx]
            amplitude_down = signal[max_idx] - signal[end_min_idx]

            segments.append({
                "start_min_idx": start_min_idx,
                "max_idx": max_idx,
                "end_min_idx": end_min_idx,
                "duration_up": duration_up,
                "duration_down": duration_down,
                "amplitude_up": amplitude_up,
                "amplitude_down": amplitude_down
            })

    return segments, maxima_indices, minima_indices

# Plot the signal with detected segments
def plot_signal_with_segments(signal, maxima_indices, minima_indices, segments):
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="Preprocessed Signal", color='blue')
    plt.plot(maxima_indices, signal[maxima_indices], "go", label="Maxima")
    plt.plot(minima_indices, signal[minima_indices], "ro", label="Minima")
    
    # Highlight each segment
    for segment in segments:
        plt.axvspan(segment["start_min_idx"], segment["end_min_idx"], color='gray', alpha=0.3)
        plt.text((segment["start_min_idx"] + segment["end_min_idx"]) / 2, 
                 signal[segment["max_idx"]],
                 f'Up-Down', color='purple', ha='center')

    plt.legend()
    plt.title("Signal with Detected Segments (Up-Down)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

def process_imu_data(imu_data_lists, fs, plotdiagrams=True):
    # Ensure lists are not empty and convert to DataFrames
    dataframes = []
    initial_empty_lists = [len(imu_data) == 0 for imu_data in imu_data_lists]  # Track initially empty lists

    c = 0;
    for imu_data in imu_data_lists:
        if imu_data:
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
        returnedJson = getMetricsSittingNew03(Limu3, Limu4, False) 
        return returnedJson

def getMetricsSittingNew03(Limu3, Limu4, plotdiagrams):
    # Limu3
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    df_Limu3['elapsed(time)'] = pd.to_datetime(df_Limu3['elapsed(time)'], unit='ms')
    df_Limu3 = df_Limu3.sort_values(by='elapsed(time)')
    df_Limu3.set_index('elapsed(time)', inplace=True)
    
    # Limu4
    df_Limu4 = pd.DataFrame(Limu4, columns=columns)
    df_Limu4['elapsed(time)'] = pd.to_datetime(df_Limu4['elapsed(time)'], unit='ms')
    df_Limu4 = df_Limu4.sort_values(by='elapsed(time)')
    df_Limu4.set_index('elapsed(time)', inplace=True)
    
    
    # Preprocess y-axis data for both feet
    df_Limu3['y_smoothed'] = preprocess_signal(df_Limu3['Y (number)'].values)
    df_Limu4['y_smoothed'] = preprocess_signal(df_Limu4['Y (number)'].values)
    
    # Detect valleys (movements) in both feet
    left_valleys = detect_valleys(df_Limu3['y_smoothed'].values)
    right_valleys = detect_valleys(df_Limu4['y_smoothed'].values)

    # Calculate metrics for each foot
    left_metrics = calculate_metrics_with_sampling(left_valleys, df_Limu3['Timestamp'].values, df_Limu3['y_smoothed'].values)
    right_metrics = calculate_metrics_with_sampling(right_valleys, df_Limu4['Timestamp'].values, df_Limu4['y_smoothed'].values)

    # Combine metrics data for saving
    metrics_data = {
        "total_metrics": {
            "RIGHT LEG": right_metrics,
            "LEFT LEG": left_metrics
        }
    }
    
    print(metrics_data);

    # Plot diagrams if required
    if plotdiagrams:
        plt.figure(figsize=(10, 6))
        plt.plot(df_Limu4['y_smoothed'], label="Smoothed y-axis data (Right Foot)")
        plt.plot(right_valleys, df_Limu4['y_smoothed'].iloc[right_valleys], "rx", label="Detected Valleys (Right Foot)")
        plt.title("Right Foot - Detected Movements (Valleys)")
        plt.xlabel("Sample Index")
        plt.ylabel("y-axis Smoothed Value")
        plt.legend()
        plt.show()

    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_HeelRaises_metrics.txt"
    save_metrics_to_txt(metrics_data, filename)
    
    return json.dumps(metrics_data, indent=4)

def save_metrics_to_txt(metrics, file_path):
    main_directory = "Sitting Metrics Data"
    sub_directory = "HeelRaises Metrics Data"
    directory = os.path.join(main_directory, sub_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_path = os.path.join(directory, file_path)

    with open(full_path, 'w') as file:
        for main_key, main_value in metrics.items():
            file.write(f"{main_key}:\n")
            if isinstance(main_value, list):  # For segment details
                for i, segment in enumerate(main_value, 1):
                    file.write(f"  Segment {i}:\n")
                    for key, value in segment.items():
                        file.write(f"    {key}: {value}\n")
                    file.write("\n")
            else:
                file.write(f"  {main_value}\n")
        file.write("\n")