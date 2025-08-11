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
from scipy.fft import fft, fftfreq
from scipy.interpolate import UnivariateSpline

total_duration_seconds = 0  # global variable

def process_imu_data_acceleration(imu_data_lists, fs, plotdiagrams=False):
    # Ensure lists are not empty and convert to DataFrames
    dataframes = []
    initial_empty_lists = [len(imu_data) == 0 for imu_data in imu_data_lists]  # Track initially empty lists

    c = 0;
    for imu_data in imu_data_lists:
        if imu_data:
            #print('imu_data = ', imu_data)
            columns = ['Timestamp', 'elapsed(time)', 'X(number)', 'Y(number)', 'Z(number)']
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
            plt.plot(df.index, df['X(number)'], label='X')
            plt.plot(df.index, df['Y(number)'], label='Y')
            plt.plot(df.index, df['Z(number)'], label='Z')
            plt.xlabel('Timestamp')
            plt.ylabel('Acceleration Components')
            plt.title(f'IMU {idx+1} Acceleration Components (X, Y, Z) over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'acceleration_components_plot_{idx+1}.png')
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




def reformat_sensor_data_acceleration(sensor_data_list):
    if not sensor_data_list:
        return []

    # Get the reference timestamp
    reference_timestamp = sensor_data_list[0].timestamp

    reformatted_data = []

    # Iterate through the sensor data list
    for data in sensor_data_list:
        timestamp = data.timestamp
        elapsed_time = timestamp - reference_timestamp
        reformatted_entry = [timestamp, elapsed_time, data.x, data.y, data.z]
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
    plt.show()

# Define thresholds
intersection_distance_threshold = 0.2  # seconds
signal_magnitude_threshold = 0.15  # magnitude


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
        print(item)
        t = item[2:-1]
        if ',' in t:
            t = t.split(',')
        else:
            t = t.split(' ')
        if "number" not in t:
            A.append([t[-6],t[-4],t[-3],t[-2],t[-1]])
    return A

def get_metrics(imu1,imu2,imu3,imu4, counter):
    Limu1 = reformat_sensor_data_acceleration(imu1)
    Limu2 = reformat_sensor_data_acceleration(imu2)
    Limu3 = reformat_sensor_data_acceleration(imu3)   
    Limu4 = reformat_sensor_data_acceleration(imu4)
    imu_data_lists = [Limu1, Limu2, Limu3, Limu4]
    
    processed_dataframes, c = process_imu_data_acceleration(imu_data_lists, 50, True)
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
        returnedJson = getMetricsGaitNew01(Limu3, Limu4, False) 
        return returnedJson


    
def getMetricsGaitNew01(Limu3, Limu4, plotdiagrams):

    global total_duration_seconds

    # #Limu1-LEFT
    columns = ['Timestamp', 'elapsed(time)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    # df_Limu3['Timestamp'] = pd.to_datetime(df_Limu3['Timestamp'], unit='ms')
    # df_Limu3 = df_Limu3.sort_values(by='Timestamp')
    # df_Limu3.set_index('Timestamp', inplace=True)
    
    # #Limu2-RIGHT
    df_Limu4 = pd.DataFrame(Limu4, columns=columns)
    # df_Limu4['Timestamp'] = pd.to_datetime(df_Limu4['Timestamp'], unit='ms')
    # df_Limu4 = df_Limu4.sort_values(by='Timestamp')
    # df_Limu4.set_index('Timestamp', inplace=True)

    # Convert the Timestamp column to datetime
    df_Limu3['Timestamp'] = pd.to_datetime(df_Limu3['Timestamp'], unit='ms')
    df_Limu4['Timestamp'] = pd.to_datetime(df_Limu4['Timestamp'], unit='ms')

    # Sort dataframes by Timestamp and remove duplicates
    df_Limu3 = df_Limu3.sort_values(by='Timestamp').drop_duplicates(subset='Timestamp').reset_index(drop=True)
    df_Limu4 = df_Limu4.sort_values(by='Timestamp').drop_duplicates(subset='Timestamp').reset_index(drop=True)

    # Ensure timestamps are strictly increasing
    df_Limu3 = df_Limu3[df_Limu3['Timestamp'].diff().dt.total_seconds() > 0]
    df_Limu4 = df_Limu4[df_Limu4['Timestamp'].diff().dt.total_seconds() > 0]

    # Find the common time period
    start_time = max(df_Limu3['Timestamp'].min(), df_Limu4['Timestamp'].min())
    end_time = min(df_Limu3['Timestamp'].max(), df_Limu4['Timestamp'].max())

    # Crop the dataframes to the common time period
    df_Limu3 = df_Limu3[(df_Limu3['Timestamp'] >= start_time) & (df_Limu3['Timestamp'] <= end_time)].reset_index(drop=True)
    df_Limu4 = df_Limu4[(df_Limu4['Timestamp'] >= start_time) & (df_Limu4['Timestamp'] <= end_time)].reset_index(drop=True)

    print("Start Time:", start_time)
    print("End Time:", end_time)
    # print("Left IMU Data Period After Cropping:", left_imu['Timestamp'].min(), "to", left_imu['Timestamp'].max())
    # print("Right IMU Data Period After Cropping:", right_imu['Timestamp'].min(), "to", right_imu['Timestamp'].max())


    
    linear_df1 = df_Limu3;
    linear_df2 = df_Limu4;
    
    # Calculate the magnitude of the linear acceleration
    linear_df1['Magnitude'] = np.sqrt(linear_df1['X(number)']**2 + linear_df1['Y (number)']**2 + linear_df1['Z (number)']**2)
    linear_df2['Magnitude'] = np.sqrt(linear_df2['X(number)']**2 + linear_df2['Y (number)']**2 + linear_df2['Z (number)']**2)

    # Apply a low-pass filter to the magnitude signals
    fs = 100  # Sampling frequency (Hz)
    cutoff = 3.0  # Cutoff frequency (Hz)


    #     # Apply the filter to the Yaw data
    # Z_filtered1 = butter_lowpass_filter(linear_df1['Z (number)'], cutoff, fs, order=5)
    # X_filtered1 = butter_lowpass_filter(linear_df1['X(number)'], cutoff, fs, order=5)
    # Y_filtered1 = butter_lowpass_filter(linear_df1['Y (number)'], cutoff, fs, order=5)

    # Z_filtered2 = butter_lowpass_filter(linear_df2['Z (number)'], cutoff, fs, order=5)
    # X_filtered2 = butter_lowpass_filter(linear_df2['X(number)'], cutoff, fs, order=5)
    # Y_filtered2 = butter_lowpass_filter(linear_df2['Y (number)'], cutoff, fs, order=5)


    linear_df1['Filtered_Magnitude'] = butter_lowpass_filter(linear_df1['Magnitude'], cutoff, fs)
    linear_df2['Filtered_Magnitude'] = butter_lowpass_filter(linear_df2['Magnitude'], cutoff, fs)

# Convert Timestamps to nanoseconds for spline fitting
    left_timestamps_ns = linear_df1['Timestamp'].astype(np.int64)
    right_timestamps_ns = linear_df2['Timestamp'].astype(np.int64)

# Create a common time series at 100 Hz within this timeframe
    common_time = pd.date_range(start=start_time, end=end_time, freq='10ms')  # 100 Hz = 10ms intervals
    common_time_ns = common_time.astype(np.int64)

# Fit a spline to the filtered magnitude signal of each leg
    left_spline_filtered_magnitude = UnivariateSpline(left_timestamps_ns, linear_df1['Filtered_Magnitude'], s=0)
    right_spline_filtered_magnitude = UnivariateSpline(right_timestamps_ns, linear_df2['Filtered_Magnitude'], s=0)

# Sample the spline at the time points of the common time series
    left_filtered_magnitude_interpolated = left_spline_filtered_magnitude(common_time_ns)
    right_filtered_magnitude_interpolated = right_spline_filtered_magnitude(common_time_ns)

    # # Plot the original, filtered, and spline-interpolated magnitude data for the left leg
    # plt.figure(figsize=(14, 6))
    # plt.plot(linear_df1['Timestamp'], linear_df1['Magnitude'], 'o', label='Left IMU Magnitude Original', color='blue')
    # plt.plot(linear_df1['Timestamp'], linear_df1['Filtered_Magnitude'], label='Left IMU Magnitude Filtered', color='green')
    # plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='cyan')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Linear Acceleration (Magnitude)')
    # plt.title('Left IMU Magnitude - Original, Filtered, and Spline')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot the original, filtered, and spline-interpolated magnitude data for the right leg
    # plt.figure(figsize=(14, 6))
    # plt.plot(linear_df2['Timestamp'], linear_df2['Magnitude'], 'o', label='Right IMU Magnitude Original', color='red')
    # plt.plot(linear_df2['Timestamp'], linear_df2['Filtered_Magnitude'], label='Right IMU Magnitude Filtered', color='orange')
    # plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='purple')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Linear Acceleration (Magnitude)')
    # plt.title('Right IMU Magnitude - Original, Filtered, and Spline')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot the spline-interpolated magnitude data for both legs
    # plt.figure(figsize=(14, 6))
    # plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue')
    # plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red')
    # plt.xlabel('Timestamp')
    # plt.ylabel('Linear Acceleration (Magnitude)')
    # plt.title('Spline-Interpolated Linear Acceleration Magnitude - Left and Right IMU')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Determine the time periods when the "left" spline is greater than the "right" and vice versa
    left_greater = left_filtered_magnitude_interpolated > right_filtered_magnitude_interpolated
    right_greater = right_filtered_magnitude_interpolated > left_filtered_magnitude_interpolated

    # # Create a step-wise plot for the comparison
    # plt.figure(figsize=(14, 6))
    # plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue', alpha=0.5)
    # plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red', alpha=0.5)

    # # Highlight the periods where left is greater than right and vice versa
    # plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
    #                 where=left_greater, facecolor='blue', alpha=0.3, label='Left > Right')
    # plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
    #                 where=right_greater, facecolor='red', alpha=0.3, label='Right > Left')

    # plt.xlabel('Timestamp')
    # plt.ylabel('Linear Acceleration (Magnitude)')
    # plt.title('Spline-Interpolated Linear Acceleration Magnitude - Left and Right IMU with Highlighted Intervals')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Detect intersection points
    proximity_threshold = 0.01  # Adjust this threshold based on your data

    # Detect intersection points and points where the curves are really close# Detect intersection points and points where the curves are really close
    intersection_indices = np.where(
        (np.diff(np.sign(left_filtered_magnitude_interpolated - right_filtered_magnitude_interpolated)) != 0) |
        (np.abs(left_filtered_magnitude_interpolated[:-1] - right_filtered_magnitude_interpolated[:-1]) < proximity_threshold))[0]
    intersection_times = common_time[intersection_indices]

    # Omit closely spaced intersecting points
    filtered_intersection_indices = []
    filtered_intersection_times = []

    for i, time in enumerate(intersection_times):
        if i == 0 or not filtered_intersection_times or (time - filtered_intersection_times[-1]).total_seconds() > intersection_distance_threshold:
            if (left_filtered_magnitude_interpolated[intersection_indices[i]] > signal_magnitude_threshold) or \
            (right_filtered_magnitude_interpolated[intersection_indices[i]] > signal_magnitude_threshold):
                filtered_intersection_indices.append(intersection_indices[i])
                filtered_intersection_times.append(time)

    intersection_indices = filtered_intersection_indices
    intersection_times = pd.to_datetime(filtered_intersection_times)

    # Function to find local extrema (minima and maxima) between two points using find_peaks
    def find_extrema_between(signal, start_idx, end_idx):
        local_segment = signal[start_idx:end_idx]
        peaks, _ = find_peaks(local_segment)
        troughs, _ = find_peaks(-local_segment)
        
        # Ensure that there is only one extrema per region
        local_max = start_idx + peaks[np.argmax(local_segment[peaks])] if len(peaks) > 0 else None
        local_min = start_idx + troughs[np.argmin(local_segment[troughs])] if len(troughs) > 0 else None
        
        return local_min, local_max

    # Find local minima and maxima between intersection points
    left_minima_indices = []
    left_maxima_indices = []
    right_minima_indices = []
    right_maxima_indices = []

    for i in range(len(intersection_indices) - 1):
        start_idx = intersection_indices[i]
        end_idx = intersection_indices[i + 1]

        left_min_idx, left_max_idx = find_extrema_between(left_filtered_magnitude_interpolated, start_idx, end_idx)
        right_min_idx, right_max_idx = find_extrema_between(right_filtered_magnitude_interpolated, start_idx, end_idx)
        
        if left_min_idx is not None:
            left_minima_indices.append(left_min_idx)
        if left_max_idx is not None:
            left_maxima_indices.append(left_max_idx)
        if right_min_idx is not None:
            right_minima_indices.append(right_min_idx)
        if right_max_idx is not None:
            right_maxima_indices.append(right_max_idx)


    # Determine the gait phases by comparing the mean magnitudes between intersection points
    gait_phases = []
    for i in range(len(intersection_times) - 1):
        start_idx = intersection_indices[i]
        end_idx = intersection_indices[i + 1]
        left_mean = np.mean(left_filtered_magnitude_interpolated[start_idx:end_idx])
        right_mean = np.mean(right_filtered_magnitude_interpolated[start_idx:end_idx])
        if left_mean < right_mean:
            gait_phases.append(('Left Stance, Right Swing', start_idx, end_idx))
        else:
            gait_phases.append(('Right Stance, Left Swing', start_idx, end_idx))

    # Calculate heel strikes and toe offs based on the determined gait phases
    left_hs_indices = []
    left_to_indices = []
    right_hs_indices = []
    right_to_indices = []

    for i in range(len(gait_phases)):
        phase, start_idx, end_idx = gait_phases[i]
        
        if 'Left Stance' in phase:
            left_min_idx, _ = find_extrema_between(left_filtered_magnitude_interpolated, start_idx, end_idx)
            if left_min_idx is not None:
                hs_idx = start_idx + (left_min_idx - start_idx) // 2
                left_hs_indices.append(hs_idx)
                to_idx = left_min_idx + (end_idx - left_min_idx) // 2
                left_to_indices.append(to_idx)
        elif 'Right Stance' in phase:
            right_min_idx, _ = find_extrema_between(right_filtered_magnitude_interpolated, start_idx, end_idx)
            if right_min_idx is not None:
                hs_idx = start_idx + (right_min_idx - start_idx) // 2
                right_hs_indices.append(hs_idx)
                to_idx = right_min_idx + (end_idx - right_min_idx) // 2
                right_to_indices.append(to_idx)

    # # Print calculated indices for verification
    # print(f"Left Heel Strike Indices: {left_hs_indices}")
    # print(f"Left Toe Off Indices: {left_to_indices}")
    # print(f"Right Heel Strike Indices: {right_hs_indices}")
    # print(f"Right Toe Off Indices: {right_to_indices}")

    # # Plot the results with the new heel strikes and toe offs
    # plt.figure(figsize=(14, 6))
    # plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue', alpha=0.5)
    # plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red', alpha=0.5)

    # # Highlight the periods where left is greater than right and vice versa
    # plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
    #                 where=left_filtered_magnitude_interpolated < right_filtered_magnitude_interpolated, facecolor='blue', alpha=0.3, label='Left Stance')
    # plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
    #                 where=right_filtered_magnitude_interpolated < left_filtered_magnitude_interpolated, facecolor='red', alpha=0.3, label='Right Stance')

    # # Mark the intersection points
    # plt.scatter(intersection_times, left_filtered_magnitude_interpolated[intersection_indices], color='black', zorder=5)

    # # Mark heel strikes and toe offs
    # plt.scatter(common_time[left_hs_indices], left_filtered_magnitude_interpolated[left_hs_indices], color='green', zorder=5, label='Left Heel Strikes')
    # plt.scatter(common_time[left_to_indices], left_filtered_magnitude_interpolated[left_to_indices], color='cyan', zorder=5, label='Left Toe Offs')
    # plt.scatter(common_time[right_hs_indices], right_filtered_magnitude_interpolated[right_hs_indices], color='yellow', zorder=5, label='Right Heel Strikes')
    # plt.scatter(common_time[right_to_indices], right_filtered_magnitude_interpolated[right_to_indices], color='magenta', zorder=5, label='Right Toe Offs')

    # plt.xlabel('Timestamp')
    # plt.ylabel('Linear Acceleration (Magnitude)')
    # plt.title('Spline-Interpolated Linear Acceleration Magnitude - Gait Events with Heel Strikes and Toe Offs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # Calculate overall gait metrics
    stride_times = [(end - start).total_seconds() for start, end in zip(intersection_times[:-1], intersection_times[1:])]
    step_times = [time / 2 for time in stride_times]
    cadence = (len(step_times) / ((common_time[-1] - common_time[0]).total_seconds())) * 60  # Steps per minute
    gait_cycle_times = stride_times  # Each stride is one gait cycle

    # Calculate individual foot gait metrics
    left_stance_times = []
    right_stance_times = []
    left_swing_times = []
    right_swing_times = []
    double_support_times = []
    left_toe_off_times = []
    right_toe_off_times = []
    left_swing_peak_times = []
    right_swing_peak_times = []

    # Calculate metrics for each phase
    for phase, start_idx, end_idx in gait_phases:
        start_time = common_time[start_idx]
        end_time = common_time[end_idx]
        duration = (end_time - start_time).total_seconds()
        if 'Left Swing' in phase:
            left_swing_times.append(duration)
            # Calculate toe off and swing peak times
            for idx in left_maxima_indices:
                if start_time <= common_time[idx] <= end_time:
                    left_swing_peak_times.append((common_time[idx] - start_time).total_seconds())
            for idx in right_minima_indices:
                if start_time <= common_time[idx] <= end_time:
                    right_toe_off_times.append((common_time[idx] - start_time).total_seconds())
        else:
            right_swing_times.append(duration)
            # Calculate toe off and swing peak times
            for idx in right_maxima_indices:
                if start_time <= common_time[idx] <= end_time:
                    right_swing_peak_times.append((common_time[idx] - start_time).total_seconds())
            for idx in left_minima_indices:
                if start_time <= common_time[idx] <= end_time:
                    left_toe_off_times.append((common_time[idx] - start_time).total_seconds())

    # Calculate stance times
    for i in range(0, len(intersection_times) - 1, 2):
        left_stance_times.append((intersection_times[i+1] - intersection_times[i]).total_seconds())
        if i + 2 < len(intersection_times):
            right_stance_times.append((intersection_times[i+2] - intersection_times[i+1]).total_seconds())



    # # Print overall gait metrics
    # print(f"Stride Times: {stride_times}")
    # print(f"Step Times: {step_times}")
    # print(f"Cadence: {cadence} steps/minute")
    # print(f"Gait Cycle Times: {gait_cycle_times}")



    # Calculate the times for the metrics
    right_single_support_times = []
    left_single_support_times = []
    double_support_times = []

    # Iterate through the cycles to calculate the times
    for i in range(min(len(right_to_indices), len(left_to_indices)) - 1):
        t2 = common_time[left_to_indices[i]]
        t5 = common_time[left_hs_indices[i + 1]]
        right_single_support_times.append((t5 - t2).total_seconds())
        
        t6 = common_time[right_to_indices[i]]
        t1 = common_time[right_hs_indices[i + 1]]
        left_single_support_times.append((t1 - t6).total_seconds())
        
        t1 = common_time[right_hs_indices[i]]
        t2 = common_time[left_to_indices[i]]
        t5 = common_time[left_hs_indices[i]]
        t6 = common_time[right_to_indices[i]]
        double_support_times.append((t2 - t1).total_seconds() + (t6 - t5).total_seconds())

    # # Print calculated gait metrics
    # print(f"Right Single Support Times: {right_single_support_times}")
    # print(f"Left Single Support Times: {left_single_support_times}")
    # print(f"Double Support Times: {double_support_times}")

    '''
    Right stance phase duration a4 = t6 âˆ’ t1
    Left stance phase duration a5 = t2|nextcycle - t5
    Right load response time a6 = t2 âˆ’ t1
    Right terminal stance time a7 = t4 âˆ’ t2
    Right pre-swing time a8 = t5 âˆ’ t4
    Right gait cycle time a9 = t1|nextcycle - t1
    Left loading response time a10 = t6 âˆ’ t5
    Left terminal stance time a11 = t7 âˆ’ t6
    Left pre-swing phase time a12 = t1|nextcycle - t7
    Left gait cycle a13 = t2|nextcycle - t2
    '''

    # # Plot the results with the new heel strikes, toe offs, and metrics
    # plt.figure(figsize=(14, 6))
    # plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue', alpha=0.5)
    # plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red', alpha=0.5)

    # # Highlight the periods where left is greater than right and vice versa
    # plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
    #                 where=left_filtered_magnitude_interpolated < right_filtered_magnitude_interpolated, facecolor='blue', alpha=0.3, label='Left Stance')
    # plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
    #                 where=right_filtered_magnitude_interpolated < left_filtered_magnitude_interpolated, facecolor='red', alpha=0.3, label='Right Stance')
    # # Mark the intersection points
    # plt.scatter(intersection_times, left_filtered_magnitude_interpolated[intersection_indices], color='black', zorder=5)

    # # Mark heel strikes and toe offs
    # plt.scatter(common_time[left_hs_indices], left_filtered_magnitude_interpolated[left_hs_indices], color='green', zorder=5, label='Left Heel Strikes')
    # plt.scatter(common_time[left_to_indices], left_filtered_magnitude_interpolated[left_to_indices], color='cyan', zorder=5, label='Left Toe Offs')
    # plt.scatter(common_time[right_hs_indices], right_filtered_magnitude_interpolated[right_hs_indices], color='yellow', zorder=5, label='Right Heel Strikes')
    # plt.scatter(common_time[right_to_indices], right_filtered_magnitude_interpolated[right_to_indices], color='magenta', zorder=5, label='Right Toe Offs')

    # Annotate gait events
    for phase, start_idx, end_idx in gait_phases:
        start_time = common_time[start_idx]
        end_time = common_time[end_idx]
        mid_time = pd.Timestamp((start_time.timestamp() + end_time.timestamp()) / 2, unit='s')
    #     plt.axvspan(start_time, end_time, color='yellow' if 'Left Swing' in phase else 'green', alpha=0.2)
    #     plt.text(mid_time, plt.ylim()[1] - 0.1 * plt.ylim()[1], phase, horizontalalignment='center', verticalalignment='top', fontsize=8, rotation=45)

    # plt.xlabel('Timestamp')
    # plt.ylabel('Linear Acceleration (Magnitude)')
    # plt.title('Spline-Interpolated Linear Acceleration Magnitude - Gait Events with Heel Strikes and Toe Offs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    df_Limu3['elapsed(time)'] = pd.to_datetime(df_Limu3['elapsed(time)'], unit='ms')
    df_Limu3.sort_values(by='elapsed(time)', inplace=True)
    df_Limu3.set_index('elapsed(time)', inplace=True)

    interval_duration_seconds = (df_Limu3.index[-1] - df_Limu3.index[0]).total_seconds()
    total_duration_seconds += interval_duration_seconds
    
    metrics_data = {
            "total_metrics": {
                "Gait Cycle":{
                    "Number of steps": int(len(step_times)),
                    "Right Single Support Time": right_single_support_times,
                    "Left Single Support Time": left_single_support_times,
                    "Right Single Support Time": right_single_support_times,
                    "Left Single Support Time": left_single_support_times,
                    "Double Support Time": double_support_times,
                    "Stride Phase Duration": stride_times,
                    "Gait Cycle Duration": gait_cycle_times,
                    "Right stance time ": right_stance_times,
                    "Right Swing Time": right_swing_times,
                    "Left stance Time": left_stance_times,
                    "Left Swing Time": left_swing_times,
                    "Right Toe Off Times": right_toe_off_times,
                    "Right Swing peak Times": right_swing_peak_times,
                    "Left Toe Off Times": left_toe_off_times,
                    "Left Swing peak Times": left_swing_peak_times,
                    "Cadence": cadence,
                    "exercise_duration_seconds": total_duration_seconds
                                }
                
        }
        }


    print(metrics_data);
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_SideStepping_metrics.txt"

        # Save the metrics to a file
    save_metrics_to_txt(metrics_data, filename)

    return json.dumps(metrics_data, indent=4)

        
def save_metrics_to_txt(metrics, file_path):
        main_directory = "Gait Metrics Data"
        sub_directory = "SideStepping Metrics Data"

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
# def calculate_correlation(metric1, metric2):
#     # Calculate the correlation between two metrics
#     correlation = metric1.corr(metric2)
#     return correlation