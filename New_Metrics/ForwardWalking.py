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
from scipy.interpolate import UnivariateSpline

# Define thresholds
intersection_distance_threshold = 0.2  # seconds
signal_magnitude_threshold = 0.15  # magnitude

def gaitanalysis(left_imu,right_imu,plotdiagrams):
    # Define thresholds
    intersection_distance_threshold = 0.2  # seconds
    signal_magnitude_threshold = 0.15  # magnitude
    # Convert the Timestamp column to datetime
    left_imu['Timestamp'] = pd.to_datetime(left_imu['Timestamp'], unit='ms')
    right_imu['Timestamp'] = pd.to_datetime(right_imu['Timestamp'], unit='ms')

    # Sort dataframes by Timestamp and remove duplicates
    left_imu = left_imu.sort_values(by='Timestamp').drop_duplicates(subset='Timestamp').reset_index(drop=True)
    right_imu = right_imu.sort_values(by='Timestamp').drop_duplicates(subset='Timestamp').reset_index(drop=True)

    # Ensure timestamps are strictly increasing
    left_imu = left_imu[left_imu['Timestamp'].diff().dt.total_seconds() > 0]
    right_imu = right_imu[right_imu['Timestamp'].diff().dt.total_seconds() > 0]

    # Find the common time period
    start_time = max(left_imu['Timestamp'].min(), right_imu['Timestamp'].min())
    end_time = min(left_imu['Timestamp'].max(), right_imu['Timestamp'].max())

    # Crop the dataframes to the common time period
    left_imu = left_imu[(left_imu['Timestamp'] >= start_time) & (left_imu['Timestamp'] <= end_time)].reset_index(drop=True)
    right_imu = right_imu[(right_imu['Timestamp'] >= start_time) & (right_imu['Timestamp'] <= end_time)].reset_index(drop=True)

    print("Start Time:", start_time)
    print("End Time:", end_time)
    #print("Left IMU Data Period After Cropping:", left_imu['Timestamp'].min(), "to", left_imu['Timestamp'].max())
    #print("Right IMU Data Period After Cropping:", right_imu['Timestamp'].min(), "to", right_imu['Timestamp'].max())

    # Calculate the magnitude of the linear acceleration
    left_imu['Magnitude'] = np.sqrt(left_imu['X(number)']**2 + left_imu['Y (number)']**2 + left_imu['Z (number)']**2)
    right_imu['Magnitude'] = np.sqrt(right_imu['X(number)']**2 + right_imu['Y (number)']**2 + right_imu['Z (number)']**2)

    # Apply a low-pass filter to the magnitude signals
    fs = 100  # Sampling frequency (Hz)
    cutoff = 3.0  # Cutoff frequency (Hz)

    left_imu['Filtered_Magnitude'] = butter_lowpass_filter(left_imu['Magnitude'], cutoff, fs)
    right_imu['Filtered_Magnitude'] = butter_lowpass_filter(right_imu['Magnitude'], cutoff, fs)

    # Convert Timestamps to nanoseconds for spline fitting
    left_timestamps_ns = left_imu['Timestamp'].astype(np.int64)
    right_timestamps_ns = right_imu['Timestamp'].astype(np.int64)

    # Create a common time series at 100 Hz within this timeframe
    common_time = pd.date_range(start=start_time, end=end_time, freq='10ms')  # 100 Hz = 10ms intervals
    common_time_ns = common_time.astype(np.int64)

    # Fit a spline to the filtered magnitude signal of each leg
    left_spline_filtered_magnitude = UnivariateSpline(left_timestamps_ns, left_imu['Filtered_Magnitude'], s=0)
    right_spline_filtered_magnitude = UnivariateSpline(right_timestamps_ns, right_imu['Filtered_Magnitude'], s=0)

    # Sample the spline at the time points of the common time series
    left_filtered_magnitude_interpolated = left_spline_filtered_magnitude(common_time_ns)
    right_filtered_magnitude_interpolated = right_spline_filtered_magnitude(common_time_ns)
    if (plotdiagrams):
        # Plot the original, filtered, and spline-interpolated magnitude data for the left leg
        plt.figure(figsize=(14, 6))
        plt.plot(left_imu['Timestamp'], left_imu['Magnitude'], 'o', label='Left IMU Magnitude Original', color='blue')
        plt.plot(left_imu['Timestamp'], left_imu['Filtered_Magnitude'], label='Left IMU Magnitude Filtered', color='green')
        plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='cyan')
        plt.xlabel('Timestamp')
        plt.ylabel('Linear Acceleration (Magnitude)')
        plt.title('Left IMU Magnitude - Original, Filtered, and Spline')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the original, filtered, and spline-interpolated magnitude data for the right leg
        plt.figure(figsize=(14, 6))
        plt.plot(right_imu['Timestamp'], right_imu['Magnitude'], 'o', label='Right IMU Magnitude Original', color='red')
        plt.plot(right_imu['Timestamp'], right_imu['Filtered_Magnitude'], label='Right IMU Magnitude Filtered', color='orange')
        plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='purple')
        plt.xlabel('Timestamp')
        plt.ylabel('Linear Acceleration (Magnitude)')
        plt.title('Right IMU Magnitude - Original, Filtered, and Spline')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the spline-interpolated magnitude data for both legs
        plt.figure(figsize=(14, 6))
        plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue')
        plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Linear Acceleration (Magnitude)')
        plt.title('Spline-Interpolated Linear Acceleration Magnitude - Left and Right IMU')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Determine the time periods when the "left" spline is greater than the "right" and vice versa
    left_greater = left_filtered_magnitude_interpolated > right_filtered_magnitude_interpolated
    right_greater = right_filtered_magnitude_interpolated > left_filtered_magnitude_interpolated

    if (plotdiagrams):
        # Create a step-wise plot for the comparison
        plt.figure(figsize=(14, 6))
        plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue', alpha=0.5)
        plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red', alpha=0.5)

        # Highlight the periods where left is greater than right and vice versa
        plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
                        where=left_greater, facecolor='blue', alpha=0.3, label='Left > Right')
        plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
                        where=right_greater, facecolor='red', alpha=0.3, label='Right > Left')

        plt.xlabel('Timestamp')
        plt.ylabel('Linear Acceleration (Magnitude)')
        plt.title('Spline-Interpolated Linear Acceleration Magnitude - Left and Right IMU with Highlighted Intervals')
        plt.legend()
        plt.grid(True)
        plt.show()

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

    # Print calculated indices for verification
    # print(f"Left Heel Strike Indices: {left_hs_indices}")
    # print(f"Left Toe Off Indices: {left_to_indices}")
    # print(f"Right Heel Strike Indices: {right_hs_indices}")
    # print(f"Right Toe Off Indices: {right_to_indices}")

    if (plotdiagrams):
        # Plot the results with the new heel strikes and toe offs
        plt.figure(figsize=(14, 6))
        plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue', alpha=0.5)
        plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red', alpha=0.5)
        # Highlight the periods where left is greater than right and vice versa
        plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
                        where=left_filtered_magnitude_interpolated < right_filtered_magnitude_interpolated, facecolor='blue', alpha=0.3, label='Left Stance')
        plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
                        where=right_filtered_magnitude_interpolated < left_filtered_magnitude_interpolated, facecolor='red', alpha=0.3, label='Right Stance')

        # Mark the intersection points
        plt.scatter(intersection_times, left_filtered_magnitude_interpolated[intersection_indices], color='black', zorder=5)

        # Mark heel strikes and toe offs
        plt.scatter(common_time[left_hs_indices], left_filtered_magnitude_interpolated[left_hs_indices], color='green', zorder=5, label='Left Heel Strikes')
        plt.scatter(common_time[left_to_indices], left_filtered_magnitude_interpolated[left_to_indices], color='cyan', zorder=5, label='Left Toe Offs')
        plt.scatter(common_time[right_hs_indices], right_filtered_magnitude_interpolated[right_hs_indices], color='yellow', zorder=5, label='Right Heel Strikes')
        plt.scatter(common_time[right_to_indices], right_filtered_magnitude_interpolated[right_to_indices], color='magenta', zorder=5, label='Right Toe Offs')

        plt.xlabel('Timestamp')
        plt.ylabel('Linear Acceleration (Magnitude)')
        plt.title('Spline-Interpolated Linear Acceleration Magnitude - Gait Events with Heel Strikes and Toe Offs')
        plt.legend()
        plt.grid(True)
        plt.show()


    # Calculate overall gait metrics
    # Consider pairs of heel strikes for stride times
    stride_times = []
    for i in range(len(right_hs_indices) - 2):
        stride_times.append((common_time[right_hs_indices[i + 2]] - common_time[right_hs_indices[i]]).total_seconds())

    step_times = [(end - start).total_seconds() for start, end in zip(intersection_times[:-1], intersection_times[1:])]
    cadence = (len(step_times) / ((common_time[-1] - common_time[0]).total_seconds())) * 60  # Steps per minute
    gait_cycle_times = stride_times  # Each stride is one gait cycle

    # Calculate individual foot gait metrics
    left_stance_times = []
    right_stance_times = []
    left_swing_times = []
    right_swing_times = []
    double_support_times = []
    single_support_times = []
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

    # Calculate double support and single support times
    for i in range(0, len(gait_phases) - 1, 2):
        left_phase_start, left_phase_end = gait_phases[i][1], gait_phases[i][2]
        right_phase_start, right_phase_end = gait_phases[i+1][1], gait_phases[i+1][2]
        
        double_support_start = max(common_time[left_phase_start], common_time[right_phase_start])
        double_support_end = min(common_time[left_phase_end], common_time[right_phase_end])
        double_support_duration = (double_support_end - double_support_start).total_seconds()
        double_support_times.append(double_support_duration)
        
        single_support_duration = (common_time[left_phase_end] - common_time[left_phase_start]).total_seconds() - double_support_duration
        single_support_times.append(single_support_duration)

    # Print overall gait metrics
    # print(f"Step Times: {step_times}")
    # print(f"Cadence: {cadence} steps/minute")
    # print(f"Gait Cycle Times: {gait_cycle_times}")

    # Print individual foot gait metrics
    # print(f"Left Stance Times: {left_stance_times}")
    # print(f"Right Stance Times: {right_stance_times}")
    # print(f"Left Swing Times: {left_swing_times}")
    # print(f"Right Swing Times: {right_swing_times}")
    # print(f"Double Support Times: {double_support_times}")
    # print(f"Single Support Times: {single_support_times}")
    # print(f"Left Toe Off Times: {left_toe_off_times}")
    # print(f"Right Toe Off Times: {right_toe_off_times}")
    # print(f"Left Swing Peak Times: {left_swing_peak_times}")
    # print(f"Right Swing Peak Times: {right_swing_peak_times}")

    # Calculate the times for the additional metrics
    right_stance_phase_durations = []
    left_stance_phase_durations = []
    right_load_response_times = []
    right_terminal_stance_times = []
    right_pre_swing_times = []
    right_gait_cycle_times = []
    left_loading_response_times = []
    left_terminal_stance_times = []
    left_pre_swing_phase_times = []
    left_gait_cycle_times = []

    if len(left_hs_indices) < 1 or len(right_hs_indices) < 1 or len(intersection_times) < 1:
        print("Not enough gait events to compute metrics.")
        return {
        "total_metrics": {
            "number_of_steps": 0,
            "cadence": 0,
            "stride_times": [],
            "left_stance_times": [],
            "right_stance_times": [],
            "left_swing_times": [],
            "right_swing_times": [],
            "double_support_times": [],
            "single_support_times": [],
            "exercise_duration_seconds": (end_time - start_time).total_seconds()
        }
    }
    
    for i in range(len(left_hs_indices) - 1):
        try:
            t1 = common_time[right_hs_indices[i]]
            t2 = common_time[left_to_indices[i]]
            t4 = intersection_times[i * 2 + 1]
            t5 = common_time[left_hs_indices[i + 1]]
            t6 = common_time[right_to_indices[i]]
            t7 = intersection_times[i * 2 + 2] if i * 2 + 2 < len(intersection_times) else intersection_times[-1]
            t1_next = common_time[right_hs_indices[i + 1]] if i + 1 < len(right_hs_indices) else common_time[right_hs_indices[i]]
        except IndexError as e:
            print(f"Skipping index {i} due to insufficient data: {e}")
            continue


        right_stance_phase_durations.append((t6 - t1).total_seconds())
        left_stance_phase_durations.append((t2 - t5).total_seconds() if i + 1 < len(left_hs_indices) else None)
        right_load_response_times.append((t2 - t1).total_seconds())
        right_terminal_stance_times.append((t4 - t2).total_seconds())
        right_pre_swing_times.append((t5 - t4).total_seconds())
        right_gait_cycle_times.append((t1_next - t1).total_seconds())
        left_loading_response_times.append((t6 - t5).total_seconds())
        left_terminal_stance_times.append((t7 - t6).total_seconds() if i * 2 + 2 < len(intersection_times) else None)
        left_pre_swing_phase_times.append((t1_next - t7).total_seconds() if i * 2 + 2 < len(intersection_times) else None)
        left_gait_cycle_times.append((t2 - common_time[left_to_indices[i - 1]]).total_seconds() if i - 1 >= 0 else None)

    if(plotdiagrams):

        # Plot the results with the new heel strikes, toe offs, and metrics
        plt.figure(figsize=(14, 6))
        plt.plot(common_time, left_filtered_magnitude_interpolated, label='Left IMU Magnitude Spline', color='blue', alpha=0.5)
        plt.plot(common_time, right_filtered_magnitude_interpolated, label='Right IMU Magnitude Spline', color='red', alpha=0.5)

        # Highlight the periods where left is greater than right and vice versa
        plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
                        where=left_filtered_magnitude_interpolated < right_filtered_magnitude_interpolated, facecolor='blue', alpha=0.3, label='Left Stance')
        plt.fill_between(common_time, left_filtered_magnitude_interpolated, right_filtered_magnitude_interpolated,
                        where=right_filtered_magnitude_interpolated < left_filtered_magnitude_interpolated, facecolor='red', alpha=0.3, label='Right Stance')

        # Mark the intersection points
        plt.scatter(intersection_times, left_filtered_magnitude_interpolated[intersection_indices], color='black', zorder=5)

        # Mark heel strikes and toe offs
        plt.scatter(common_time[left_hs_indices], left_filtered_magnitude_interpolated[left_hs_indices], color='green', zorder=5, label='Left Heel Strikes')
        plt.scatter(common_time[left_to_indices], left_filtered_magnitude_interpolated[left_to_indices], color='cyan', zorder=5, label='Left Toe Offs')
        plt.scatter(common_time[right_hs_indices], right_filtered_magnitude_interpolated[right_hs_indices], color='yellow', zorder=5, label='Right Heel Strikes')
        plt.scatter(common_time[right_to_indices], right_filtered_magnitude_interpolated[right_to_indices], color='magenta', zorder=5, label='Right Toe Offs')
        # Mark the intersection points
        plt.scatter(intersection_times, left_filtered_magnitude_interpolated[intersection_indices], color='black', zorder=5)

        # Mark heel strikes and toe offs
        plt.scatter(common_time[left_hs_indices], left_filtered_magnitude_interpolated[left_hs_indices], color='green', zorder=5, label='Left Heel Strikes')
        plt.scatter(common_time[left_to_indices], left_filtered_magnitude_interpolated[left_to_indices], color='cyan', zorder=5, label='Left Toe Offs')
        plt.scatter(common_time[right_hs_indices], right_filtered_magnitude_interpolated[right_hs_indices], color='yellow', zorder=5, label='Right Heel Strikes')
        plt.scatter(common_time[right_to_indices], right_filtered_magnitude_interpolated[right_to_indices], color='magenta', zorder=5, label='Right Toe Offs')

        # Annotate gait events
        for phase, start_idx, end_idx in gait_phases:
            start_time = common_time[start_idx]
            end_time = common_time[end_idx]
            mid_time = pd.Timestamp((start_time.timestamp() + end_time.timestamp()) / 2, unit='s')
            plt.axvspan(start_time, end_time, color='yellow' if 'Left Swing' in phase else 'green', alpha=0.2)
            plt.text(mid_time, plt.ylim()[1] - 0.1 * plt.ylim()[1], phase, horizontalalignment='center', verticalalignment='top', fontsize=8, rotation=45)

        plt.xlabel('Timestamp')
        plt.ylabel('Linear Acceleration (Magnitude)')
        plt.title('Spline-Interpolated Linear Acceleration Magnitude - Gait Events with Heel Strikes and Toe Offs')
        plt.legend()
        plt.grid(True)
        plt.show()

    metrics_data = {
        "total_metrics": {
            "number_of_steps": len(step_times),
            "stride_times": stride_times,
            "mean_stride_time": np.mean(stride_times) if stride_times else None,
            "std_stride_time": np.std(stride_times) if stride_times else None,
            "step_times": step_times,
            "mean_step_time": np.mean(step_times) if step_times else None,
            "std_step_time": np.std(step_times) if step_times else None,
            "cadence": cadence,
            "left_stance_times": left_stance_times,
            "mean_left_stance_time": np.mean(left_stance_times) if left_stance_times else None,
            "std_left_stance_time": np.std(left_stance_times) if left_stance_times else None,
            "right_stance_times": right_stance_times,
            "mean_right_stance_time": np.mean(right_stance_times) if right_stance_times else None,
            "std_right_stance_time": np.std(right_stance_times) if right_stance_times else None,
            "left_swing_times": left_swing_times,
            "mean_left_swing_time": np.mean(left_swing_times) if left_swing_times else None,
            "std_left_swing_time": np.std(left_swing_times) if left_swing_times else None,
            "right_swing_times": right_swing_times,
            "mean_right_swing_time": np.mean(right_swing_times) if right_swing_times else None,
            "std_right_swing_time": np.std(right_swing_times) if right_swing_times else None,
            "double_support_times": double_support_times,
            "mean_double_support_time": np.mean(double_support_times) if double_support_times else None,
            "std_double_support_time": np.std(double_support_times) if double_support_times else None,
            "single_support_times": single_support_times,
            "mean_single_support_time": np.mean(single_support_times) if single_support_times else None,
            "std_single_support_time": np.std(single_support_times) if single_support_times else None,
        }
    }
 
    return (metrics_data)


# Function to apply a Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y



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

    # if plotdiagrams:
    #     for idx, df in enumerate(resampled_dataframes):
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(df.index, df['W(number)'], label='W')
    #         plt.plot(df.index, df['X(number)'], label='X')
    #         plt.plot(df.index, df['Y(number)'], label='Y')
    #         plt.plot(df.index, df['Z(number)'], label='Z')
    #         plt.xlabel('Timestamp')
    #         plt.ylabel('Quaternion Components')
    #         plt.title(f'IMU {idx+1} Quaternion Components (W, X, Y, Z) over Time')
    #         plt.legend()
    #         plt.xticks(rotation=45)
    #         plt.tight_layout()
    #         plt.savefig(f'quaternion_components_plot_{idx+1}.png')
    #         # plt.show()

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
    Limu3 = reformat_sensor_data_acceleration(imu3)   
    Limu4 = reformat_sensor_data_acceleration(imu4)

    imu_data_lists1 = [Limu1, Limu2]
    imu_data_lists2 = [Limu3, Limu4]

    processed_dataframes1, c1 = process_imu_data(imu_data_lists1, 50, True)
    processed_dataframes2, c2 = process_imu_data_acceleration(imu_data_lists2, 100, True)


    Limu1 = processed_dataframes1[0]
    Limu2 = processed_dataframes1[1]
    Limu3 = processed_dataframes2[0]
    Limu4 = processed_dataframes2[1]

    if(len(Limu2) > 0 and len(Limu3) > 0 and len(Limu4) > 0 ):
        returnedJson = getMetricsGaitNew01(Limu1, Limu2, Limu3, Limu4 ,False) 
        return returnedJson
    
# Function to apply a low-pass filter
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def getMetricsGaitNew01( Limu1, Limu2,  Limu3, Limu4, plotdiagrams):

    fs1 = 100  # Sampling frequency (Hz) Legs
    cutoff1 = 3.0  # Cutoff frequency (Hz) Legs
    fs2 = 50 # Sampling frequency (Hz) Head
    cutoff2 = 0.5 # Cutoff frequency (Hz) Head

    # #Limu1-HEAD
    columnss = ['Timestamp', 'elapsed(time)',  'W(number)', 'X(number)', 'Y(number)', 'Z(number)']

    df_Limu1 = pd.DataFrame(Limu1, columns=columnss)
    df_Limu1['Timestamp'] = pd.to_datetime(df_Limu1['Timestamp'])
    df_Limu1 = df_Limu1.sort_values(by='Timestamp')
    df_Limu1.set_index('Timestamp', inplace=True)
    df_Limu2 = pd.DataFrame(Limu2, columns=columnss)
    df_Limu2['Timestamp'] = pd.to_datetime(df_Limu2['Timestamp'])
    df_Limu2 = df_Limu2.sort_values(by='Timestamp')
    df_Limu2.set_index('Timestamp', inplace=True)

    # #Limu3-LEFT
    columns = ['Timestamp', 'elapsed(time)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu3 = pd.DataFrame(Limu3, columns=columns)
    
    # #Limu4-RIGHT
    df_Limu4 = pd.DataFrame(Limu4, columns=columns)
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

    linear_df3 = df_Limu3;
    linear_df4 = df_Limu4;

    # Low-pass filter parameters
    cutoff = 0.5  # Cutoff frequency in Hz (adjust as needed)
    fs = 100  # Sampling frequency in Hz (adjust based on your data)
    order = 4  # Filter order

    # Ensure the signals are aligned (same length); truncate if necessary
    min_length = min(len(df_Limu2), len(df_Limu3), len(df_Limu4))
    df_Limu2 = df_Limu2.iloc[:min_length]
    df_Limu3 = df_Limu3.iloc[:min_length]
    df_Limu4 = df_Limu4.iloc[:min_length]

    # Apply low-pass filter to the signal
    df_Limu2['filtered_signal'] = low_pass_filter(df_Limu2['W(number)'], cutoff, fs, order)

    # Apply moving average smoothing to the signal
    window_size = 20  # Adjust the window size as needed
    df_Limu2['smoothed_signal'] = df_Limu2['filtered_signal'].rolling(window=window_size, center=True).mean()

    peaks1, _ = find_peaks(df_Limu2['smoothed_signal'], height=None, distance=865)
    minima1, _ = find_peaks(-df_Limu2['smoothed_signal'], height=None, distance=865)

    # Combine peaks and minima, and sort them
    all_extrema = sorted(peaks1.tolist() + minima1.tolist())
    df_Limu2['first_differential'] = df_Limu2['smoothed_signal'].diff()
    total_duration_seconds = (df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds()
     # Initialize storage for aggregated metrics
    aggregated_metrics = {
        "total_steps": 0,
        "stride_times": [],
        "cadence": 0,
        "left_stance_times": [],
        "right_stance_times": [],
        "left_swing_times": [],
        "right_swing_times": [],
        "double_support_times": [],
        "single_support_times": [],
        "exercise_duration_seconds": total_duration_seconds
    }

    for i in range(len(all_extrema) - 1):
        start = all_extrema[i]
        end = all_extrema[i + 1]
        left_imu = df_Limu3[start:end]
        right_imu = df_Limu4[start:end]
         # Get segment metrics
        segment_metrics = gaitanalysis(left_imu, right_imu, False)

        # Aggregate metrics
        aggregated_metrics["total_steps"] += segment_metrics.get("total_metrics", {}).get("number_of_steps", 0)
        aggregated_metrics["stride_times"].extend(segment_metrics.get("total_metrics", {}).get("stride_times", []))
        aggregated_metrics["cadence"] += segment_metrics.get("total_metrics", {}).get("cadence", 0)
        aggregated_metrics["left_stance_times"].extend(segment_metrics.get("total_metrics", {}).get("left_stance_times", []))
        aggregated_metrics["right_stance_times"].extend(segment_metrics.get("total_metrics", {}).get("right_stance_times", []))
        aggregated_metrics["left_swing_times"].extend(segment_metrics.get("total_metrics", {}).get("left_swing_times", []))
        aggregated_metrics["right_swing_times"].extend(segment_metrics.get("total_metrics", {}).get("right_swing_times", []))
        aggregated_metrics["double_support_times"].extend(segment_metrics.get("total_metrics", {}).get("double_support_times", []))
        aggregated_metrics["single_support_times"].extend(segment_metrics.get("total_metrics", {}).get("single_support_times", []))

    total_elapsed_time = (end_time - start_time).total_seconds()
    cadence = (aggregated_metrics["total_steps"] / total_elapsed_time) * 60 if total_elapsed_time > 0 else 0

    aggregated_metrics.update({
        "cadence": cadence,
        "mean_stride_time": np.mean(aggregated_metrics["stride_times"]) if aggregated_metrics["stride_times"] else 0,
        "std_stride_time": np.std(aggregated_metrics["stride_times"]) if aggregated_metrics["stride_times"] else 0,
    })
    
    print(aggregated_metrics);

    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_ForwardWalking_metrics.txt"
    # Save the metrics to a file
    #save_metrics_to_txt(aggregated_metrics, filename)


    return json.dumps(aggregated_metrics, indent=4)

        
def save_metrics_to_txt(metrics, file_path):
        main_directory = "Walking"
        sub_directory = "Walking Metrics Data"

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
