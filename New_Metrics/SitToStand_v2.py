import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from scipy.signal import medfilt

# Function to apply a Butterworth low-pass filter
def low_pass_filter(signal, cutoff_freq, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Design the filter
    filtered_signal = filtfilt(b, a, signal)  # Apply the filter
    return filtered_signal

def moving_average_filter(signal, window_size=25):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Define slope constraints for each segment based on the initial ideal signal
def define_slope_constraints(initial_y):
    constraints = []
    epsilon = 0.01  # Small tolerance

    for i in range(1, len(initial_y)):
        if initial_y[i] > initial_y[i - 1]:  # Positive slope
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: params[i + len(initial_y)] - params[i + len(initial_y) - 1] - epsilon
            })
        elif initial_y[i] < initial_y[i - 1]:  # Negative slope
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: -(params[i + len(initial_y)] - params[i + len(initial_y) - 1]) - epsilon
            })
        else:  # Plateau
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: params[i + len(initial_y)] - params[i + len(initial_y) - 1] + epsilon
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: -(params[i + len(initial_y)] - params[i + len(initial_y) - 1]) + epsilon
            })

    return constraints


# Load and normalize the IMU signal data from CSV
def load_signal_from_csv(file_path, column_name='y'):
    data = pd.read_csv(file_path)
    signal = data[column_name].values
    plt.plot(signal)
    plt.show()
    signal, spike_indices = remove_wide_spikes_with_surrounding_avg(signal, kernel_size=151, threshold_factor=2)

    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return signal

# Estimate thresholds from autocorrelation
def estimate_autocorr_thresholds(signal):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr /= np.max(autocorr)
    peaks, _ = find_peaks(autocorr, height=0.2)
    if len(peaks) < 2:
        return None, None, autocorr
    distance_threshold = int(np.mean(np.diff(peaks)))
    height_threshold = 0.5 * np.max(autocorr[peaks[1:]])
    return distance_threshold, height_threshold, autocorr, peaks

# Modified Peak Detection with Dynamic Thresholds and Filtering by STD
def detect_repetitions_peaks(signal, distance_threshold, height_threshold):
    peaks, properties = find_peaks(signal, height=height_threshold, distance=distance_threshold)
    #print(peaks)
    peak_heights = properties['peak_heights']
    mean_height = np.mean(peak_heights)
    #print(mean_height)
    std_height = np.std(peak_heights)
    #print(std_height)
    filtered_peaks = [peak for peak, height in zip(peaks, peak_heights) if height >= mean_height - 2*std_height]
    #print(filtered_peaks)
    return np.array(filtered_peaks)

# Autocorrelation-based Detection
def detect_repetitions_autocorr(signal, distance_threshold=50, height_threshold=0.2):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr /= np.max(autocorr)
    peaks, _ = find_peaks(autocorr, height=height_threshold, distance=distance_threshold)
    return peaks, autocorr

# FFT-based Detection
def detect_repetitions_fft(signal, sampling_rate=100):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)[:N // 2]
    amplitude_spectrum = 2.0 / N * np.abs(yf[:N // 2])
    dominant_freq_idx = np.argmax(amplitude_spectrum)
    dominant_frequency = xf[dominant_freq_idx] if dominant_freq_idx > 0 else None
    fft_repetitions = []
    if dominant_frequency and dominant_frequency > 0:
        fft_interval = int(sampling_rate / dominant_frequency)
        fft_repetitions = np.arange(0, len(signal), fft_interval)
    return fft_repetitions, amplitude_spectrum, xf

# Calculate minima between peaks
def find_minima_between_peaks(signal, peaks):
    minima = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if start < end:
            min_idx = np.argmin(signal[start:end]) + start
            minima.append(min_idx)
    return np.array(minima)


def find_minima_between_peaks2(signal, peaks):
    minima = []
    
    # Find minimum before the first peak, if any
    if peaks[0] > 0:
        pre_first_peak_min_idx = np.argmin(signal[:peaks[0]])
        minima.append(pre_first_peak_min_idx)
    
    # Find minima between each pair of peaks
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if start < end:
            min_idx = np.argmin(signal[start:end]) + start
            minima.append(min_idx)
    
    return np.array(minima)


# Calculate x-axis distances from each peak to the nearest left and right minima
def calculate_distances(signal, peaks, minima):
    distances = []
    for peak in peaks:
        # Find the closest minima to the left
        left_minima = minima[minima < peak]
        left_distance = peak - left_minima[-1] if len(left_minima) > 0 else None
        
        # Find the closest minima to the right
        right_minima = minima[minima > peak]
        right_distance = right_minima[0] - peak if len(right_minima) > 0 else None
        
        distances.append((left_distance, right_distance))
    
    return distances

# Moving average function
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


# Identify outliers based on x-axis distances
def identify_outlier_peaks(distances, peaks, stf=3.0):
    """
    Identify outlier peaks based on the standard deviation filter (stf).
    A peak is considered an outlier if its distance to the nearest minimum
    deviates by more than `stf` standard deviations from the mean distance.
    """
    # Separate left and right distances, ignoring None values
    left_distances = [d[0] for d in distances if d[0] is not None]
    right_distances = [d[1] for d in distances if d[1] is not None]
    
    # Calculate mean and standard deviation for both left and right distances
    left_mean, left_std = np.mean(left_distances), np.std(left_distances)
    right_mean, right_std = np.mean(right_distances), np.std(right_distances)
    
    # Identify outliers
    outliers = []
    for i, (left, right) in enumerate(distances):
        left_outlier = left is not None and abs(left - left_mean) > stf * left_std
        right_outlier = right is not None and abs(right - right_mean) > stf * right_std
        if left_outlier or right_outlier:
            outliers.append(peaks[i])
    
    return np.array(outliers)

# Plotting function
def plot_repetitions(signal, peak_reps, autocorr_reps, autocorr, minima, outliers):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot original signal with repetition points
    axs[0].plot(signal, label="IMU Signal")
    for rep in peak_reps:
        axs[0].axvline(x=rep, color='orange', linestyle='--', label='Maxima Detection' if rep   == peak_reps[0] else "")
    # Plot minima
    for min_idx in minima:
        axs[0].axvline(x=min_idx, color='green', linestyle='--', label='Minima Detection' if min_idx   == minima[0] else "")
    # Plot outliers
    for outlier in outliers:
        axs[0].axvline(x=outlier, color='blue', linestyle='--', label='Outliers Detection' if outlier   == outliers[0] else "")    


    axs[0].set_title("IMU Signal with Detected Repetitions")
    axs[0].legend()
    axs[0].grid(True)

    # Plot autocorrelation
    axs[1].plot(autocorr, label="Autocorrelation")
    axs[1].plot(autocorr_reps, autocorr[autocorr_reps], "x", label="Autocorr Peaks")
    axs[1].set_title("Autocorrelation and Peaks")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Plot the signal, peaks, and minima
def plot_signal_with_peaks_minima(signal, peaks, minima, outliers, showplot):
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="IMU Signal", color='blue')
    # Check if peaks array is not empty
    if peaks is not None and len(peaks) > 0:
        plt.plot(peaks, signal[peaks], 'o', label="Final Peaks", color='orange')
    
    # Check if minima array is not empty
    if minima is not None and len(minima) > 0:
        plt.plot(minima, signal[minima], 'x', label="Final Minima", color='green')
    
    # Check if outliers array is not empty
    if outliers is not None and len(outliers) > 0:
        plt.plot(outliers, signal[outliers], 'ro', label="Outlier Peaks")
    
    plt.legend()
    plt.title("IMU Signal with Final Peaks and Minima")
    plt.grid(True)
    plt.show()

# Remove outliers from peaks
def remove_outliers(peaks, outliers):
    return np.array([p for p in peaks if p not in outliers])


def create_ideal_signal(segment_length, critical_points_x, critical_points_y):
    """
    Scales the x-coordinates of the critical points in the ideal signal to fit a given segment length and offsets to zero.

    Parameters:
    - segment_length: Length of the real signal segment to match.
    - critical_points_x: Original x-coordinates of critical points in the ideal signal.
    - critical_points_y: Original y-coordinates of critical points in the ideal signal.

    Returns:
    - offset_x: Scaled and zero-offset x-coordinates matching the segment length.
    - stretched_y: Original y-coordinates (unchanged).
    """

    # Calculate the scaling factor for the x-axis to match the segment length
    original_length = critical_points_x[-1] - critical_points_x[0]
    scale_factor = segment_length / original_length if original_length > 0 else 1

    # Apply scaling to the x-coordinates
    stretched_x = critical_points_x * scale_factor

    # Offset the scaled x-coordinates to start from zero
    offset_x = stretched_x - stretched_x[0]

    # Clip and round to integer values within the segment length bounds
    offset_x = np.clip(offset_x, 0, segment_length - 1).round().astype(int)

    # y-coordinates remain unchanged
    stretched_y = critical_points_y

    return offset_x, stretched_y


# Step 3: Create a function to fit piecewise linear segments
def piecewise_linear_model(x_points, y_points, x_values):
    # Interpolate the y values at the given x_values using the piecewise linear segments
    return np.interp(x_values, x_points, y_points)

# Step 4: Define the objective function to minimize the error between the real signal and the model
def objective_function(params, real_signal, x_values):
    num_critical_points = len(params) // 2
    critical_points_x = params[:num_critical_points]
    critical_points_y = params[num_critical_points:]
    
    # Ensure that the x critical points remain ordered
    if np.any(np.diff(critical_points_x) <= 0):
        return np.inf  # Invalid configuration, penalize heavily

    # Create the piecewise linear model with the given parameters
    fitted_signal = piecewise_linear_model(critical_points_x, critical_points_y, x_values)
    
    # Calculate the sum of squared differences (error)
    error = np.sum((fitted_signal - real_signal) ** 2)
    return error

# Step 5: Optimize the critical points to fit the real signal
def optimize_critical_points(real_signal, x_values, critical_points_x, critical_points_y):
    # Reset ideal signal x-points to start from 0 and stretch to the length of the real signal segment
    segment_length = len(real_signal)
    initial_x = np.linspace(0, segment_length - 1, len(critical_points_x))  # Reset and scale x-points
    initial_y = critical_points_y  # Use provided y-points directly
    
    # Flatten initial critical points into a single array for optimization
    initial_params = np.concatenate([initial_x, initial_y])
    
    # Define bounds for optimization
    bounds = [(0, segment_length)] * len(initial_x) + [(np.min(real_signal), np.max(real_signal))] * len(initial_y)
    
    # Optimize the critical points using the objective function
    result = minimize(objective_function, initial_params, args=(real_signal, x_values), bounds=bounds, method='SLSQP')
    
    # Extract optimized x and y values for critical points
    optimized_params = result.x
    optimized_x = optimized_params[:len(initial_x)]
    optimized_y = optimized_params[len(initial_x):]
    
    return optimized_x, optimized_y



# Adjust the optimize_critical_points function to include the new slope constraints
def optimize_critical_points_with_constraints(real_signal, initial_x, initial_y, x_values):
    initial_params = np.concatenate([initial_x, initial_y])
    
    # Define bounds as in your previous code
    bounds = [(0, len(x_values))] * len(initial_x) + [(np.min(real_signal), np.max(real_signal))] * len(initial_y)

    # Add slope constraints to the optimizer
    constraints = define_slope_constraints(initial_y)

    # Run the optimization with constraints
    result = minimize(
        objective_function, initial_params,
        args=(real_signal, x_values),
        bounds=bounds, constraints=constraints,
        method='SLSQP'
    )

    optimized_params = result.x
    optimized_x = optimized_params[:len(initial_x)]
    optimized_y = optimized_params[len(initial_x):]
    
    return optimized_x, optimized_y

def process_subsegments(signal, minima, maxima, critical_points_x, critical_points_y, showplot):
    all_optimized_x = []
    all_optimized_y = []

    # Detect all movements using groups of minima and maxima
    num_movements = (len(minima) - 1) // 3  # Each movement consists of three minima and two maxima
    movement_subsegments = []
    ideal_pairs = []

    # Define ideal pairs dynamically for each detected movement
    for movement_idx in range(num_movements):
        start_min = movement_idx * 3
        ideal_start = movement_idx * 4
        
        # Generate subsegments and corresponding ideal pairs for the current movement
        movement_subsegments.append([
            (minima[start_min], maxima[start_min], minima[start_min + 1]),
            (minima[start_min + 1], maxima[start_min + 1], minima[start_min + 2]),
            (minima[start_min + 2], maxima[start_min + 2], minima[start_min + 3])
        ])

        # Corresponding ideal signal pairs for each subsegment
        ideal_pairs.append([
            (0, 5),  # Ideal points for first subsegment (minimum1-max1-minimum2)
            (4, 9),  # Ideal points for second subsegment (minimum2-max2-minimum3)
            (8, 13)  # Ideal points for third subsegment (minimum3-max3-minimum4)
        ])

    # Process each movement's subsegments and ideal pairs
    for movement_idx, subsegments in enumerate(movement_subsegments):
        optimized_segments_x = []
        optimized_segments_y = []

        for (start, peak, end), (ideal_start, ideal_end) in zip(subsegments, ideal_pairs[movement_idx]):
            # Extract the segment of the real signal between minima and maxima
            segment = signal[start:end+1]
            x_values = np.arange(start, end+1)  # Global x-values for the segment
            
            # Extract the corresponding portion of the ideal signal
            ideal_x = critical_points_x[ideal_start:ideal_end+1]
            ideal_y = critical_points_y[ideal_start:ideal_end+1]
            
            # Run the optimization for this subsegment
            optimized_x, optimized_y = plot_segment_with_critical_points(segment, x_values, ideal_x, ideal_y, showplot)
            
            # Offset optimized x values back to the original coordinate space
            offset_optimized_x = optimized_x + start
            
            # Store each subsegment as a list of arrays (for later overlap handling)
            optimized_segments_x.append(offset_optimized_x)
            optimized_segments_y.append(optimized_y)
        
        # Combine results for this movement with overlap handling
        movement_optimized_x = []
        movement_optimized_y = []

        for i in range(len(optimized_segments_x) - 1):
            current_x, current_y = optimized_segments_x[i], optimized_segments_y[i]
            next_x, next_y = optimized_segments_x[i + 1], optimized_segments_y[i + 1]
            
            # Store non-overlapping points
            segment_x = list(current_x[:-2])
            segment_y = list(current_y[:-2])

            # Calculate and add averaged overlapping points
            avg_x = (current_x[-2:] + next_x[:2]) / 2
            avg_y = (current_y[-2:] + next_y[:2]) / 2

            # Append averaged overlap points
            segment_x.extend(avg_x)
            segment_y.extend(avg_y)

            # Append finalized segment
            movement_optimized_x.append(segment_x)
            movement_optimized_y.append(segment_y)

        # Add remaining points from the last subsegment of the movement
        last_x = list(optimized_segments_x[-1][2:])
        last_y = list(optimized_segments_y[-1][2:])
        movement_optimized_x.append(last_x)
        movement_optimized_y.append(last_y)

        # Flatten each movement segment to list of lists
        all_optimized_x.append([item for sublist in movement_optimized_x for item in sublist])
        all_optimized_y.append([item for sublist in movement_optimized_y for item in sublist])

    return all_optimized_x, all_optimized_y












import matplotlib.pyplot as plt

def plot_first_segment(signal, minima, maxima, critical_points_x, critical_points_y, showplot=True):
    # Define the first movement’s subsegments and corresponding ideal pairs
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


    subsegments = [
        (minima[0], maxima[0], minima[1]),
        (minima[1], maxima[1], minima[2]),
        (minima[2], maxima[2], minima[3])
    ]
    
    ideal_pairs = [
        (0, 5),  # Ideal points for first subsegment (minimum1-max1-minimum2)
        (4, 9),  # Ideal points for second subsegment (minimum2-max2-minimum3)
        (8, 13)  # Ideal points for third subsegment (minimum3-max3-minimum4)
    ]
    
    optimized_segments_x = []
    optimized_segments_y = []
    
    # Process each subsegment, apply optimization, and gather results
    for (start, peak, end), (ideal_start, ideal_end) in zip(subsegments, ideal_pairs):
        # Extract the segment of the real signal and the ideal points
        segment = signal[start:end+1]
        segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))

        x_values = np.arange(start, end+1)
        
        ideal_x = critical_points_x[ideal_start:ideal_end+1]
        ideal_y = critical_points_y[ideal_start:ideal_end+1]
        
        # Optimize for this subsegment
        optimized_x, optimized_y = plot_segment_with_critical_points(segment, x_values, ideal_x, ideal_y, showplot=False)
        
        # Offset the optimized x values back to the original space
        optimized_segments_x.append(optimized_x + start)
        optimized_segments_y.append(optimized_y)
    
    # Merge the optimized subsegments
    merged_optimized_x = []
    merged_optimized_y = []
    for i in range(len(optimized_segments_x) - 1):
        current_x, current_y = optimized_segments_x[i], optimized_segments_y[i]
        next_x, next_y = optimized_segments_x[i + 1], optimized_segments_y[i + 1]
        
        # Add non-overlapping points of the current subsegment
        merged_optimized_x.extend(current_x[:-2])
        merged_optimized_y.extend(current_y[:-2])
        
        # Average the overlapping points
        avg_x = (current_x[-2:] + next_x[:2]) / 2
        avg_y = (current_y[-2:] + next_y[:2]) / 2
        
        merged_optimized_x.extend(avg_x)
        merged_optimized_y.extend(avg_y)

        # Add the remaining points from the last segment
        if i == len(optimized_segments_x) - 2:
            merged_optimized_x.extend(next_x[2:])
            merged_optimized_y.extend(next_y[2:])
    
    # Convert merged lists to arrays for plotting
    merged_optimized_x = np.array(merged_optimized_x)
    merged_optimized_y = np.array(merged_optimized_y)
    
    # Plot the original segment, ideal signal, and optimized merged signal
    plt.figure(figsize=(12, 6))
    
    # Plot the original signal segment
    plt.plot(np.arange(subsegments[0][0], subsegments[-1][2] + 1), 
             signal[subsegments[0][0]:subsegments[-1][2] + 1], 
             label="Original Segment", color='blue')
    
    # Plot the combined ideal signal
    combined_ideal_x = np.concatenate([critical_points_x[ideal_pairs[0][0]:ideal_pairs[-1][1] + 1]])
    combined_ideal_y = np.concatenate([critical_points_y[ideal_pairs[0][0]:ideal_pairs[-1][1] + 1]])
    plt.plot(combined_ideal_x + subsegments[0][0], combined_ideal_y, 
             label="Combined Ideal Signal", linestyle="--", color='green')
    
    # Plot the merged optimized signal
    plt.plot(merged_optimized_x, merged_optimized_y, 
             label="Merged Optimized Signal", linestyle="--", color='red')
    
    # Customize and show plot
    plt.title("Comparison of Original Segment, Ideal Signal, and Merged Optimized Signal")
    plt.legend()
    plt.grid(True)
    
    if showplot:
        plt.show()
    
    return merged_optimized_x, merged_optimized_y









def classify_segment(y1, y2, epsilon=0.1):
    """Classify the segment between two critical points based on their y-values."""
    if abs(y2 - y1) < epsilon:
        return 'plateau'
    elif y2 > y1:
        return 'increase'
    else:
        return 'decrease'

def detect_and_correct_violations(optimized_x, optimized_y):
    """Ensure that no two consecutive segments are of the same type (increase, plateau, decrease)."""
    classifications = []
    num_points = len(optimized_x)
    
    # Classify each segment between critical points
    for i in range(num_points - 1):
        segment_type = classify_segment(optimized_y[i], optimized_y[i + 1])
        classifications.append(segment_type)
    
    # List to store the corrected critical points
    corrected_x = [optimized_x[0]]
    corrected_y = [optimized_y[0]]
    
    # Iterate over classifications and skip violations
    for i in range(1, len(classifications)):
        if classifications[i] != classifications[i - 1]:  # No violation, keep the point
            corrected_x.append(optimized_x[i])
            corrected_y.append(optimized_y[i])
        else:
            # If violation (consecutive same type), we skip the current point
            print(f"Skipping point {i + 1} to avoid consecutive {classifications[i]}")

    # Add the last point (always keep it)
    corrected_x.append(optimized_x[-1])
    corrected_y.append(optimized_y[-1])
    
    return np.array(corrected_x), np.array(corrected_y)

def postprocess_and_plot(optimized_x, optimized_y, real_signal, x_values):
    # Correct the critical points by skipping invalid transitions
    corrected_x, corrected_y = detect_and_correct_violations(optimized_x, optimized_y)
    
    # Plot the corrected signal
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, real_signal, label="Real Signal", color='blue')
    optimized_signal = np.interp(np.arange(len(real_signal)), corrected_x, corrected_y)
    plt.plot(optimized_signal, label="Corrected Ideal Signal", linestyle="--", color='red')
    
    # Mark the corrected critical points in red
    for i, (x, y) in enumerate(zip(corrected_x, corrected_y)):
        plt.plot(x, y, 'ro')
        plt.text(x, y, f'{i+1}', color="red", fontsize=12)
    
    plt.legend()
    plt.title("Corrected Ideal Signal with Numbered Critical Points")
    plt.grid(True)
    plt.show()

def process_segments(signal, minima, critical_points_x, critical_points_y, showplot, minima_per_repetition=4):
    all_optimized_x = []
    all_optimized_y = []
    
    # Iterate over minima in steps of minima_per_repetition to capture each full repetition
    for i in range(0, len(minima) - (minima_per_repetition - 1), minima_per_repetition):
        # Extract segment between the first and last minima in the group
        start, end = minima[i], minima[i + minima_per_repetition - 1]
        segment = signal[start:end + 1]
        x_values = np.arange(len(segment))

        # Plot the segment with the optimized critical points
        optimized_x, optimized_y = plot_segment_with_critical_points(segment, x_values, critical_points_x, critical_points_y, showplot)

        # Append optimized points for each segment
        all_optimized_x.append(optimized_x)
        all_optimized_y.append(optimized_y)

    return np.array(all_optimized_x), np.array(all_optimized_y)



def plot_segment_with_critical_points(real_signal, x_values, critical_points_x, critical_points_y, showplot):
    real_signal = (real_signal - np.min(real_signal)) / (np.max(real_signal) - np.min(real_signal))

    # Generate x values corresponding to the extracted real signal
    x_values = np.arange(len(real_signal))

    # Create the initial ideal signal (critical points)
    #print('len(real_signal) = ', len(real_signal))

    initial_x, initial_y = create_ideal_signal(len(real_signal), critical_points_x, critical_points_y)

    # Step 7: Optimize the critical points to fit the real signal
    #optimized_x, optimized_y = optimize_critical_points(real_signal, x_values)
    optimized_x, optimized_y = optimize_critical_points_with_constraints(real_signal, initial_x, initial_y, x_values)
    #optimized_x, optimized_y = split_and_optimize_signal(real_signal, initial_x, initial_y, x_values)
    
    # Step 8: Visualize the results
    plt.figure(figsize=(10, 6))

    # Plot the real signal (from 'y' data in the CSV)
    plt.plot(x_values, real_signal, label="Real Signal (y from CSV)", color='blue')

    # Plot the initial ideal signal (before optimization)
    initial_fitted_signal = piecewise_linear_model(initial_x, initial_y, x_values)
    plt.plot(x_values, initial_fitted_signal, label="Initial Ideal Signal", linestyle="--", color='green')

    # Plot the optimized ideal signal (after optimization)
    optimized_fitted_signal = piecewise_linear_model(optimized_x, optimized_y, x_values)
    plt.plot(x_values, optimized_fitted_signal, label="Optimized Ideal Signal", linestyle="--", color='red')

    # Mark the initial critical points in blue and add numbering
    for i, (x, y) in enumerate(zip(initial_x, initial_y), 1):
        plt.plot(x, y, 'bo')
        plt.text(x, y, str(i), color="violet", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Mark the optimized critical points in red and add numbering
    for i, (x, y) in enumerate(zip(optimized_x, optimized_y), 1):
        plt.plot(x, y, 'ro')
        plt.text(x, y, str(i), color="red", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    if (showplot):
        plt.legend()
        plt.title("Real Signal (CSV) vs Optimized Ideal Signal with Critical Points")
        plt.grid(True)
        plt.show()

    # Print the optimized critical points
    #print("Optimized critical points (x):", optimized_x)
    #print("Optimized critical points (y):", optimized_y)
    # Return the optimized critical points
    return optimized_x, optimized_y


# Function to detect and replace wide spikes with the surrounding average
def remove_wide_spikes_with_surrounding_avg(signal, kernel_size=151, threshold_factor=2):
    # Step 1: Apply median filter to create a smoothed version of the signal
    filtered_signal = medfilt(signal, kernel_size=kernel_size)

    # Step 2: Calculate the difference between original and smoothed signals
    deviation = np.abs(signal - filtered_signal)

    # Set a threshold to detect large deviations (spikes)
    deviation_threshold = threshold_factor * np.std(deviation)
    spike_indices = np.where(deviation > deviation_threshold)[0]

    # Step 3: Process contiguous spike segments
    processed_signal = signal.copy()
    spike_start = None
    for i in range(len(spike_indices) - 1):
        # Detect the start and end of a spike segment
        if spike_start is None:
            spike_start = spike_indices[i]
        
        # If there’s a gap in spike indices, process the previous segment
        if spike_indices[i+1] - spike_indices[i] > 1:
            spike_end = spike_indices[i]

            # Replace the segment with the average of points around it
            left_value = processed_signal[spike_start - 1] if spike_start > 0 else processed_signal[spike_end + 1]
            right_value = processed_signal[spike_end + 1] if spike_end + 1 < len(signal) else left_value
            surrounding_avg = (left_value + right_value) / 2

            # Replace the values in the segment with the surrounding average
            processed_signal[spike_start:spike_end + 1] = surrounding_avg
            
            # Reset spike_start for the next segment
            spike_start = None

    # Handle the last segment if it ends at the end of spike_indices
    if spike_start is not None:
        spike_end = spike_indices[-1]
        left_value = processed_signal[spike_start - 1] if spike_start > 0 else processed_signal[spike_end + 1]
        right_value = processed_signal[spike_end + 1] if spike_end + 1 < len(signal) else left_value
        surrounding_avg = (left_value + right_value) / 2
        processed_signal[spike_start:spike_end + 1] = surrounding_avg

    return processed_signal, spike_indices


import numpy as np


def process_movements(all_optimized_x, all_optimized_y, movements):
    movement_stats = {}

    # Loop through each movement to calculate aggregate metrics
    for movement in movements:
        name = movement["name"]
        start_idx, end_idx = movement["start_idx"], movement["end_idx"]
        
        # Collect data across all repetitions for this movement
        movement_segments_y = []
        movement_durations = []

        for segment_idx in range(len(all_optimized_x)):
            optimized_x = all_optimized_x[segment_idx]
            optimized_y = all_optimized_y[segment_idx]
            
            # Ensure indices are within bounds for each segment
            if start_idx >= len(optimized_x) or end_idx >= len(optimized_x):
                #print(f"Warning: Movement '{name}' index out of bounds for repetition {segment_idx}. Skipping.")
                continue
            
            # Extract the y values and duration for the current movement
            segment_y = optimized_y[start_idx:end_idx + 1]
            duration = optimized_x[end_idx] - optimized_x[start_idx]
            
            # Append to the lists for this movement across repetitions
            movement_segments_y.append(segment_y)
            movement_durations.append(duration)
        
        # Convert lists to numpy arrays for easier calculations
        movement_segments_y = np.array(movement_segments_y)
        movement_durations = np.array(movement_durations)
        
        # Calculate overall statistics across all repetitions
        movement_metrics = {
            "mean": np.mean(movement_segments_y),
            "std": np.std(movement_segments_y),
            "max": np.max(movement_segments_y),
            "min": np.min(movement_segments_y),
            "mean_duration": np.mean(movement_durations),
            "std_duration": np.std(movement_durations)
        }

        # Calculate symmetry by comparing the first and second halves of repetitions
        half_reps = len(movement_segments_y) // 2
        if half_reps > 0:
            first_half = movement_segments_y[:half_reps]
            second_half = movement_segments_y[half_reps:]

            symmetry = {
                "mean_diff": abs(np.mean(first_half) - np.mean(second_half)),
                "std_diff": abs(np.std(first_half) - np.std(second_half)),
                "max_diff": abs(np.max(first_half) - np.max(second_half)),
                "min_diff": abs(np.min(first_half) - np.min(second_half)),
                "mean_duration_diff": abs(np.mean(movement_durations[:half_reps]) - np.mean(movement_durations[half_reps:]))
            }
        else:
            symmetry = {}

        # Store the aggregate metrics and symmetry for this movement
        movement_stats[name] = {
            "metrics": movement_metrics,
            "symmetry": symmetry
        }
    
    return movement_stats

# Calculate aggregated metrics across all repetitions and evaluate symmetry
def calculate_movement_metrics(movements, all_reps_movement_results):
    movement_data = {}
    num_repetitions = len(all_reps_movement_results)

    # Process each movement independently across all repetitions
    for movement in movements:
        name = movement["name"]
        start_idx, end_idx = movement["start_idx"], movement["end_idx"]

        # Gather movement metrics across repetitions
        movement_segments = []
        for rep in all_reps_movement_results:
            movement_metrics = rep[start_idx:end_idx + 1]
            movement_segments.append(movement_metrics)

        # Convert movement segments to an array to facilitate statistics
        movement_segments = np.array(movement_segments)

        # Calculate overall metrics for this movement
        movement_metrics = {
            "mean": np.mean(movement_segments),
            "std": np.std(movement_segments),
            "max": np.max(movement_segments),
            "min": np.min(movement_segments),
            "num_repetitions": num_repetitions,
        }

        # Symmetry analysis: Divide repetitions in half and compare metrics
        half_reps = num_repetitions // 2
        first_half_metrics = movement_segments[:half_reps]
        second_half_metrics = movement_segments[half_reps:num_repetitions]

        symmetry = {
            "mean_diff": abs(np.mean(first_half_metrics) - np.mean(second_half_metrics)),
            "std_diff": abs(np.std(first_half_metrics) - np.std(second_half_metrics)),
            "max_diff": abs(np.max(first_half_metrics) - np.max(second_half_metrics)),
            "min_diff": abs(np.min(first_half_metrics) - np.min(second_half_metrics)),
        }

        # Store results for each movement
        movement_data[name] = {
            "metrics": movement_metrics,
            "symmetry": symmetry
        }

    return movement_data

# Example usage
# movements = [...]  # Define your movement list here as before
# all_optimized_x, all_optimized_y = process_subsegments(signal, minima, maxima, critical_points_x, critical_points_y, showplot)
# all_reps_movement_results = process_movements(all_optimized_x, all_optimized_y, movements)
# movement_data = calculate_movement_metrics(movements, all_reps_movement_results)



# Step 3: Generate JSON Output
def generate_json_output(movements, all_optimized_x, all_optimized_y):
    movement_data = calculate_movement_metrics(movements, all_optimized_x, all_optimized_y)
    json_output = json.dumps(movement_data, indent=4)
    return json_output

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
            #print("DF")
            #print(df)
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
    imu_data_lists = [Limu1, Limu2, Limu3, Limu4]
    processed_dataframes, c = process_imu_data(imu_data_lists, 50, True)

    Limu1 = processed_dataframes[0]
    
    if (c >= 2):
        Limu2 = processed_dataframes[1]
    if (c >= 3):
        Limu3 = processed_dataframes[2]
    if (c >= 4):
        Limu4 = processed_dataframes[3]

    if(len(Limu1)>0 and len(Limu2) > 0):
        returnedJson = getMetricsSittingNew01(Limu1, Limu2, False) 
        return returnedJson

def getMetricsSittingNew01(Limu1, Limu2, plotdiagrams=False):
    showplot = plotdiagrams
    
    # Process Limu1 data
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['elapsed(time)'] = pd.to_datetime(df_Limu1['elapsed(time)'], unit='ms')
    df_Limu1 = df_Limu1.sort_values(by='elapsed(time)')
    df_Limu1.set_index('elapsed(time)', inplace=True)
    
    # Process Limu2 data (desired signal)
    df_Limu2 = pd.DataFrame(Limu2, columns=columns)
    df_Limu2['elapsed(time)'] = pd.to_datetime(df_Limu2['elapsed(time)'], unit='ms')
    df_Limu2 = df_Limu2.sort_values(by='elapsed(time)')
    df_Limu2.set_index('elapsed(time)', inplace=True)
    
    
    y_signal = df_Limu2['W(number)']
    timestamps = pd.to_numeric(df_Limu2.index) / 1e3  # Convert to seconds
    timestamps = pd.Series(timestamps)
    
    # Smooth signal to remove noise
    window_size = 50
    smoothed_signal = y_signal.rolling(window=window_size, center=True).mean()
    
    # Detect valleys (bend phase) and peaks (stand phase)
    valleys, _ = find_peaks(-smoothed_signal, prominence=0.02)  # Bend phase (low W values)
    peaks, _ = find_peaks(smoothed_signal, prominence=0.02)  # Stand phase (high W values)
    
    final_movements = []
    stand_durations = []
    i = 0
    
    while i < len(valleys) - 1:                        
        try:
            start_bend = valleys[i]  # Start of bending
            mid_bend = next(idx for idx in valleys if idx > start_bend)  # Midway point before standing
            stand_peak = next(idx for idx in peaks if idx > mid_bend)  # Peak for standing up
            end_sit = next(idx for idx in valleys if idx > stand_peak)  # Returning to seat
            
            # Validate as a full sit-to-stand repetition
            movement_range = smoothed_signal.iloc[stand_peak] - smoothed_signal.iloc[start_bend]
            if movement_range > 0.05:
                final_movements.append({
                    "start_time": timestamps.iloc[start_bend],
                    "end_time": timestamps.iloc[end_sit],
                    "range_degrees": movement_range,
                    "duration": timestamps.iloc[end_sit] - timestamps.iloc[start_bend],
                    "stand_time": timestamps.iloc[end_sit] - timestamps.iloc[stand_peak]  # Time standing
                })
                stand_durations.append(timestamps.iloc[end_sit] - timestamps.iloc[stand_peak])
            
            i = valleys.tolist().index(end_sit)
        except StopIteration:
            break
        
        i += 1
    
    if showplot:
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, smoothed_signal, label='Smoothed Signal', linewidth=2)
        plt.scatter(timestamps.iloc[peaks], smoothed_signal.iloc[peaks], color='red', label='Stand (Peaks)', zorder=5)
        plt.scatter(timestamps.iloc[valleys], smoothed_signal.iloc[valleys], color='blue', label='Bend/Sit (Valleys)', zorder=5)
        
        for move in final_movements:
            plt.axvspan(move["start_time"], move["end_time"], color='yellow', alpha=0.3, label='Detected Movement')
        
        plt.title("Sit to Stand Repetition Detection")
        plt.xlabel("Time (mseconds)")
        plt.ylabel("Amplitude (W)")
        plt.legend()
        plt.grid()
        plt.show()
    
    # Compute metrics
    if final_movements:
        durations = [m['duration'] for m in final_movements]
        ranges = [m['range_degrees'] for m in final_movements]
        exercise_duration =(df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds()
        
        metrics_data = {
            "total_metrics" :{
            "number_of_movements": len(final_movements),
            "pace_movements_per_second": len(final_movements) / exercise_duration,
            "mean_range_degrees": np.mean(ranges),
            "std_range_degrees": np.std(ranges),
            "mean_duration_seconds": np.mean(durations)/1000000.0,
            "std_duration_seconds": np.std(durations)/1000000.0,
            "mean_stand_time_seconds": np.mean(stand_durations)/1000000.0,
            "std_stand_time_seconds": np.std(stand_durations)/1000000.0,
            "exercise_duration_seconds": exercise_duration
        }
        }
    else:
        metrics_data = {
            "total_metrics" :{
            "number_of_movements": 0,
            "pace_movements_per_second": 0,
            "mean_range_degrees": 0,
            "std_range_degrees": 0,
            "mean_duration_seconds": 0,
            "std_duration_seconds": 0,
            "mean_stand_time_seconds": 0,
            "std_stand_time_seconds": 0,
            "exercise_duration_seconds": 0
        }
        }
    
    print(metrics_data)
    
    # Save metrics to a file
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_Sit_To_Stand_metrics.txt"
    with open(filename, 'w') as file:
        json.dump(metrics_data, file, indent=4)
    
    return json.dumps(metrics_data, indent=4)
                                                                                                                                             
    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Sitting Metrics Data"
    sub_directory = "SitToStand Metrics Data"

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
    