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


def process_imu_data(imu_data_lists, fs, plotdiagrams=False):
    # Ensure lists are not empty and convert to DataFrames
    dataframes = []
    c = 0;
    for imu_data in imu_data_lists:
        if imu_data:
            columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y (number)', 'Z (number)']
            df = pd.DataFrame(imu_data, columns=columns)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df.sort_values(by='Timestamp')
            df.set_index('Timestamp', inplace=True)
            dataframes.append(df)
            c = c + 1

    if not dataframes:
        return None  # No data to process

    # Find the common time range
    max_start_time = max(df.index[0] for df in dataframes)
    min_end_time = min(df.index[-1] for df in dataframes)

    # Filter dataframes to the common time range
    dataframes = [df[max_start_time:min_end_time] for df in dataframes]

    # Determine the maximum number of samples across lists
    max_samples = max(len(df) for df in dataframes)

    # Resample and interpolate dataframes to have the same number of samples
    resampled_dataframes = []
    for df in dataframes:
        df_resampled = df.resample(f'{1000//fs}ms').mean()  # Resampling to match the sampling frequency
        df_interpolated = df_resampled.interpolate(method='time')
        df_interpolated = df_interpolated.dropna().head(max_samples)
        resampled_dataframes.append(df_interpolated)

    if plotdiagrams:
        for idx, df in enumerate(resampled_dataframes):
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['W(number)'], label='W')
            plt.plot(df.index, df['X(number)'], label='X')
            plt.plot(df.index, df['Y (number)'], label='Y')
            plt.plot(df.index, df['Z (number)'], label='Z')
            plt.xlabel('Timestamp')
            plt.ylabel('Quaternion Components')
            plt.title(f'IMU {idx+1} Quaternion Components (W, X, Y, Z) over Time')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'quaternion_components_plot_{idx+1}.png')
            # plt.show()

    # Convert the processed DataFrames back to lists
    resampled_lists = [df.reset_index().values.tolist() for df in resampled_dataframes]

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

    imu_data_lists = [Limu1, Limu2, Limu3, Limu4]
    processed_dataframes, c = process_imu_data(imu_data_lists, 50, True)
    Limu1 = processed_dataframes[0]
    if (c >= 2):
        Limu2 = processed_dataframes[1]
    if (c >= 3):
        Limu3 = processed_dataframes[2]
    if (c >= 4):
        Limu4 = processed_dataframes[3]

    if(len(Limu1) > 0):
        returnedJson = getMetricsSittingOld02(Limu1, False) 
        return returnedJson

def getMetricsSittingOld02(Limu1, plotdiagrams):
    import numpy as np
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    from scipy.signal import find_peaks

    # ---------------- helpers ----------------
    def _resample_1d(arr, target_len):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return np.zeros(target_len, dtype=float)
        if len(arr) == 1:
            return np.full(target_len, arr[0], dtype=float)
        x_old = np.linspace(0.0, 1.0, len(arr))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, arr)

    # ---------------- df ----------------
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y(number)', 'Z(number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['Timestamp'] = pd.to_datetime(df_Limu1['Timestamp'])
    df_Limu1 = df_Limu1.sort_values(by='Timestamp')
    df_Limu1.set_index('Timestamp', inplace=True)

    # ---------------- quaternion -> euler (deg) ----------------
    quaternions = df_Limu1[['X(number)', 'Y(number)', 'Z(number)', 'W(number)']].to_numpy()
    rotations = R.from_quat(quaternions)
    euler_angles_degrees = rotations.as_euler('xyz', degrees=True)
    euler_df_degrees = pd.DataFrame(
        euler_angles_degrees,
        columns=['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)'],
        index=df_Limu1.index
    )

    if plotdiagrams:
        plt.figure(figsize=(12, 8))
        plt.plot(euler_df_degrees.index, euler_df_degrees['Roll (degrees)'], label='Roll', linewidth=1)
        plt.plot(euler_df_degrees.index, euler_df_degrees['Pitch (degrees)'], label='Pitch', linewidth=1)
        plt.plot(euler_df_degrees.index, euler_df_degrees['Yaw (degrees)'], label='Yaw', linewidth=1)
        plt.xlabel('Timestamp')
        plt.ylabel('Euler Angles (degrees)')
        plt.title('Euler Angles (Roll, Pitch, Yaw) over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ---------------- filter pitch (head up/down) ----------------
    fs = 50
    cutoff = 0.75
    pitch = euler_df_degrees['Pitch (degrees)'].to_numpy()
    pitch_filtered = butter_lowpass_filter(pitch, cutoff, fs, order=5)

    if plotdiagrams:
        plt.figure(figsize=(12, 6))
        plt.plot(euler_df_degrees.index, pitch, label='Original Pitch', linewidth=1, alpha=0.5)
        plt.plot(euler_df_degrees.index, pitch_filtered, label='Filtered Pitch', linewidth=2)
        plt.xlabel('Timestamp')
        plt.ylabel('Pitch (degrees)')
        plt.title('Pitch Signal Filtering')
        plt.legend()
        plt.show()

    # ---------------- detect peaks/valleys ----------------
    peaks, _ = find_peaks(pitch_filtered)
    valleys, _ = find_peaks(-pitch_filtered)

    if len(peaks) == 0 or len(valleys) == 0:
        total_duration_seconds = float((df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds())
        return json.dumps({
            "total_metrics": {
                "number_of_movements": 0,
                "pace_movements_per_second": 0.0,
                "mean_range_degrees": 0.0,
                "std_range_degrees": 0.0,
                "mean_duration_seconds": 0.0,
                "std_duration_seconds": 0.0,
                "exercise_duration_seconds": total_duration_seconds
            },
            "velocity_curves_deg_per_s": [],
            "velocity_mean_curve_deg_per_s": [],
            "velocity_std_curve_deg_per_s": [],
            "peak_velocities_deg_per_s": [],
            "rom_peaks": []
        }, indent=4)

    # enforce valley -> peak pairing
    if valleys[0] > peaks[0]:
        peaks = peaks[1:]
    if len(peaks) > 0 and len(valleys) > 0 and peaks[-1] < valleys[-1]:
        valleys = valleys[:-1]

    movement_pairs = [(valleys[i], peaks[i]) for i in range(min(len(peaks), len(valleys)))]

    if plotdiagrams:
        plt.figure(figsize=(12, 6))
        plt.plot(pitch_filtered, label='Filtered Pitch', linewidth=1)
        plt.plot(peaks, pitch_filtered[peaks], "x", label='Maxima')
        plt.plot(valleys, pitch_filtered[valleys], "o", label='Minima')
        plt.xlabel('Sample index')
        plt.ylabel('Pitch (degrees)')
        plt.title('Pitch Signal with Detected Movements')
        plt.legend()
        plt.show()

    # ---------------- ranges + filter significant ----------------
    movement_ranges = [float(pitch_filtered[p] - pitch_filtered[v]) for v, p in movement_pairs]
    significant = [(pair, rng) for pair, rng in zip(movement_pairs, movement_ranges) if rng >= 5.0]

    filtered_pairs = [pair for pair, _ in significant]
    filtered_ranges = [rng for _, rng in significant]

    # ---------------- durations + total metrics ----------------
    movement_durations = []
    for s, e in filtered_pairs:
        start_time = df_Limu1.iloc[s].name
        end_time = df_Limu1.iloc[e].name
        movement_durations.append(float((end_time - start_time).total_seconds()))

    total_duration_seconds = float((df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds())
    pace = float(len(filtered_pairs) / total_duration_seconds) if total_duration_seconds > 0 else 0.0

    mean_range = float(np.mean(filtered_ranges)) if len(filtered_ranges) > 0 else 0.0
    std_range = float(np.std(filtered_ranges, ddof=1)) if len(filtered_ranges) >= 2 else 0.0

    mean_duration = float(np.mean(movement_durations)) if len(movement_durations) > 0 else 0.0
    std_duration = float(np.std(movement_durations, ddof=1)) if len(movement_durations) >= 2 else 0.0

    # ==========================================================
    # NEW METRICS (same schema as you requested for head pitch)
    # ==========================================================
    pitch_filtered = np.asarray(pitch_filtered, dtype=float)

    t0 = df_Limu1.index[0]
    t_sec = np.array([(ts - t0).total_seconds() for ts in df_Limu1.index], dtype=float)

    # sample-wise velocity (deg/s)
    vel_samples = np.abs(np.diff(pitch_filtered)) * fs
    vel_t_sec = t_sec[1:]

    # (A) per-1-second peak velocity curve
    n_bins = int(np.ceil(total_duration_seconds)) + 1
    velocity_curves_deg_per_s = [0.0] * n_bins

    if len(vel_samples) > 0:
        bin_idx = np.floor(vel_t_sec).astype(int)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid]
        v = vel_samples[valid]
        for b in np.unique(bin_idx):
            velocity_curves_deg_per_s[b] = float(np.max(v[bin_idx == b]))

    # (B) per-movement velocity curves -> mean/std curve
    TARGET_LEN = 101
    movement_velocity_curves = []
    peak_velocities_deg_per_s = []
    rom_peaks = []

    for start_idx, end_idx in filtered_pairs:
        s = max(int(start_idx), 0)
        e = min(int(end_idx), len(pitch_filtered) - 1)
        if e <= s:
            continue

        vel_seg = vel_samples[s:max(e, 1)]
        if len(vel_seg) == 0:
            vel_seg = np.array([0.0], dtype=float)

        peak_velocities_deg_per_s.append(float(np.max(vel_seg)))
        movement_velocity_curves.append(_resample_1d(vel_seg, TARGET_LEN))

        rom_peaks.append({
            "peak_deg": float(pitch_filtered[e]),
            "peak_time_s": float(t_sec[e]),
            "valley_deg": float(pitch_filtered[s]),
            "valley_time_s": float(t_sec[s]),
        })

    if len(movement_velocity_curves) == 0:
        velocity_mean_curve_deg_per_s = [0.0] * TARGET_LEN
        velocity_std_curve_deg_per_s = [0.0] * TARGET_LEN
    else:
        M = np.vstack(movement_velocity_curves)
        velocity_mean_curve_deg_per_s = [float(x) for x in np.mean(M, axis=0)]
        if M.shape[0] >= 2:
            velocity_std_curve_deg_per_s = [float(x) for x in np.std(M, axis=0, ddof=1)]
        else:
            velocity_std_curve_deg_per_s = [0.0] * TARGET_LEN

    # ---------------- final json (EXACT keys) ----------------
    metrics_data = {
        "total_metrics": {
            "number_of_movements": int(len(filtered_pairs)),
            "pace_movements_per_second": float(pace),
            "mean_range_degrees": float(mean_range),
            "std_range_degrees": float(std_range),
            "mean_duration_seconds": float(mean_duration),
            "std_duration_seconds": float(std_duration),
            "exercise_duration_seconds": float(total_duration_seconds)
        },
        "velocity_curves_deg_per_s": [float(x) for x in velocity_curves_deg_per_s],
        "velocity_mean_curve_deg_per_s": [float(x) for x in velocity_mean_curve_deg_per_s],
        "velocity_std_curve_deg_per_s": [float(x) for x in velocity_std_curve_deg_per_s],
        "peak_velocities_deg_per_s": [float(x) for x in peak_velocities_deg_per_s],
        "rom_peaks": rom_peaks
    }

    return json.dumps(metrics_data, indent=4)
