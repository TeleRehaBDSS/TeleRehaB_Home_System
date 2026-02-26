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
    #Limu1 = striplist(imu1)
    #Limu2 = striplist(imu2)
    #Limu3 = striplist(imu3)
    #Limu4 = striplist(imu4)

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

    #print('Limu1 = ', Limu1)
    if(len(Limu1) > 0):
        print('procceding to metrics...')
        returnedJson = getMetricsSittingOld01(Limu1,Limu2,Limu3,Limu4, False) 
        return returnedJson
    
def _resample_1d(arr, target_len):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return np.zeros(target_len, dtype=float)
    if len(arr) == 1:
        return np.full(target_len, arr[0], dtype=float)

    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, arr)

def _safe_std(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1))

def getMetricsSittingOld01(Limu1, Limu2, Limu3, Limu4, plotdiagrams):
    fs = 50
    cutoff = 0.5

    import numpy as np
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    from scipy.signal import find_peaks

    # ---------- helpers (local) ----------
    def _resample_1d(arr, target_len):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return np.zeros(target_len, dtype=float)
        if len(arr) == 1:
            return np.full(target_len, arr[0], dtype=float)
        x_old = np.linspace(0.0, 1.0, len(arr))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, arr)

    # ---------- build df ----------
    columns = ['Timestamp', 'elapsed(time)', 'W(number)', 'X(number)', 'Y(number)', 'Z(number)']
    df_Limu1 = pd.DataFrame(Limu1, columns=columns)
    df_Limu1['Timestamp'] = pd.to_datetime(df_Limu1['Timestamp'])
    df_Limu1 = df_Limu1.sort_values(by='Timestamp')
    df_Limu1.set_index('Timestamp', inplace=True)

    # ---------- quaternion -> euler (deg) ----------
    quaternions = df_Limu1[['X(number)', 'Y(number)', 'Z(number)', 'W(number)']].to_numpy()
    rotations = R.from_quat(quaternions)
    euler_angles_degrees = rotations.as_euler('xyz', degrees=True)
    euler_df_degrees = pd.DataFrame(
        euler_angles_degrees,
        columns=['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)'],
        index=df_Limu1.index
    )

    # ---------- optional plots ----------
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

    # ---------- filter yaw ----------
    yaw_filtered = butter_lowpass_filter(euler_df_degrees['Yaw (degrees)'].to_numpy(), cutoff, fs, order=5)

    if plotdiagrams:
        plt.figure(figsize=(12, 6))
        plt.plot(euler_df_degrees.index, euler_df_degrees['Yaw (degrees)'], label='Original Yaw', linewidth=1, alpha=0.5)
        plt.plot(euler_df_degrees.index, yaw_filtered, label='Filtered Yaw', linewidth=2)
        plt.xlabel('Timestamp')
        plt.ylabel('Yaw (degrees)')
        plt.title('Yaw Signal Filtering')
        plt.legend()
        plt.show()

    # ---------- detect peaks/valleys ----------
    peaks, _ = find_peaks(yaw_filtered)
    valleys, _ = find_peaks(-yaw_filtered)

    if len(peaks) == 0 or len(valleys) == 0:
        return json.dumps({
            "total_metrics": {
                "number_of_movements": 0,
                "pace_movements_per_second": 0.0,
                "mean_range_degrees": 0.0,
                "std_range_degrees": 0.0,
                "mean_duration_seconds": 0.0,
                "std_duration_seconds": 0.0,
                "Exercise duration (seconds)": float((df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds())
            },
            "velocity_curves_deg_per_s": [],
            "velocity_mean_curve_deg_per_s": [],
            "velocity_std_curve_deg_per_s": [],
            "peak_velocities_deg_per_s": [],
            "rom_peaks": []
        }, indent=4)

    # Align so we always pair (valley -> peak)
    if valleys[0] > peaks[0]:
        peaks = peaks[1:]
    if len(peaks) > 0 and len(valleys) > 0 and peaks[-1] < valleys[-1]:
        valleys = valleys[:-1]

    movement_pairs = []
    for i in range(min(len(peaks), len(valleys))):
        movement_pairs.append((valleys[i], peaks[i]))

    if plotdiagrams:
        plt.figure(figsize=(12, 6))
        plt.plot(yaw_filtered, label='Filtered Yaw', linewidth=1)
        plt.plot(peaks, yaw_filtered[peaks], "x", label='Maxima')
        plt.plot(valleys, yaw_filtered[valleys], "o", label='Minima')
        plt.xlabel('Sample index')
        plt.ylabel('Yaw (degrees)')
        plt.title('Yaw Signal with Detected Movements')
        plt.legend()
        plt.show()

    # ---------- movement ranges + filter ----------
    movement_ranges = []
    for valley, peak in movement_pairs:
        movement_ranges.append(float(yaw_filtered[peak] - yaw_filtered[valley]))

    # Keep only "significant" movements (your threshold = 5 deg)
    significant = [(pair, rng) for pair, rng in zip(movement_pairs, movement_ranges) if rng >= 5.0]
    filtered_pairs = [pair for pair, _ in significant]
    filtered_ranges = [rng for _, rng in significant]

    # ---------- durations, pace, mean/std ----------
    movement_durations = []
    for start, end in filtered_pairs:
        start_time = df_Limu1.iloc[start].name
        end_time = df_Limu1.iloc[end].name
        movement_durations.append(float((end_time - start_time).total_seconds()))

    total_duration_seconds = float((df_Limu1.index[-1] - df_Limu1.index[0]).total_seconds())
    pace = float(len(filtered_pairs) / total_duration_seconds) if total_duration_seconds > 0 else 0.0

    mean_range = float(np.mean(filtered_ranges)) if len(filtered_ranges) > 0 else 0.0
    std_range = float(np.std(filtered_ranges, ddof=1)) if len(filtered_ranges) >= 2 else 0.0

    mean_duration = float(np.mean(movement_durations)) if len(movement_durations) > 0 else 0.0
    std_duration = float(np.std(movement_durations, ddof=1)) if len(movement_durations) >= 2 else 0.0

    # ==========================================================
    # NEW METRICS (exact schema you asked)
    # ==========================================================

    yaw_filtered = np.asarray(yaw_filtered, dtype=float)

    # elapsed seconds array aligned to yaw samples
    t0 = df_Limu1.index[0]
    t_sec = np.array([(ts - t0).total_seconds() for ts in df_Limu1.index], dtype=float)

    # sample-wise velocity (deg/s)
    vel_samples = np.abs(np.diff(yaw_filtered)) * fs
    vel_t_sec = t_sec[1:]  # aligns with vel_samples

    # (A) velocity_curves_deg_per_s: per-1-second peak velocity, length = ceil(duration)+1
    n_bins = int(np.ceil(total_duration_seconds)) + 1
    velocity_curves_deg_per_s = [0.0] * n_bins

    if len(vel_samples) > 0:
        bin_idx = np.floor(vel_t_sec).astype(int)
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid]
        v = vel_samples[valid]
        for b in np.unique(bin_idx):
            velocity_curves_deg_per_s[b] = float(np.max(v[bin_idx == b]))

    # (B) per-movement curves -> mean/std curve
    TARGET_LEN = 101
    movement_velocity_curves = []
    peak_velocities_deg_per_s = []
    rom_peaks = []

    for start_idx, end_idx in filtered_pairs:
        s = max(int(start_idx), 0)
        e = min(int(end_idx), len(yaw_filtered) - 1)
        if e <= s:
            continue

        # velocity segment slice (approx)
        vs = max(s, 0)
        ve = max(e, 1)
        vel_seg = vel_samples[vs:ve]
        if len(vel_seg) == 0:
            vel_seg = np.array([0.0], dtype=float)

        peak_velocities_deg_per_s.append(float(np.max(vel_seg)))
        movement_velocity_curves.append(_resample_1d(vel_seg, TARGET_LEN))

        rom_peaks.append({
            "peak_deg": float(yaw_filtered[e]),
            "peak_time_s": float(t_sec[e]),
            "valley_deg": float(yaw_filtered[s]),
            "valley_time_s": float(t_sec[s]),
        })

    if len(movement_velocity_curves) == 0:
        velocity_mean_curve_deg_per_s = [0.0] * TARGET_LEN
        velocity_std_curve_deg_per_s = [0.0] * TARGET_LEN
    else:
        M = np.vstack(movement_velocity_curves)  # (movements, TARGET_LEN)
        velocity_mean_curve_deg_per_s = [float(x) for x in np.mean(M, axis=0)]
        if M.shape[0] >= 2:
            velocity_std_curve_deg_per_s = [float(x) for x in np.std(M, axis=0, ddof=1)]
        else:
            velocity_std_curve_deg_per_s = [0.0] * TARGET_LEN

    # ---------- final json (EXACT keys) ----------
    metrics_data = {
        "total_metrics": {
            "number_of_movements": int(len(filtered_pairs)),
            "pace_movements_per_second": float(pace),
            "mean_range_degrees": float(mean_range),
            "std_range_degrees": float(std_range),
            "mean_duration_seconds": float(mean_duration),
            "std_duration_seconds": float(std_duration),
            "Exercise duration (seconds)": float(total_duration_seconds)
        },
        "velocity_curves_deg_per_s": [float(x) for x in velocity_curves_deg_per_s],
        "velocity_mean_curve_deg_per_s": [float(x) for x in velocity_mean_curve_deg_per_s],
        "velocity_std_curve_deg_per_s": [float(x) for x in velocity_std_curve_deg_per_s],
        "peak_velocities_deg_per_s": [float(x) for x in peak_velocities_deg_per_s],
        "rom_peaks": rom_peaks
    }

    return json.dumps(metrics_data, indent=4)
