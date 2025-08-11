import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d



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
        returnedJson = getMetricsSeatingOld03(Limu1, Limu2, False) 
        return returnedJson

def getMetricsSeatingOld03(Limu1, Limu2, plotdiagrams):
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

    # --- Convert quaternion to Euler (roll, pitch, yaw) ---
    quat = df_Limu2[['X(number)', 'Y (number)', 'Z (number)', 'W(number)']].values
    rotations = R.from_quat(quat)
    euler_angles = rotations.as_euler('xyz', degrees=True)
    roll = gaussian_filter1d(euler_angles[:, 0], sigma=5)

    # --- Zero-crossing detection ---
    zero_crossings = np.where(np.diff(np.sign(roll)) != 0)[0]

    # --- Detect reps and movement durations ---
    reps = []
    right_durations = []
    left_durations = []
    right_angles = []
    left_angles = []

    zc_list = list(zero_crossings)
    for i in range(len(zc_list) - 1):
        start = zc_list[i]
        end = zc_list[i + 1]
        segment = roll[start:end]
        if len(segment) == 0:
            continue

        peak_idx_rel = np.argmax(np.abs(segment))
        peak_value = segment[peak_idx_rel]
        peak_idx = start + peak_idx_rel

        start_time = df_Limu2.index[start]
        end_time = df_Limu2.index[end]
        duration = (end_time - start_time).total_seconds()

        if peak_value > 0:
            reps.append(('right', peak_idx, peak_value))
            right_durations.append(duration)
            right_angles.append(peak_value)
        else:
            reps.append(('left', peak_idx, peak_value))
            left_durations.append(duration)
            left_angles.append(abs(peak_value))  # abs for positive angle

    # --- Summary Stats ---
    total_duration_seconds = (df_Limu2.index[-1] - df_Limu2.index[0]).total_seconds()

    mean_right_angle = np.mean(right_angles) if right_angles else 0
    mean_left_angle = np.mean(left_angles) if left_angles else 0
    symmetry = round((min(mean_left_angle, mean_right_angle) / max(mean_left_angle, mean_right_angle)) * 100, 2) if right_angles and left_angles else 0

    metrics_data = {
        "total_metrics": {
            "number_of_movements": len(reps),
            "movement_mean_time": round(np.mean(right_durations + left_durations), 3) if reps else 0,
            "movement_std_time": round(np.std(right_durations + left_durations), 3) if reps else 0,
            "range_mean_degrees_pelvis_right": round(mean_right_angle, 2),
            "range_std_degrees_pelvis_right": round(np.std(right_angles), 2) if right_angles else 0,
            "range_mean_degrees_pelvis_left": round(mean_left_angle, 2),
            "range_std_degrees_pelvis_left": round(np.std(left_angles), 2) if left_angles else 0,
            "symmetry": symmetry,
            "exercise_duration_seconds": round(total_duration_seconds, 2)
        }
    }

    print(metrics_data)
        
        
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{datetime_string}_SideBendingOver_metrics.txt"

    # Save the metrics to a file
    save_metrics_to_txt(metrics_data, filename)


    return json.dumps(metrics_data, indent=4)

    
def save_metrics_to_txt(metrics, file_path):
    main_directory = "Sitting Metrics Data"
    sub_directory = "SeatedBendingOver Metrics Data"

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
