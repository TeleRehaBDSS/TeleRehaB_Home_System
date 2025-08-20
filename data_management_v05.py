import time
import json
import numpy as np
import re
from shared_variables import enableInterpolation, isFoundFirstTimestamp, firstTimestamp, imus, firstPacket, timeToCallMetrics, sensorDataToUpload, imu1Queue, imu2Queue, imu3Queue, imu4Queue, mqttState, enableMetrics, imu1FinalQueue, imu2FinalQueue, imu3FinalQueue, imu4FinalQueue, csv_file_path, imus, counter, startReceiving, lastDataTime, enableConnectionToAPI, feedbackData
from mqtt_messages import send_voice_instructions
from New_Metrics.MaintainingFocus_HeadUpandDown import get_metrics as get_metrics_MaintainingFocus_HeadUpandDown
from New_Metrics.MaintainingFocus_Headrotation import get_metrics as get_metrics_MaintainingFocus_Headrotation
from New_Metrics.SeatedBendingOver_v1 import get_metrics as get_metrics_SeatedBendingOver
from New_Metrics.TrunkRotation import get_metrics as get_metrics_Trunk_rotation
from New_Metrics.ToeRaisesQuat import get_metrics as get_metrics_ToeRaises
from New_Metrics.SitToStand_v2 import get_metrics as get_metrics_SitToStand
from New_Metrics.HeelRaisesQuat import get_metrics as get_metrics_HeelRaises
from New_Metrics.SeatedMarchingSpot import get_metrics as get_metrics_SeatedMarchingSpot
from New_Metrics.StandingBalance import get_metrics as get_metrics_StandingBalance
from New_Metrics.StandingBalanceFoam import get_metrics as get_metrics_StandingBalanceFoam
from New_Metrics.StandingBendingOver import get_metrics as get_metrics_StandingBendingOver
from New_Metrics.StandingTurn import get_metrics as get_metrics_StandingTurn
from New_Metrics.LateralWeightShiftsQuat import get_metrics as get_metrics_LateralWeightShifts
from New_Metrics.LimitsOfStability import get_metrics as get_metrics_LimitsofStability
from New_Metrics.ForwardReach import get_metrics as get_metrics_ForwardReach
from New_Metrics.ForwardWalking import get_metrics as get_metrics_ForwardWalking
from New_Metrics.ForwardWalkingYaw import get_metrics as get_metricsForwardWalkingYaw
from New_Metrics.ForwardWalkingTilt import get_metrics as get_metricsForwardWalkingTilt
from New_Metrics.SideStepping import get_metrics as get_metrics_SideStepping
from New_Metrics.WalkingHorizontalHeadTurns import get_metrics as get_metrics_WalkingHorizontalHeadTurns
from New_Metrics.Hip_External import get_metrics as get_metrics_HipExternal
from New_Metrics.Lateral_Trunk_Flexion import get_metrics as get_metrics_LateralTrunkFlexion
from New_Metrics.Calf_Stretch import get_metrics as get_metrics_CalfStretch
from New_Metrics.OverheadReach import get_metrics as get_metrics_OverheadReach
from New_Metrics.Side_Bend import get_metrics as get_metrics_SideBend
from mqtt_messages import init_mqtt_client,CAMERA_TOPIC
from csv_management import write_in_files
from api_management import upload_sensor_data, postExerciseScore
from sensor_data import SensorData
import multiprocessing as mp
from scipy.interpolate import interp1d
import csv
import zipfile
import queue




def zip_folder(folder_path):
    zip_filename = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    return zip_filename


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


import os
import matplotlib.pyplot as plt
from datetime import datetime

def get_data_tranch(q1, q2, q3, q4, counter, exercise):
    imu1List = []
    imu2List = []
    imu3List = []
    imu4List = []

    # Empty the queues into respective lists
    while not q1.empty():
        imu1List.append(q1.get())
    while not q2.empty():
        imu2List.append(q2.get())
    while not q3.empty():
        imu3List.append(q3.get())
    while not q4.empty():
        imu4List.append(q4.get())

    
    # Create directory for plots, common folder for the exercise and timestamp
    current_time = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{current_time}/{exercise}_{current_time}"
    os.makedirs(folder_name, exist_ok=True)

    # Save the data lists to CSV files
    save_list_to_csv(imu1List, 'HEAD', counter, folder_name)
    save_list_to_csv(imu2List, 'PELVIS', counter, folder_name)
    save_list_to_csv(imu3List, 'LEFTFOOT', counter, folder_name)
    save_list_to_csv(imu4List, 'RIGHTFOOT', counter, folder_name)

    # Plot and save the data for each IMU
    plot_imu_data(imu1List, "HEAD", counter, folder_name)
    plot_imu_data(imu2List, "PELVIS", counter, folder_name)
    plot_imu_data(imu3List, "LEFTFOOT", counter, folder_name)
    plot_imu_data(imu4List, "RIGHTFOOT", counter, folder_name)

    # Handle metrics if enabled
    if enableMetrics:
        if exercise == 'exer_01':
            metrics = get_metrics_MaintainingFocus_Headrotation(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_02':
            metrics = get_metrics_MaintainingFocus_HeadUpandDown(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_03':
            metrics = get_metrics_SeatedBendingOver(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_04':
            metrics = get_metrics_Trunk_rotation(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_05':
            metrics = get_metrics_ToeRaises(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_06':
            metrics = get_metrics_HeelRaises(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_07':
            metrics = get_metrics_SeatedMarchingSpot(imu1List, imu2List, imu3List, imu4List, counter) 
        elif exercise == 'exer_08':
            metrics = get_metrics_SitToStand(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_09':
             metrics = get_metrics_StandingBalance(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_10':
            metrics = get_metrics_StandingBalanceFoam(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_11':
            metrics = get_metrics_StandingBendingOver(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_12':
            metrics = get_metrics_StandingTurn(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_13':
            metrics = get_metrics_LateralWeightShifts(imu1List, imu2List, imu3List, imu4List, counter)    
        elif exercise == 'exer_14':
            metrics = get_metrics_LimitsofStability(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_15':
            metrics = get_metrics_ForwardReach(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_16':
            metrics = get_metrics_ForwardWalking(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_17':
            metrics = get_metricsForwardWalkingYaw(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_18':
            metrics = get_metricsForwardWalkingTilt(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_19':
            metrics = get_metrics_SideStepping(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_20' :
            metrics = get_metrics_WalkingHorizontalHeadTurns(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_21':
            metrics = get_metrics_HipExternal(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_22':
            metrics = get_metrics_LateralTrunkFlexion(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_23' :
            metrics = get_metrics_CalfStretch(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise =='exer_24' :
            metrics = get_metrics_OverheadReach(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_25':
            metrics = get_metrics_SideBend(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_28':
            metrics = get_metrics_MaintainingFocus_Headrotation(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_29':
            metrics = get_metrics_MaintainingFocus_HeadUpandDown(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_30':
            metrics = get_metrics_StandingBalance(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_31' :
            metrics = get_metrics_StandingBendingOver(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise =='exer_32' :
            metrics = get_metrics_OverheadReach(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_33':
            metrics = get_metrics_ForwardWalking(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_34':
            metrics = get_metricsForwardWalkingYaw(imu1List, imu2List, imu3List, imu4List, counter)
        elif exercise == 'exer_35':
            metrics = get_metricsForwardWalkingTilt(imu1List, imu2List, imu3List, imu4List, counter)
        else:
            metrics = None
        return metrics,folder_name
    return None


    # Final actions when counter reaches 4
    #if counter == 4:
    #    postExerciseScore(metrics, "1")    
    #print(metrics)

def plot_imu_data(imu_list, body_part, counter, folder_name):
    if not imu_list:
        return  # Skip if the list is empty

    # Extract w, x, y, z and timestamps
    timestamps = [item.timestamp for item in imu_list]
    w_values = [item.w for item in imu_list]
    x_values = [item.x for item in imu_list]
    y_values = [item.y for item in imu_list]
    z_values = [item.z for item in imu_list]

    # Create 4 subplots for w, x, y, z
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    fig.suptitle(f'{body_part} - Counter {counter}', fontsize=16)

    # Plot each component
    axs[0].plot(timestamps, w_values, label='w')
    axs[0].set_title(f'{body_part} - w')
    axs[0].set_xlabel('Timestamp')
    axs[0].set_ylabel('w')

    axs[1].plot(timestamps, x_values, label='x')
    axs[1].set_title(f'{body_part} - x')
    axs[1].set_xlabel('Timestamp')
    axs[1].set_ylabel('x')

    axs[2].plot(timestamps, y_values, label='y')
    axs[2].set_title(f'{body_part} - y')
    axs[2].set_xlabel('Timestamp')
    axs[2].set_ylabel('y')

    axs[3].plot(timestamps, z_values, label='z')
    axs[3].set_title(f'{body_part} - z')
    axs[3].set_xlabel('Timestamp')
    axs[3].set_ylabel('z')

    # Save the figure in the common folder with the counter in the file name
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{folder_name}/{body_part}_counter_{counter}.png')
    plt.close(fig)



import time
import logging
import sys

# It's good practice to set up proper logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def scheduler(scheduleQueue):
    try:
        time.sleep(1)
        client = init_mqtt_client("client_scheduler")
        logging.info("Waiting for 'STARTCOUNTING' message...")

        # Wait until "STARTCOUNTING" message is received in the scheduleQueue
        while True:
            # Using a direct blocking get is safer and more efficient than checking if empty first.
            message = scheduleQueue.get()
            if message == "startcounting":
                logging.info("Received 'startcounting'. Starting the scheduler...")
                try:
                    send_voice_instructions(client, "bph0082")
                    client.publish(CAMERA_TOPIC, "CAMERA_START")
                except Exception as e:
                    logging.error(f"Error during 'startcounting' actions: {e}", exc_info=True)
                break  # Exit the loop and proceed to start the scheduling
            else:
                logging.warning(f"Received unexpected message in queue: {message}")


        # Now start the main loop
        logging.info("Starting main scheduling loop...")
        while True:
            time.sleep(timeToCallMetrics)
            scheduleQueue.put("GO")

    except Exception as e:
        # This is the most important part. It will catch any crash and log it.
        logging.error(f"FATAL ERROR in scheduler process: {e}", exc_info=True)
        # exc_info=True will print the full error traceback

def parse_config_message(config_message):
    config_dict = {}
    parts = config_message.split(",")

    for part in parts:
        if "=" in part:
            body_part, rest = part.split("=")
            mac_address, mode = rest.split("-")
            if mode != "OFF":  # Only add to config_dict if the mode is not OFF
                config_dict[body_part] = (mac_address, mode)
        else:
            exercise_code = part  # Extract the exercise code (last part)
    print("config_dict = ", config_dict)
    print('------------------------------------------')
    return config_dict, exercise_code


def initialize_imu_data_structures(config_dict, manager):
    imu_data = {}

    for role in config_dict.keys():
        # For each IMU device (role), initialize a list and two queues
        imu_list = []
        imu_queue = manager.Queue()
        imu_final_queue = manager.Queue()
        
        # Store the lists and queues in the dictionary for that role (e.g., HEAD, PELVIS)
        imu_data[role] = (imu_list, imu_queue, imu_final_queue)
    
    return imu_data

def safe_get_imu_queue(imu_data, body_part, default_queue):
    """Safely retrieve the IMU queue, or return a default if the body part is not in the configuration."""
    return imu_data.get(body_part, [None, default_queue])[1]  # Access the queue from imu_data or return the default queue


def receive_imu_data(q, scheduleQueue, config_message, exercise,metrics_queue,ctg_queue):
    client = init_mqtt_client("client_receive_imu_data")
    # Initialize multiprocessing manager
    manager = mp.Manager()
    imu_config, exercise_code = parse_config_message(config_message)
    print('exercise_code =', exercise_code )

    # Create the data structures with manager-based queues
    imu_data = initialize_imu_data_structures(imu_config, manager)
    #print('imu_data = ', imu_data)

    global isFoundFirstTimestamp, enableInterpolation
    # Sample configuration message
    INTERVALS = 0;
    # Set of body parts that have received at least one sample
    received_body_parts = set()

     # **Clear old queue data**
    for role in imu_data:
        imu_list, imu_queue, imu_finalqueue = imu_data[role]
        imu_list.clear()  # Clear the stored sensor list
        while not imu_queue.empty():
            imu_queue.get()  # Empty the real-time processing queue
        while not imu_finalqueue.empty():
            imu_finalqueue.get()  # Empty the final processing queue
    
    # The set of body parts expected based on the configuration
    expected_body_parts = {key for key, value in imu_data.items() if value[1] != "OFF"}
    print('expected_body_parts = ', expected_body_parts)
    
    # If the shared data queue `q` contains old data, clear it before processing the new config
    while not q.empty():
        q.get()

    if exercise_code == 'exer_01':
        condition_met = False;
        start_time = time.time();

        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if ((INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>2):
                                    send_voice_instructions(client, "bph0080")

                                if ((INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>5):
                                    send_voice_instructions(client, "bph0085")

                                if ((INTERVALS==3) and metrics_dict["total_metrics"]["number_of_movements"]>10):
                                    send_voice_instructions(client, "bph0079")
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_02':
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if ((INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>2):
                                    send_voice_instructions(client, "bph0081")
                                if ((INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>5):
                                    send_voice_instructions(client, "bph0085")
                                if ((INTERVALS==3) and metrics_dict["total_metrics"]["number_of_movements"]>10):
                                    send_voice_instructions(client, "bph0086")
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_03': #Sitting Bending Over
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            TIMEOUT=40
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=TIMEOUT)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        start_time = time.monotonic()
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)

                                if ((INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>2):
                                    send_voice_instructions(client, "bph0085")

                                if ((INTERVALS==3) and metrics_dict["total_metrics"]["number_of_movements"]<5):
                                    send_voice_instructions(client, "bph0056")

                                if metrics_dict["total_metrics"]["number_of_movements"]>=5:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return  # Exit the function immediately to stop the exercise
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                    
                                    break;
                    elif time.monotonic() - start_time >40:
                        print("check data stream")
                        send_voice_instructions(client, "bph0082")
                        time.sleep(40)
                        send_voice_instructions(client, "bph0083")
                        break;    
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40)
                send_voice_instructions(client, "bph0083")
                break;

    elif exercise_code == 'exer_04': #Trunk Rotation
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                # interpolate
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)

                                if ((INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>2):
                                    send_voice_instructions(client, "bph0080")
                                if ((INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>6):
                                    send_voice_instructions(client, "bph0086")

                                if metrics_dict["total_metrics"]["number_of_movements"]>=20:
                                    send_voice_instructions(client, "bph0083") 
                                    while not q.empty():
                                        q.get()
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    return
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_05': #Toe Raises
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)

                                if ((INTERVALS==1) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >5):
                                    send_voice_instructions(client, "bph0080")
                                
                                if ((INTERVALS==2) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >10):
                                    send_voice_instructions(client, "bph0085")

                                if (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >= 15:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics =  metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return  # Exit the function immediately to stop the exercise

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_06': #Heel Raises
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address
                    #print(body_part)

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if ((INTERVALS==1) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >5):
                                    send_voice_instructions(client, "bph0081")
                                
                                if ((INTERVALS==2) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >8):
                                    send_voice_instructions(client, "bph0086")

                                if (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >= 15:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return  # Exit the function immediately to stop the exercise
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_07': #Seated Marching Spot
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )
                                
                                metrics_dict = json.loads(metrics)
                                if ((INTERVALS==1) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >= 5):
                                    send_voice_instructions(client, "bph0079")
                                if ((INTERVALS==2) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >= 9):
                                    send_voice_instructions(client, "bph0085")
                                
                                
                                if (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] >=10) and (metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"] >= 10):
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return  # Exit the function immediately to stop the exercise
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_08': #Sit To Stand
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                # interpolate
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>1:
                                    send_voice_instructions(client, "bph0079")
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>3:
                                    send_voice_instructions(client, "bph0086")
                                if (INTERVALS==3) and metrics_dict["total_metrics"]["number_of_movements"]<5:
                                    send_voice_instructions(client, "bph0056")    
                                
                                if metrics_dict["total_metrics"]["number_of_movements"] >=5:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_09': #Standing Balance
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;   
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )
                                
                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>2:
                                    send_voice_instructions(client, "bph0053")
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_10': #Balance Foam
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                # interpolate
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>2:
                                    send_voice_instructions(client, "bph0053")
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_11': #Standing Bend Over
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>1:
                                    send_voice_instructions(client, "bph0081")
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>3:
                                    send_voice_instructions(client, "bph0086")
                                
                                if metrics_dict["total_metrics"]["number_of_movements"] >=5:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_12': #Standing Turn
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 5 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                # interpolate
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate
                                
                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)


                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_13': #Lateral Weight Shifts
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict=json.loads(metrics)
                                if ((INTERVALS==1) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >2):
                                    send_voice_instructions(client, "bph0080")
                                
                                if ((INTERVALS==2) and (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >4):
                                    send_voice_instructions(client, "bph0085")

                                if (metrics_dict["total_metrics"]['LEFT LEG']["number_of_movements"] + metrics_dict["total_metrics"]['RIGHT LEG']["number_of_movements"]) >= 10:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics =  metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return  # Exit the function immediately to stop the exercise


                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_14': #Limits of Stability
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 2 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )
                                
                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==0) and metrics_dict["total_metrics"]["number_of_movements"]>5:
                                    send_voice_instructions(client, "bph0081")
                                if (INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>7:
                                    send_voice_instructions(client, "bph0086")

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 2:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(25);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_15': #Forward Reach
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)

                                if (INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>2:
                                    send_voice_instructions(client,"bph0079")
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>5:
                                    send_voice_instructions(client,"bph0085")
                                if (INTERVALS==3) and metrics_dict["total_metrics"]["number_of_movements"]>12:
                                    send_voice_instructions(client,"bph0086")

                                if metrics_dict["total_metrics"]["number_of_movements"] >= 20:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_16': #Forward walking
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_17': #Forward Walking Yaw
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_18': #Forward Walking Tilt
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_19':#Side Stepping
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try :
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout = 40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_20': #Walking Scanning
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                # interpolate
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_21': #Hip External
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_22':#Lateral Trunk Flexion
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 60):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=60)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_23': #Calf Stretch
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_24': #Overhead Reach
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>1:
                                    send_voice_instructions(client, "bph0081")
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>3:
                                    send_voice_instructions(client, "bph0086")
                                
                                if metrics_dict["total_metrics"]["number_of_movements"] >=5000:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;

    elif exercise_code == 'exer_25': #Side_Bend
        condition_met = False;
        start_time = time.time();
        while INTERVALS < 4 or not q.empty():
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty():
                            message = scheduleQueue.get()  # Block until a message is received
                            if message == "GO":
                                # Call the data processing function based on the queues
                                print('I am calling get_data_tranch')
                                process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                metrics,folder_path = get_data_tranch(
                                    safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                    safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                    safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                    safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                    INTERVALS,  # Pass the counter variable
                                    exercise_code  # Pass the exercise code
                                )

                                metrics_dict = json.loads(metrics)
                                if (INTERVALS==1) and metrics_dict["total_metrics"]["number_of_movements"]>1:
                                    send_voice_instructions(client, "bph0081")
                                if (INTERVALS==2) and metrics_dict["total_metrics"]["number_of_movements"]>3:
                                    send_voice_instructions(client, "bph0086")
                                
                                if metrics_dict["total_metrics"]["number_of_movements"] >=10:
                                    send_voice_instructions(client, "bph0083")
                                    final_metrics = metrics
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    while not q.empty():
                                        q.get()
                                    return
                                
                                scheduleQueue.get()  # Consume the scheduled task
                                INTERVALS += 1
                                print(f"Intervals = {INTERVALS}")

                                if INTERVALS == 4:
                                    send_voice_instructions(client, "bph0083")
                                    print("Processing the entire data stream...")
                                    final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                    if final_metrics is not None:
                                        metrics_queue.put(final_metrics)
                                        data_zip_path = zip_folder(folder_path)
                                        scheduleQueue.put(("data_zip", data_zip_path))
                                    break;
            except Exception as e:
                print("check data stream")
                send_voice_instructions(client, "bph0082")
                time.sleep(40);
                send_voice_instructions(client, "bph0083")
                break;
    elif exercise_code == 'exer_28':
        condition_met = False;
        start_time = time.time();
        while (not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream");
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    elif exercise_code == 'exer_29':
        condition_met = False;
        start_time = time.time();
        while (  not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    elif exercise_code == 'exer_30':
        condition_met = False;
        start_time = time.time();
        while (  not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    elif exercise_code == 'exer_31':
        condition_met = False;
        start_time = time.time();
        while (  not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    elif exercise_code == 'exer_32':
        condition_met = False;
        start_time = time.time();
        while (  not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    elif exercise_code == 'exer_33':
        condition_met = False;
        start_time = time.time();
        while (  not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    elif exercise_code == 'exer_34':
        condition_met = False;
        start_time = time.time();        
        while ( (not q.empty()) or INTERVALS <= 4 ) :
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;    
    elif exercise_code == 'exer_35':
        condition_met = False;
        start_time = time.time();
        while (  not q.empty()) or INTERVALS <= 5:
            try:
                if not condition_met and (time.time() - start_time > 40):
                    raise TimeoutError("Timeout: expected body parts not received")

                data = q.get(timeout=40)  # Read data from the dataQueue

                if ":" in data:  # Check for valid sensor data format
                    # Convert JSON data to SensorData object
                    sensor_data = SensorData.from_json(data)

                    # Use deviceMacAddress as the body part identifier
                    body_part = sensor_data.device_mac_address

                    if body_part in expected_body_parts:
                        received_body_parts.add(body_part);
                    
                    if (received_body_parts == expected_body_parts):
                        condition_met = True;
                        scheduleQueue.put('startcounting')
                        # Fetch the relevant lists and queues from imu_data
                        if body_part in imu_data:
                            imu_list, imu_queue, imu_finalqueue = imu_data[body_part]

                            # Add the sensor data to the appropriate structures
                            imu_list.append(sensor_data)
                            imu_queue.put(sensor_data)
                            imu_finalqueue.put(sensor_data)
                        else:
                            print(f"Unrecognized body part: {body_part}")

                        # Check if there is something in the schedule queue
                        if not scheduleQueue.empty() or not ctg_queue.empty():
                            if not scheduleQueue.empty():
                                message = scheduleQueue.get()  # Block until a message is received
                                if message == "GO":
                                    # Call the data processing function based on the queues
                                    print('I am calling get_data_tranch')
                                    process_and_interpolate_imus(imu_data, target_rate_hz=100)  # 100 Hz target rate

                                    metrics,folder_path = get_data_tranch(
                                        safe_get_imu_queue(imu_data, 'HEAD', manager.Queue()),  # imu1Queue
                                        safe_get_imu_queue(imu_data, 'PELVIS', manager.Queue()),  # imu2Queue
                                        safe_get_imu_queue(imu_data, 'LEFTFOOT', manager.Queue()),  # imu3Queue
                                        safe_get_imu_queue(imu_data, 'RIGHTFOOT', manager.Queue()),  # imu4Queue
                                        INTERVALS,  # Pass the counter variable
                                        exercise_code  # Pass the exercise code
                                    )

                                    metrics_dict = json.loads(metrics)
                                    
                                    scheduleQueue.get()  # Consume the scheduled task
                                    INTERVALS += 1
                                    print(f"Intervals = {INTERVALS}")

                            if  not (ctg_queue.empty()) :
                                while not ctg_queue.empty():
                                    ctg_queue.get()

                                print("Processing the entire data stream...")
                                final_metrics,folder_path = get_data_tranch(
                                        imu_data.get('HEAD', [None, None, manager.Queue()])[2],  # imu1FinalQueue
                                        imu_data.get('PELVIS', [None, None, manager.Queue()])[2],  # imu2FinalQueue
                                        imu_data.get('LEFTFOOT', [None, None, manager.Queue()])[2],  # imu3FinalQueue
                                        imu_data.get('RIGHTFOOT', [None, None, manager.Queue()])[2],  # imu4FinalQueue
                                        INTERVALS,
                                        exercise_code
                                    )
                                if final_metrics is not None:
                                    metrics_queue.put(final_metrics)
                                    data_zip_path = zip_folder(folder_path)
                                    scheduleQueue.put(("data_zip", data_zip_path))
                                break;
            except Exception as e:
                print("check data stream")
                while ctg_queue.empty():
                    time.sleep(0.5)
                time.sleep(1)
                while not ctg_queue.empty():
                    ctg_queue.get()
                break;
    return 0;
            

def save_list_to_csv(imu_list, body_part, counter, folder_name):
    if not imu_list:
        return  # Skip if the list is empty

    # Create the CSV file path
    csv_file_path = os.path.join(folder_name, f"{body_part}_data_counter_{counter}.csv")
    
    # Write data to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'w', 'x', 'y', 'z'])  # CSV headers
        for data in imu_list:
            writer.writerow([data.timestamp, data.w, data.x, data.y, data.z])


def find_common_time_period(imu_data):
    """
    Finds the common time period for all available IMU data lists.
    Returns the maximum start time and the minimum end time across all non-empty lists.
    """
    max_start_time = float('-inf')
    min_end_time = float('inf')

    for body_part, (imu_list, _, _) in imu_data.items():
        if imu_list:  # Check if the list is not empty
            # Access the timestamp from the first and last SensorData object in the list
            start_time = imu_list[0].timestamp
            end_time = imu_list[-1].timestamp

            # Update the common time range
            max_start_time = max(max_start_time, start_time)
            min_end_time = min(min_end_time, end_time)

    return max_start_time, min_end_time

# Function to crop the IMU data
def crop_imu_data(imu_data, max_start_time, min_end_time):
    cropped_data = {}
    
    for body_part, (imu_list, imu_queue, imu_finalqueue) in imu_data.items():
        # Filter data between max_start_time and min_end_time
        cropped_list = [d for d in imu_list if max_start_time <= d['timestamp'] <= min_end_time]
        cropped_data[body_part] = (cropped_list, imu_queue, imu_finalqueue)
    
    return cropped_data


# Function to interpolate IMU data to 100Hz
def interpolate_imu_data(imu_list, target_rate_hz, max_start_time, min_end_time):
    """
    Interpolate the IMU data to match the target sampling rate and common time period.
    """
    if not imu_list:
        return []

    # Extract timestamps and other data (e.g., w, x, y, z)
    timestamps = np.array([data.timestamp for data in imu_list])
    w_values = np.array([data.w for data in imu_list])
    x_values = np.array([data.x for data in imu_list])
    y_values = np.array([data.y for data in imu_list])
    z_values = np.array([data.z for data in imu_list])

    # Sort data by timestamps to ensure chronological order
    sorted_indices = np.argsort(timestamps)
    timestamps = timestamps[sorted_indices]
    w_values = w_values[sorted_indices]
    x_values = x_values[sorted_indices]
    y_values = y_values[sorted_indices]
    z_values = z_values[sorted_indices]

    # Remove duplicates in timestamps to ensure each timestamp is unique
    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)
    w_values = w_values[unique_indices]
    x_values = x_values[unique_indices]
    y_values = y_values[unique_indices]
    z_values = z_values[unique_indices]

    # Limit timestamps to the common time period
    mask = (unique_timestamps >= max_start_time) & (unique_timestamps <= min_end_time)
    unique_timestamps = unique_timestamps[mask]
    w_values = w_values[mask]
    x_values = x_values[mask]
    y_values = y_values[mask]
    z_values = z_values[mask]

    if len(unique_timestamps) == 0:
        return []  # Return an empty list if no valid data points

    # Calculate the new timestamps with the target rate
    new_timestamps = np.arange(max_start_time, min_end_time, 1000.0 / target_rate_hz)  # target rate in ms
    new_timestamps = np.unique(new_timestamps)  # Ensure unique timestamps for the new resampled timestamps
    
    # Interpolate values to the new timestamps
    interp_w = np.interp(new_timestamps, unique_timestamps, w_values)
    interp_x = np.interp(new_timestamps, unique_timestamps, x_values)
    interp_y = np.interp(new_timestamps, unique_timestamps, y_values)
    interp_z = np.interp(new_timestamps, unique_timestamps, z_values)

    # Return the interpolated data as a list of SensorData objects
    interpolated_data = []
    for i in range(len(new_timestamps)):
        interpolated_data.append(SensorData(
            device_mac_address=imu_list[0].device_mac_address,  # Use the same device address
            timestamp=new_timestamps[i],
            w=interp_w[i],
            x=interp_x[i],
            y=interp_y[i],
            z=interp_z[i]
        ))
    return interpolated_data



def process_and_interpolate_imus(imu_data, target_rate_hz):
    """
    Process and interpolate each IMU data list based on the target sampling rate.
    """
    # Find the common time period for all IMU data
    max_start_time, min_end_time = find_common_time_period(imu_data)

    for body_part, (imu_list, imu_queue, imu_finalqueue) in imu_data.items():
        if imu_list:
            #imu_list.sort(key=lambda x: x.timestamp)
            # Interpolate the list and replace the original data with interpolated data
            interpolated_list = interpolate_imu_data(imu_list, target_rate_hz, max_start_time, min_end_time)

            # Clear the imu_queue and imu_finalqueue before adding new interpolated data
            while not imu_queue.empty():
                imu_queue.get()
            
            while not imu_finalqueue.empty():
                imu_finalqueue.get()

            # Update the imu_queue and imu_finalqueue with the interpolated data
            for data in interpolated_list:
                imu_queue.put(data)  # Add the interpolated data to the imu_queue
                imu_finalqueue.put(data)  # Also add to the final queue



            imu_data[body_part] = (interpolated_list, imu_queue, imu_finalqueue)
            print(f"{body_part} data interpolated.")
        else:
            print(f"No data available for {body_part}, skipping interpolation.")

