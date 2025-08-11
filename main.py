import logging
import json
import paho.mqtt.client as mqtt
import sys
import os
import multiprocessing as mp
import threading
import time
import requests
import configparser
from datetime import datetime
from mqtt_messages import init_mqtt_client, set_language, start_exercise_demo, send_voice_instructions,send_message_with_speech_to_text,send_message_with_speech_to_text_2,send_exit,start_cognitive_games,start_exergames,send_message_with_speech_to_text_ctg,send_message_with_speech_to_text_ctg_2,send_voice_instructions_ctg,app_connected
from data_management_v05 import scheduler, receive_imu_data
from api_management import login, get_device_api_key
from scoring import give_score
from configure_file_management import read_configure_file
from Polar_test import start_ble_process 
from shared_variables import (
    queueData, scheduleQueue, enableConnectionToAPI,
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL,
    mqttState
)
from UDPSERVER import start_multicast_server
from UDPClient import SendMyIP
from websocketServer import run_websocket_server
from pathlib import Path


# Get the directory where the script is located
BASE_DIR = Path(__file__).resolve().parent

# Construct the paths for config and logo
CONFIG_PATH = BASE_DIR / 'config.ini'
TOPIC_PING = "healthcheck/AREYOUALIVE"
TOPIC_PONG = "healthcheck/IAMALIVE"

def reorder_exercises(exercises): #Function that add the exer and cognitive games at the end
    priority_ids = list(range(1, 28))  
    special_id = 43

    # Split exercises
    priority_exercises = [ex for ex in exercises if ex["exerciseId"] in priority_ids]
    special_exercise = [ex for ex in exercises if ex["exerciseId"] == special_id]
    other_exercises = [ex for ex in exercises if ex["exerciseId"] not in priority_ids + [special_id]]

    # Sort each group
    priority_exercises.sort(key=lambda x: x["exerciseId"])
    other_exercises.sort(key=lambda x: x["exerciseId"])

    # Concatenate
    return priority_exercises + special_exercise + other_exercises

def send_heartbeat():
    global received_response
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)

    while True:
        received_response.value = 0  # Reset flag
        client.publish(TOPIC_PING, "AREYOUALIVE")
        print('------------------HEALTHCHECK-----------------------')
        time.sleep(30)  # Wait for 30 seconds

        # Wait for a response for a few more seconds before logging failure
        timeout = time.time() + 5
        while time.time() < timeout:
            if received_response.value == 1:
                break
            time.sleep(1)

        if received_response.value == 0:
            print("WARNING: No response received from the mobile app!")
            os.system("pkill -f 'gnome-terminal'")
    
            sys.exit(1)  # Exit the program with error code 1

def on_message_healthcheck(client, userdata, msg):
    global received_response
    if msg.topic == TOPIC_PONG:
        received_response.value = 1  # Mark that the response was received
        print('I got msg from app')

def start_mqtt_listener():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message_healthcheck
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.subscribe(TOPIC_PONG)
    client.loop_forever()  # Blocking call to listen continuously

# Define file storage API endpoints
FILE_STORAGE_BASE_URL = "https://telerehab-develop.biomed.ntua.gr/filestorage"
LOGS_ENDPOINT = f"{FILE_STORAGE_BASE_URL}/Logs"
DATA_ENDPOINT = f"{FILE_STORAGE_BASE_URL}/Data"

def get_api_key():
    """Fetch API key from config file."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config['API'].get('key_edge', '')

def convert_log_to_txt(log_file):
    """Converts .log file to .txt format."""
    txt_file = log_file.replace(".log", ".txt")
    with open(log_file, 'r') as lf, open(txt_file, 'w') as tf:
        tf.write(lf.read())
    return txt_file

def upload_file(file_path, file_type="Logs"):
    """Uploads a file to the file storage API."""
    if file_path.endswith(".log"):
        file_path = convert_log_to_txt(file_path)
    
    url = f"{FILE_STORAGE_BASE_URL}/{file_type}"
    headers = {
        'Authorization': get_api_key()
    }
    files = {'files': open(file_path, 'rb')}
    response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        logging.info(f"File {file_path} uploaded successfully to {file_type}.")
    else:
        logging.error(f"Failed to upload {file_path}. Status: {response.status_code}, Response: {response.text}")

def get_file_list(file_type="Logs"):
    """Retrieves the list of files from the file storage API."""
    url = f"{FILE_STORAGE_BASE_URL}/{file_type}/list"
    headers = {
        'Authorization': get_api_key()
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to retrieve file list. Status: {response.status_code}, Response: {response.text}")
        return None

def download_file(file_id, save_path, file_type="Logs"):
    """Downloads a file from the file storage API."""
    url = f"{FILE_STORAGE_BASE_URL}/{file_type}/list/{file_id}"
    headers = {
        'Authorization': get_api_key()
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"File {file_id} downloaded successfully to {save_path}.")
    else:
        logging.error(f"Failed to download file {file_id}. Status: {response.status_code}, Response: {response.text}")


# Set up logger with unique filename
log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"session_{log_timestamp}.log"
logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s') 
file_handler = logging.FileHandler(log_filename, mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Redirect stdout and stderr to logger
class StreamToLogger:
    def __init__(self, log_level):
        self.log_level = log_level

    def write(self, message):
        if message.strip():
            self.log_level(message.strip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger.info)
sys.stderr = StreamToLogger(logger.error)

def get_devices():
    """Fetch the daily schedule from the API."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    api_key_edge = config['API'].get('key_edge', '')
    
    url = 'http://telerehab-develop.biomed.ntua.gr/api/PatientDeviceSet'
    headers = {
        'accept': '*/*',
        'Authorization': api_key_edge
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
# Helper functions for API interaction
def get_daily_schedule():
    """Fetch the daily schedule from the API."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    api_key_edge = config['API'].get('key_edge', '')
    
    url = 'http://telerehab-develop.biomed.ntua.gr/api/PatientSchedule/daily'
    headers = {
        'accept': '*/*',
        'Authorization': api_key_edge
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def post_results(score, exercise_id):
    """Fetch the daily schedule from the API."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    api_key_edge = config['API'].get('key_edge', '')
    """Post metrics to the PerformanceScore API."""
    try:
        url = "https://telerehab-develop.biomed.ntua.gr/api/PerformanceScore"
        date_posted = datetime.now().isoformat()
        post_data = {
            "score": score,
            "exerciseId": exercise_id,
            "datePosted": date_posted
        }
        headers = {
            "Authorization": api_key_edge,  
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=post_data, headers=headers)
        
        if response.status_code == 200:
            logger.info(f"Metrics successfully posted for exercise ID {exercise_id}")
        else:
            logger.error(f"Failed to post metrics for exercise ID {exercise_id}. Status code: {response.status_code}")
            logger.error("Response: " + response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error posting results: {e}")

# Main logic to run scenario
def runScenario(queueData):

    init_mqtt_client()

    logging.basicConfig(level=logging.INFO)
    client.publish('STARTVC', 'STARTVC')
    time.sleep(4)
    
    # Stop recording after data collection is done
    client.publish('StopRecording', 'StopRecording')
    time.sleep(2)
    print("Waiting for app to connect...")
    while not app_connected.value:
        time.sleep(1)
    print("App connected, continuing...")

    try:
        set_language("EN")
    except Exception as e:
        print(f"Language selection failed{e}")
        return

    try:
        metrics_queue = mp.Queue()
        #polar_queue = mp.Queue()
        logger.info('Running scenario...')
        
        while True:
            devices = get_devices() #Get the device set from the endpoint
            # Initialize variables for IMU serial numbers
            imu_serials = {}

            # Extract serial numbers based on IMU names
            for device in devices[0]['devices']:
                name = device['name']
                serial_number = device['serialNumber']
                imu_serials[name] = serial_number

            # Assign each to a variable if needed
            imu_head = imu_serials.get('imu-one')
            imu_pelvis = imu_serials.get('imu-two')
            imu_left = imu_serials.get('imu-three')
            imu_right = imu_serials.get('imu-four')

            # Fetch the daily schedule
            exercises = get_daily_schedule()
            exercises = reorder_exercises(exercises)
            print("Get list",exercises)
            if not isinstance(exercises, list) or not exercises:
                send_exit()
                break                    

            # Process each exercise in the schedule
            for exercise in exercises:
                logger.info(f"Processing Exercise ID: {exercise['exerciseId']}")
                
                if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                    try:
                        start_exercise_demo(exercise)
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                elif exercise["exerciseId"] in [28,29,30,31,32,33,34,35,36]:
                    try:
                        start_exergames(exercise)
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                else:
                    try:
                        results = start_cognitive_games(exercise)
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                    if results:
                        metrics = results
                        post_results(json.dumps(metrics), exercise['exerciseId'])
                    else:
                        logger.warning("No results returned from cognitive game.")
                
                # Determine the config message based on exercise ID
                if exercise['exerciseId'] == 1 or exercise['exerciseId'] == 28:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_01"
                elif exercise['exerciseId'] == 2 or exercise['exerciseId'] == 29:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_02"
                elif exercise['exerciseId'] == 3 :
                    if exercise['progression'] == 0 or exercise['progression'] == 1:
                        config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_03"
                    else:
                        config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_25" #Side Bend Here
                elif exercise['exerciseId'] == 4 or exercise['exerciseId'] == 30:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_09"
                elif exercise['exerciseId'] == 5:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_10"
                elif exercise['exerciseId'] == 6 or exercise['exerciseId'] == 31:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_11"
                elif exercise['exerciseId'] == 7:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_12"
                elif exercise['exerciseId'] == 8 or exercise['exerciseId'] == 33:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_16"
                elif exercise['exerciseId'] == 9 or exercise['exerciseId'] == 34:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_17"
                elif exercise['exerciseId'] == 10 or exercise['exerciseId'] == 35 or exercise['exerciseId'] == 36:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_18"
                elif exercise['exerciseId'] == 11:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_21"
                elif exercise['exerciseId'] == 12:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_22"
                elif exercise['exerciseId'] == 13:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_23"
                elif exercise['exerciseId'] == 14:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_04"
                elif exercise['exerciseId'] == 15:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_05"
                elif exercise['exerciseId'] == 16:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_06"
                elif exercise['exerciseId'] == 17:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_07"
                elif exercise['exerciseId'] == 18:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_08"
                elif exercise['exerciseId'] == 19:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_13"
                elif exercise['exerciseId'] == 20:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_14"
                elif exercise['exerciseId'] == 21:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_15"
                elif exercise['exerciseId'] == 22:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_19"
                elif exercise['exerciseId'] == 23:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_20"
                elif exercise["exerciseId"] == 43 or exercise['exerciseId'] == 32:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_24"
                else:
                    logger.warning(f"No config message found for Exercise ID: {exercise['exerciseId']}")
                    
                
                # Publish configuration and start the exercise
                topic = "IMUsettings"

                # Publish the configuration message to start the exercise
                print('--- Starting the exercise ---')
                if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,43]:
                    client.publish("IMUsettings", config_message)
                    time.sleep(2)
                    client.publish('StartRecording', 'start')
                    # Start the scheduler process
                    scheduler_process = mp.Process(target=scheduler, args=(scheduleQueue,))
                    scheduler_process.start()

                    #Start the process to receive Polar data
                    # polar_proc = mp.Process(target=start_ble_process, args=(0, polar_queue))  # Adjust adapter index if needed
                    # polar_proc.start()

                    # Wait for Polar connection or failure
                    time.sleep(5)  # Give some time to attempt connection

                    # Start the process to receive IMU data
                    imu_process = mp.Process(
                        target=receive_imu_data,
                        args=(queueData, scheduleQueue, config_message, exercise,metrics_queue,)
                    )

                    imu_process.start()

                    # Wait for the IMU process to finish
                    imu_process.join()

                    data_zip_path = None
                    while not scheduleQueue.empty():
                        msg = scheduleQueue.get()
                        if isinstance(msg, tuple) and msg[0] == "data_zip":
                            data_zip_path = msg[1]
                            break

                    if data_zip_path:
                        upload_file(data_zip_path, "Data")
                    # Terminate the scheduler process
                    scheduler_process.terminate()
                    scheduler_process.join()

                    # Stop recording after data collection is done
                    client.publish('StopRecording', 'StopRecording')
                    time.sleep(2)

                    # Check if Polar connection failed early
                    # if not polar_proc.is_alive() or (not polar_queue.empty() and polar_queue.get() is None):
                    #     logging.warning("Polar H10 is not connected. Proceeding without heart rate data.")
                    #     polar_proc.terminate()
                    #     polar_proc.join()
                    
                    # polar_data = []
                    # while not polar_queue.empty():
                    #     polar_data.append(polar_queue.get())
                    
                    # Post metrics after the exercise ends`

                else:
                    print("###cognitive###")
                
                if not metrics_queue.empty():
                    metrics = metrics_queue.get()
                    print(metrics)
                    try:
                        metrics = json.loads(metrics) if isinstance(metrics, str) else metrics
                    except json.JSONDecodeError:
                        print("Metrics could not be parsed as JSON.")
                        return
                    print(f"Metrics for Exercise {exercise['exerciseId']}: {metrics}")

                    #Post the results
                    #metrics["polar_data"] = polar_data
                    if (exercise["isFirstSession"])== True :
                        try:
                    # Combine sending voice instruction and waiting for response
                            symptomps_response = send_message_with_speech_to_text("bph0101")
                        except Exception as e:
                            logger.error(f"Failed to send voice instruction or get response for Exercise ID {exercise['exerciseId']}: {e}")
                            return
                        
                        #If the answer is no ask the patient to move in another exercise
                        if symptomps_response == "no":
                            # Ask if wanna to move to another exercise
                            metrics["symptoms"] = {"symptom_check": "no"}

                        else:
                            # Create a new key in metrics for symptoms
                            metrics["symptoms"] = {}

                            # Ask about headache
                            try:
                                headache_response = send_message_with_speech_to_text("bph0077")
                                metrics["symptoms"]["headache"] = {"present": headache_response}
                                if headache_response == "yes":
                                    rate_headache = send_message_with_speech_to_text_2("bph0110")
                                    metrics["symptoms"]["headache"]["severity"] = rate_headache
                            except Exception as e:
                                logger.error("Error getting headache response: %s", e)

                            # Ask about disorientation
                            try:
                                disorientated_response = send_message_with_speech_to_text("bph0087")
                                metrics["symptoms"]["disorientated"] = {"present": disorientated_response}
                                if disorientated_response == "yes":
                                    rate_disorientated = send_message_with_speech_to_text_2("bph0110")
                                    metrics["symptoms"]["disorientated"]["severity"] = rate_disorientated
                            except Exception as e:
                                logger.error("Error getting disorientation response: %s", e)

                            # Ask about blurry vision
                            try:
                                blurry_vision_response = send_message_with_speech_to_text("bph0089")
                                metrics["symptoms"]["blurry_vision"] = {"present": blurry_vision_response}
                                if blurry_vision_response == "yes":
                                    rate_blurry_vision = send_message_with_speech_to_text_2("bph0110")
                                    metrics["symptoms"]["blurry_vision"]["severity"] = rate_blurry_vision
                            except Exception as e:
                                logger.error("Error getting blurry vision response: %s", e)
                                send_voice_instructions("bph")
                                break;
                    
                    score = give_score(metrics, exercise['exerciseId']) 
                    metrics["score"] = score
                    post_results(json.dumps(metrics), exercise['exerciseId'])
                   
                else:
                    try:
                        metrics = {"metrics": ["ERROR IN METRICS", "ERROR IN METRICS", "ERROR IN METRICS"]}
                        post_results(json.dumps(metrics), exercise['exerciseId'])
                    except json.JSONDecodeError:
                        print("Metrics_2 could not be parsed as JSON.")
                        return
                    
                
                
                # Mark the exercise as completed
                print(f"Exercise {exercise['exerciseName']} completed.")
                time.sleep(10)

                # Fetch updated schedule after processing current exercises
                exercises = get_daily_schedule()
                ###If there are no exercises then end the 
                if not isinstance(exercises, list) or not exercises:
                     # Close all handlers and rename the log
                    txt_file_path = convert_log_to_txt(log_filename)
                    upload_file(txt_file_path, "Logs")

                    for handler in logger.handlers[:]:
                        handler.flush()
                        handler.close()
                        logger.removeHandler(handler)
                    if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                        send_voice_instructions("bph0222")
                        send_voice_instructions("bph0108")
                    else :
                        send_voice_instructions_ctg("bph0222")
                        send_voice_instructions_ctg("bph0108")
                    client.publish('EXIT','EXIT')
                    send_exit()
                    break;
                try:
                    time.sleep(2)
                    if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                        response = send_message_with_speech_to_text("bph0088")
                    else :
                        response = send_message_with_speech_to_text_ctg("bph0088")
                except Exception as e:
                    logger.error(f"Failed to send voice instruction or get response for Exercise ID {exercise['exerciseId']}: {e}")
                    return

                if response == "no":
                    # Close all handlers and rename the log

                    txt_file_path = convert_log_to_txt(log_filename)
                    upload_file(txt_file_path, "Logs")

                    for handler in logger.handlers[:]:
                        handler.flush()
                        handler.close()
                        logger.removeHandler(handler)
                    
                    print("User chose to stop. Exiting scenario.")
                    if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                        send_voice_instructions("bph0222")
                        send_voice_instructions("bph0108")
                    else:
                        send_voice_instructions_ctg("bph0222")
                        send_voice_instructions_ctg("bph0108")
                    client.publish('EXIT','EXIT')
                    send_exit()
                    return
                elif response == "yes":
                    print("User chose to continue. Proceeding with next exercise.")
                    if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                        send_voice_instructions("bph0045")
                    else:
                        send_voice_instructions_ctg("bph0045")
                    continue
            

    except requests.exceptions.RequestException as e:
        logger.error(f"Error: {e}")

# MQTT setup
def on_connect(client, userdata, flags, rc):
    logger.info("Connected to MQTT broker with result code " + str(rc))
    client.subscribe("IMUsettings")
    client.subscribe("DeviceStatus")
    client.subscribe("StopRecording")
    client.subscribe("TerminationTopic")
    client.subscribe("STARTVC")
    client.subscribe("EXIT")

def on_message(client, userdata, msg):
    logger.info(f"Message Received -> {msg.payload.decode()}")

# Start MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)

client_process = mp.Process(target=SendMyIP, args=())
client_process.start()

# Publish loop
def publish_loop():
    topic = "IMUsettings"
    while True:
        time.sleep(1)

# Start necessary processes
if enableConnectionToAPI:
    login()
    get_device_api_key()

server_process = mp.Process(target=start_multicast_server, args=(queueData,))
server_process.start()

# iamalive_process = mp.Process(target=checkIAMALIVE, args=(iamalive_queue,))
# iamalive_process.start()

thread = threading.Thread(target=publish_loop)
# thread.start()

received_response = mp.Value('b', 0)  # Shared flag to track responses
listener_process = mp.Process(target=start_mqtt_listener)
listener_process.start()


threadscenario = threading.Thread(target=runScenario, args=(queueData,))
threadscenario.start()

#send_heartbeat()
client.loop_forever()


#client.loop_start()
#send_heartbeat()


threadscenario.join()




