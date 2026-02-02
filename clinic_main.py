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
from mqtt_messages import init_mqtt_client, set_language, start_exercise_demo, send_voice_instructions,send_message_with_speech_to_text,send_message_with_speech_to_text_2,send_exit,start_cognitive_games,start_exergames,send_message_with_speech_to_text_ctg,send_message_with_speech_to_text_ctg_2,send_voice_instructions_ctg,app_connected,start_video,stop_video, reset_global_flags,POLAR_TOPIC,ACK_TOPIC
from data_management_v05 import scheduler, receive_imu_data
from api_management import login, get_device_api_key
from scoring import give_score_AI
from configure_file_management import read_configure_file
#from Polar_test import start_ble_process 
from shared_variables import (
    queueData, scheduleQueue, enableConnectionToAPI,
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL,
    mqttState, DEBUG
)
from UDPClient2 import broadcast_ip
from UDPSERVER import start_multicast_server
from UDPSERVER import start_unicast_server
from websocketServer import run_websocket_server
from pathlib import Path
from mqtt_messages import ctg_queue
import re
MAC_RE = re.compile(r'[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}')

def clean_mac(x):
    if not x:
        return None
    m = MAC_RE.search(x)
    return m.group(0).upper() if m else None

# Get the directory where the script is located at
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH_2 = BASE_DIR / 'clinic.ini'
# Load config
config = configparser.ConfigParser()
config.read(CONFIG_PATH_2)
# Get clinic_id and strip quotes
clinic_id = config.get("CLINIC", "clinic_id").strip('"')

DEMO_TOPIC = f"exercise@{clinic_id}/demo"
MSG_TOPIC = f"exercise@{clinic_id}/msg"
EXIT_TOPIC = f"exercise@{clinic_id}/exit"


# Construct the paths for config and logo
CONFIG_PATH = BASE_DIR / 'config.ini'
TOPIC_PING = f"healthcheck@{clinic_id}/AREYOUALIVE"
TOPIC_PONG = f"healthcheck@{clinic_id}/IAMALIVE"

camera_result = mp.Manager().dict()
polar_result = mp.Manager().dict()


def start_broadcast_process():
    p = mp.Process(target=broadcast_ip)
    p.start()
    return p

def reorder_exercises2(session):
    group_definitions = {
        "Stretching": [11, 12, 13],
        "Sitting Exercises": [1, 2, 3, 14, 15, 16, 17, 18],
        "Standing Exercises": [4, 5, 6, 7,  19, 20, 21, 43],
        "Walking Exercises": [8, 9, 10, 22],
        "Optokinetic Exercises": [24, 25, 26, 27],
        "Exergames": [28, 29, 30, 31, 32, 33, 34, 35, 36],
        "Cognitive Games": [37, 38, 39, 40, 41, 42]
    }

    # Initialize ordered buckets
    grouped = {
        "Stretching": [],
        "Sitting Exercises": [],
        "Standing Exercises": [],
        "Walking Exercises": [],
        "Optokinetic Exercises": [],
        "Exergames": [],
        "Cognitive Games": [],
    }

    # Track assigned IDs to avoid duplicates
    assigned_ids = set()

    # Process each exercise in input order
    for ex in session:
        eid = ex["exerciseId"]
        for group_name, id_list in group_definitions.items():
            if eid in id_list and eid not in assigned_ids:
                grouped[group_name].append(ex)
                assigned_ids.add(eid)
                break

    # Return in desired group order
    ordered_session = []
    for group_name in grouped:
        ordered_session.extend(grouped[group_name])
    return ordered_session


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

# Define file storage API endpoints
FILE_STORAGE_BASE_URL = "https://telerehab.biomed.ntua.gr/filestorage"
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
    
    url = 'http://telerehab.biomed.ntua.gr/api/PatientDeviceSet'
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
    
    url = 'http://telerehab.biomed.ntua.gr/api/PatientSchedule/daily'
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
        url = "https://telerehab.biomed.ntua.gr/api/PerformanceScore"
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

        if (not DEBUG):
            response = requests.post(url, json=post_data, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Metrics successfully posted for exercise ID {exercise_id}")
            else:
                logger.error(f"Failed to post metrics for exercise ID {exercise_id}. Status code: {response.status_code}")
                logger.error("Response: " + response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error posting results: {e}")

def get_patient_id():
    """Fetch the daily schedule from the API."""
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    api_key_edge = config['API'].get('key_edge', '')
    
    url = 'http://telerehab.biomed.ntua.gr/api/PatientDeviceSet/Check'
    headers = {
        'accept': '*/*',
        'Authorization': api_key_edge
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("patientId")  # ✅ just the int
    else:
        response.raise_for_status()

def post_patient_ip(mac_address):
    patient_id = get_patient_id()
    if patient_id is None:
        raise ValueError("Could not retrieve patientId")

    url = 'https://telerehab.biomed.ntua.gr/api/PatientDeviceSetIp'
    headers = {
        'accept': '*/*',
        'Authorization': "1IeT76UWcgAkA9SvOkjcy9nsVxpJXrQVNPMQevkDEfUU",
        'Content-Type': 'application/json'
    }
    payload = {
        'patientId': patient_id,
        'data': mac_address
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print("MAC address posted successfully.")
    else:
        print(f"Error posting MAC: {response.status_code}, {response.text}")
        response.raise_for_status()

def mqtt_heartbeat_server(clinic_id, broker, port=1883):
    PING_TOPIC = f"heartbeat/{clinic_id}/ping"
    ACK_TOPIC = f"heartbeat/{clinic_id}/ack"

    def on_connect(client, userdata, flags, rc):
        print("Connected to hardbeat server with result code", rc)
        client.subscribe(PING_TOPIC)

    def on_message(client, userdata, msg):
        payload = msg.payload.decode()
        print(f"Received heartbeat on {msg.topic}: {msg.payload.decode()}")
        if msg.topic == PING_TOPIC:
            client.publish(ACK_TOPIC, payload="ack", qos=1)


    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port, 60)
    client.loop_forever()


def start_heartbeat_process():
    p = mp.Process(target=mqtt_heartbeat_server, args=(clinic_id, MQTT_BROKER_HOST, MQTT_BROKER_PORT))
    p.start()
    print("Heartbeat server process started.")
    return p


# Main logic to run scenario
def runScenario(queueData):

    client = init_mqtt_client(name = "client_runScenario")
    post_patient_ip(MQTT_BROKER_HOST)
    logging.basicConfig(level=logging.INFO)

    client.publish(ACK_TOPIC, payload="ack", qos=1)

    client.publish(f'TELEREHAB@{clinic_id}/STARTVC', 'STARTVC', qos=1, retain=False)
    time.sleep(4)
    client.publish(ACK_TOPIC, payload="ack", qos=1)


    # Stop recording after data collection is done
    client.publish(f'TELEREHAB@{clinic_id}/StopRecording', 'STOP_RECORDING')
    time.sleep(2)
    print("Waiting for app to connect...")
    t0 = time.time()
    
    while not app_connected.value:
        if time.time() - t0 > 40:
            logger.error("Timeout waiting for app connection. Exiting.")
            send_exit(client)
            return
        time.sleep(2)  # slower => less spam
        client.publish(f'TELEREHAB@{clinic_id}/STARTVC', 'STARTVC', qos=1, retain=False)
    
    print("App connected, continuing...")

    try:
        time.sleep(2)
        set_language(client, "EN")
    except Exception as e:
        print(f"Language selection failed{e}")
        return

    try:
        metrics_queue = mp.Queue()
        #polar_queue = mp.Queue()
        logger.info('Running scenario...')
        
        while True:
            devices = get_devices() 

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

            imu_head   = clean_mac(imu_head)
            imu_pelvis = clean_mac(imu_pelvis)
            imu_left   = clean_mac(imu_left)
            imu_right  = clean_mac(imu_right)

            print(imu_head)
            print(imu_pelvis)
            print(imu_left)
            print(imu_right)
            # Fetch the daily schedule
            exercises = get_daily_schedule()
            print('--- Daily Schedule --- original order')
            for ex in exercises:
                print(ex["exerciseId"], ex["description"])
            exercises = reorder_exercises2(exercises)

            print('--- Daily Schedule --- reordered order')
            for ex in exercises:
                print(ex["exerciseId"], ex["description"])

            if not isinstance(exercises, list) or not exercises:
                send_exit(client)
                break

            # Process each exercise in the schedule
            for exercise in exercises:

                print('********************STEP 01***********************')

                reset_global_flags()

                logger.info(f"Processing Exercise ID: {exercise['exerciseId']}")
                
                # Determine the config message based on exercise ID
                config_message = None
                if exercise['exerciseId'] == 1 :
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_01"
                elif exercise['exerciseId'] == 2 :
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_02"
                elif exercise['exerciseId'] == 3 :
                    if exercise['progression'] == 0 or exercise['progression'] == 1:
                        config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_03"
                    else:
                        config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_25" #Side Bend Here
                elif exercise['exerciseId'] == 4:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_09"
                elif exercise['exerciseId'] == 5:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_10"
                elif exercise['exerciseId'] == 6 :
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_11"
                elif exercise['exerciseId'] == 7:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-QUATERNIONS,RIGHTFOOT={imu_right}-QUATERNIONS,exer_12"
                elif exercise['exerciseId'] == 8 :
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_16"
                elif exercise['exerciseId'] == 9 :
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_17"
                elif exercise['exerciseId'] == 10 :
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
                elif exercise["exerciseId"] == 43 :
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_24"
                elif exercise['exerciseId'] == 28:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_28"
                elif exercise['exerciseId'] == 29:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-OFF,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_29"
                elif exercise['exerciseId'] == 30:
                    config_message = f"HEAD={imu_head}-OFF,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_30"
                elif exercise['exerciseId'] == 31:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_31"
                elif exercise['exerciseId'] == 32:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-OFF,RIGHTFOOT={imu_right}-OFF,exer_32"
                elif exercise['exerciseId'] == 33:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_33"
                elif exercise['exerciseId'] == 34:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_34"
                elif exercise['exerciseId'] == 35 or exercise['exerciseId'] == 36:
                    config_message = f"HEAD={imu_head}-QUATERNIONS,PELVIS={imu_pelvis}-QUATERNIONS,LEFTFOOT={imu_left}-LINEARACCELERATION,RIGHTFOOT={imu_right}-LINEARACCELERATION,exer_35"
                
                if config_message:
                    client.publish(f"TELEREHAB@{clinic_id}/IMUsettings", config_message) #Changed 20251031,this was done after DEMO_END
                    time.sleep(4)
                    client.publish(f'TELEREHAB@{clinic_id}/StartRecording', 'START_RECORDING')

                if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,43]:
                    try:
                        start_exercise_demo(client, exercise)

                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                elif exercise["exerciseId"] in [28,29,30,31,32,33,34,35,36]:
                    try:
                        start_exergames(client, exercise)
                        client.publish("CAMERA_START", "start")
                        time.sleep(7)
                        client.publish("CAMERA_STOP", "stop")
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                elif exercise["exerciseId"] in [24,25,26,27]:
                    try:
                        start_exercise_demo(client, exercise)
                        client.publish("CAMERA_START", "start")
                        time.sleep(7)
                        client.publish("CAMERA_STOP", "stop")
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                    time.sleep(5)
                    send_voice_instructions(client, "bph0082")
                    try:
                        start_video(client, exercise)
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                    time.sleep(30)
                    try:
                        stop_video(client, exercise)
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue    
                else:
                    try:
                        results = start_cognitive_games(client, exercise)
                    except Exception as e:
                        logger.error(f"Demonstration failed for Exercise ID {exercise['exerciseId']}: {e}")
                        continue
                    if results:
                        metrics = results
                        post_results(json.dumps(metrics), exercise['exerciseId'])
                    else:
                        metrics = {"metrics": ["ERROR IN METRICS", "ERROR IN METRICS", "ERROR IN METRICS"]}
                        post_results(json.dumps(metrics), exercise['exerciseId'])
                        logger.warning("No results returned from cognitive game.")

                print('********************STEP 02***********************')

                # Publish configuration and start the exercise
                #topic = f"TELEREHAB@{clinic_id}/IMUsettings"
                print('********************STEP 03***********************')

                # Publish the configuration message to start the exercise
                print('--- Starting the exercise ---')
                if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28,29,30,31,32,33,34,35,36,43]:
                    client.publish(POLAR_TOPIC, "start")
                    # Start the scheduler process
                    try:
                        time.sleep(1)
                        scheduler_process = mp.Process(target=scheduler, args=(scheduleQueue,))
                        scheduler_process.start()
                    except:
                            print(f"!!! FATAL ERROR in clinic_main: {e}")

                    print('********************STEP 04***********************')

                    # Wait for Polar connection or failure
                    #time.sleep(5)  # Give some time to attempt connection

                    # Start the process to receive IMU data
                    imu_process = mp.Process(
                        target=receive_imu_data,
                        args=(queueData, scheduleQueue, config_message, exercise,metrics_queue,ctg_queue,)
                    )

                    imu_process.start()
                    # Wait for the IMU process to finish
                    imu_process.join()

                    print('********************STEP 05***********************')

                    client.publish(POLAR_TOPIC, "stop")
                    data_zip_path = None
                    while not scheduleQueue.empty():
                        msg = scheduleQueue.get()
                        if isinstance(msg, tuple) and msg[0] == "data_zip":
                            data_zip_path = msg[1]
                            break
                    if data_zip_path:
                        upload_file(data_zip_path, "Data")
                    # Terminate the scheduler process
                    
                    scheduleQueue.put("EXIT_NO_DATA");
                    scheduler_process.terminate()
                    scheduler_process.join()
                    print('********************STEP 06***********************')

                    # Stop recording after data collection is done
                    client.publish(f'TELEREHAB@{clinic_id}/StopRecording', 'STOP_RECORDING')
                    time.sleep(2)


                else:
                    print("###cognitive###")
                while not scheduleQueue.empty():
                        msg = scheduleQueue.get()
                response = 'X';
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
                    
                    if (exercise["isFirstSession"])== True and exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43] :
                        try:
                    # Combine sending voice instruction and waiting for response
                            symptomps_response = send_message_with_speech_to_text(client, "bph0101")
                        except Exception as e:
                            logger.error(f"Failed to send voice instruction or get response for Exercise ID {exercise['exerciseId']}: {e}")
                            return
                        
                        #If the answer is no ask the patient to move in another exercise
                        if symptomps_response == "no":
                            # Ask if wanna to move to another exercise
                            metrics["symptoms"] = {"symptom_check": "no"}

                        elif symptomps_response == "yes":
                            # Create a new key in metrics for symptoms
                            metrics["symptoms"] = {}

                            # Ask about headache
                            try:
                                headache_response = send_message_with_speech_to_text(client, "bph0077")
                                metrics["symptoms"]["headache"] = {"present": headache_response}
                                if headache_response == "yes":
                                    rate_headache = send_message_with_speech_to_text_2(client, "bph0091")
                                    metrics["symptoms"]["headache"]["severity"] = rate_headache
                            except Exception as e:
                                logger.error("Error getting headache response: %s", e)

                            # Ask about disorientation
                            try:
                                disorientated_response = send_message_with_speech_to_text(client, "bph0087")
                                metrics["symptoms"]["disorientated"] = {"present": disorientated_response}
                                #if disorientated_response == "yes":
                                #    rate_disorientated = send_message_with_speech_to_text_2(client, "bph0110")
                                #   metrics["symptoms"]["disorientated"]["severity"] = rate_disorientated
                            except Exception as e:
                                logger.error("Error getting disorientation response: %s", e)

                            # Ask about blurry vision
                            try:
                                blurry_vision_response = send_message_with_speech_to_text(client, "bph0089")
                                metrics["symptoms"]["blurry_vision"] = {"present": blurry_vision_response}
                                if blurry_vision_response == "yes":
                                    rate_blurry_vision = send_message_with_speech_to_text_2(client, "bph0092")
                                    metrics["symptoms"]["blurry_vision"]["severity"] = rate_blurry_vision
                            except Exception as e:
                                logger.error("Error getting blurry vision response: %s", e)
                                break;
                        else: #response = APPKILLED
                            metrics["symptoms"] = {"symptom_check": "novirtualcoach"}

                        response = symptomps_response;
                    score = give_score_AI(metrics, exercise['exerciseId']) 
                    print(score)
                    print(metrics)
                    metrics["score"] = score
                    
                    try:
                        with open("posture_results.txt", "r") as f:
                            camera_metrics = f.read()
                            print(camera_metrics)
                            metrics["camera"] = camera_metrics
                    except Exception as e:
                        logger.warning(f"Camera data not available: {e}")
                        metrics["camera"] = {"error": "Camera data not available"}

                    try:
                        with open("polar_results.txt", "r") as f:
                            polar_metrics = f.read()
                            print(polar_metrics)
                            metrics["polar_data"] = polar_metrics
                    except Exception as e:
                        logger.warning(f"Polar data not available: {e}")
                        metrics["polar_data"] = {"error": "Polar data not available"}
                    post_results(json.dumps(metrics), exercise['exerciseId'])
                else:
                    try:
                        metrics = {"metrics": ["ERROR IN METRICS", "ERROR IN METRICS", "ERROR IN METRICS"],
                                   "score" : "-1"}
                        post_results(json.dumps(metrics), exercise['exerciseId'])
                    except json.JSONDecodeError:
                        print("Metrics_2 could not be parsed as JSON.")
                        return
                
                # Mark the exercise as completed
                print(f"Exercise {exercise['exerciseName']} completed.")
                time.sleep(1)
                print('********************STEP 07***********************')

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
                        if (response != "APPKILLED"):
                            send_voice_instructions(client, "bph0222")
                            send_voice_instructions(client, "bph0108")
                    else:
                        if (response != "APPKILLED"):
                            send_voice_instructions_ctg(client, "bph0222")
                            send_voice_instructions_ctg(client, "bph0108")
                    client.publish(f'TELEREHAB@{clinic_id}/EXIT','EXIT')
                    client.publish(f'camera@{clinic_id}','CAMERAOUT')
                    client.publish(f'polar@{clinic_id}','polarout')
                    send_exit(client)
                    break;
                try:
                    time.sleep(2)
                    if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                        if (response != "APPKILLED"):
                            response = send_message_with_speech_to_text(client, "bph0088")
                    else:
                        if (response != "APPKILLED"):
                            response = send_message_with_speech_to_text_ctg(client, "bph0088")
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
                        send_voice_instructions(client, "bph0222")
                        send_voice_instructions(client, "bph0108")
                    else:
                        send_voice_instructions_ctg(client, "bph0222")
                        send_voice_instructions_ctg(client, "bph0108")
                    client.publish(f'TELEREHAB@{clinic_id}/EXIT','EXIT')
                    client.publish(f'camera@{clinic_id}','CAMERAOUT')
                    client.publish(f'polar@{clinic_id}','polarout')
                    send_exit(client)
                    return
                elif response == "yes":
                    print("User chose to continue. Proceeding with next exercise.")
                    if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                        send_voice_instructions(client, "bph0045")
                    else:
                        send_voice_instructions_ctg(client, "bph0045")
                    continue
                elif response == "APPKILLED":
                    print('User killed the app. Proceeding with next exercise.');
                    #client.publish(f'TELEREHAB@{clinic_id}/STARTVC', 'STARTVC')
                    time.sleep(4)
                    
                    # Stop recording after data collection is done
                    #client.publish(f'TELEREHAB@{clinic_id}/StopRecording', 'STOP_RECORDING')
                    #time.sleep(2)
                    print("Waiting for app to connect after kill...")
                    while not app_connected.value:
                        time.sleep(1)
                    print("App connected, continuing...")
                    time.sleep(3)
                    try:
                        time.sleep(3)
                        set_language(client, "EN")
                    except Exception as e:
                        print(f"Language selection failed{e}")
                        return

                    print("User chose to kill the app and continue. Proceeding with next exercise.")
                    #if exercise["exerciseId"] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,43]:
                    #    send_voice_instructions(client, "bph0045")
                    #else:
                    #    send_voice_instructions_ctg(client, "bph0045")
                    continue                    
            

    except requests.exceptions.RequestException as e:
        logger.error(f"Error: {e}")

# MQTT setup
def on_connect(client, userdata, flags, rc):
    logger.info("Connected to MQTT broker with result code " + str(rc))
    client.subscribe(f"TELEREHAB@{clinic_id}/IMUsettings")
    client.subscribe(f"TELEREHAB@{clinic_id}/DeviceStatus")
    client.subscribe(f"TELEREHAB@{clinic_id}/StartRecording")
    client.subscribe(f"TELEREHAB@{clinic_id}/StopRecording")
    client.subscribe(f"TELEREHAB@{clinic_id}/TerminationTopic")
    client.subscribe(f"TELEREHAB@{clinic_id}/STARTVC")
    client.subscribe(f"TELEREHAB@{clinic_id}/EXIT")


def on_message(client, userdata, msg):
    payload_raw = msg.payload.decode().strip()
    logger.info(f"Message Received -> {msg.payload.decode()}")

    # ---- NEW: set app_connected από DeviceStatus ----
    if msg.topic == f"TELEREHAB@{clinic_id}/DeviceStatus":
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = payload_raw

        # Πιάσε και JSON και απλό string
        status = ""
        if isinstance(payload, dict):
            status = str(payload.get("status") or payload.get("action") or "").lower()
        else:
            status = str(payload).lower()

        if status in ("connected", "ready", "ok", "app_connected"):
            app_connected.value = True
            logger.info("App connection acknowledged via DeviceStatus -> app_connected = True")


#client_kill_VR = init_mqtt_client(name = "client_kill_VR")
#send_exit(client_kill_VR)

# Start MQTT client
client = mqtt.Client(client_id="client_main")
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)

process = start_broadcast_process()


# Publish loop
def publish_loop():
    topic = f"TELEREHAB@{clinic_id}/IMUsettings"
    while True:
        time.sleep(1)

# Start necessary processes
if enableConnectionToAPI:
    login()
    get_device_api_key()

#server_process = mp.Process(target=start_multicast_server, args=(queueData,))
server_process = mp.Process(target=start_unicast_server, args=(queueData,))
server_process.start()

received_response = mp.Value('b', 0)  # Shared flag to track responses

threadscenario = threading.Thread(target=runScenario, args=(queueData,))
threadscenario.start()


client.loop_forever()
#client.loop_start()

threadscenario.join()
#proc.join();
