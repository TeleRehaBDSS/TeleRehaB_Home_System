import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import FindMyIP as ip
import multiprocessing as mp
import configparser
from pathlib import Path
import subprocess
from shared_variables import (
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)

app_connected = mp.Value('b', False)
BASE_DIR = Path(__file__).resolve().parent
# Construct the paths for config and logo
CONFIG_PATH_2 = BASE_DIR / 'clinic.ini'
# Load config
config = configparser.ConfigParser()
config.read(CONFIG_PATH_2)
# Get clinic_id and strip quotes
clinic_id = config.get("CLINIC", "clinic_id").strip('"')

# MQTT Configuration
local_ip = ip.internal()
#MQTT_BROKER_HOST = local_ip
#MQTT_BROKER_HOST = 'broker.emqx.io'  # Public MQTT broker for testing
#MQTT_BROKER_PORT = 1883
#MQTT_KEEP_ALIVE_INTERVAL = 60

# Topics
DEMO_TOPIC = f"exercise@{clinic_id}/demo"
MSG_TOPIC = f"exercise@{clinic_id}/msg"
EXIT_TOPIC = f"exercise@{clinic_id}/exit"
IAMALIVETOPIC = "iamalive_topic"
CAMERA_TOPIC = f"camera@{clinic_id}"
POLAR_TOPIC = f"polar@{clinic_id}"
PING_TOPIC = f"heartbeat/{clinic_id}/ping"
ACK_TOPIC = f"heartbeat/{clinic_id}/ack"

# Global Flags
ack_received = False
demo_start_received = False
demo_end_received = False
finish_received = False
finish_response=None
ctg_results_received = False
ctg_results_data = None
video_stop = None
video_ack = None
iamalive_queue = mp.Queue()
ctg_queue = mp.Queue()

def wait_for_ctg_results(timeout=70):
    global ctg_results_received, ctg_results_data

    print("[wait_for_ctg_results] entered")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if ctg_results_received and ctg_results_data:
            try:
                print("CTG_RESULTS received, attempting to parse...")
                cleaned_data = ctg_results_data.replace('\\n', '').replace('\\"', '"')
                parsed_data = json.loads(cleaned_data)
                if isinstance(parsed_data, str):
                    parsed_data = json.loads(parsed_data)
                print("Parsing successful!")
                return parsed_data
            except Exception as e:
                print(f"Failed to parse CTG results: {e}")
                return None
        time.sleep(0.1)

    print("Timeout waiting for CTG_RESULTS.")
    return None


# Reset all global flags and responses
def reset_global_flags():
    global ack_received, demo_start_received, demo_end_received, finish_received, finish_response,ctg_received,video_stop,video_ack, ctg_results_data, ctg_results_received,ack_ctg_received
    ack_received = False
    demo_start_received = False
    demo_end_received = False
    finish_received = False  
    finish_response = None
    ctg_received = False
    video_stop = False
    video_ack = False
    ctg_results_data = False
    ctg_results_received = False
    ack_ctg_received = False

def reset_ctg():
    global ctg_results_data, ctg_results_received,ack_ctg_received
    ctg_results_data = False
    ctg_results_received = False
    ack_ctg_received = False

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    #client.subscribe([(DEMO_TOPIC, 0), (MSG_TOPIC, 0), (EXIT_TOPIC,0),(CAMERA_TOPIC,0),(POLAR_TOPIC,0),(PING_TOPIC,0),(ACK_TOPIC,0)])
    client.subscribe([(DEMO_TOPIC, 0), (MSG_TOPIC, 0), (EXIT_TOPIC,0),(CAMERA_TOPIC,0),(POLAR_TOPIC,0)])


def on_message(client, userdata, msg):
    global ack_received, demo_start_received, demo_end_received, finish_received,finish_response,ctg_received,ctg_results_received,ctg_results_data,ack_ctg_received,video_ack,video_stop
    #reset_global_flags()

    try:
        payload = json.loads(msg.payload.decode())  # Try to parse JSON
        print(f"Received JSON message on {msg.topic}: {payload}")
        #print("finish_received =",  finish_received)
        #print('--------------------------------------------------------------------')
    except json.JSONDecodeError:
        payload = msg.payload.decode()  # If not JSON, use raw string
        print(f"Received non-JSON message on {msg.topic}: {payload}")

    print(f"Received message on {msg.topic}: {payload}")


    if msg.topic == EXIT_TOPIC:
        if isinstance(payload, dict) and payload.get("action") == "ACK":
            print("EXIT acknowledged. Powering off the system...")
            try:
                subprocess.run(["sudo", "systemctl", "poweroff"], check=False)
            except Exception as e:
                print(f"[ERROR] Shutdown failed: {e}")

            
    elif msg.topic == DEMO_TOPIC:
        if isinstance(payload, dict) and payload.get("action") == "Connected":
            print("App has connected.")
            app_connected.value = True
        if payload.get("action") == "ACK":
            ack_received = True
        if payload.get("action") == "ACK_CTG":
            ack_ctg_received = True
        elif payload.get("action") == "DEMO_START":
            demo_start_received = True
        elif payload.get("action") == "DEMO_END":
            demo_end_received = True
        elif payload.get("action") == "FINISH":
            finish_received = True
        elif payload.get("action") == "CTG_END":
            ctg_results_received = True # needs to be removed when cognitive are fixed
            ctg_received = True
            ctg_queue.put('OK')
        elif payload.get("action") == "CTG_RESULTS":
            ctg_results_received = True
            ctg_results_data = payload.get("message")
    elif msg.topic == MSG_TOPIC:
        if payload.get("action") == "ACK_VIDEO":
            #print("GOT ACK")
            video_ack = True
        if payload.get("action") == "VIDEO_END":
            video_stop = True
        if payload.get("action") == "ACK":
            ack_received = True
        elif payload.get("action") == "FINISH":
            finish_received = True
        elif payload.get("action") == "FINISH_RESPONSE":
            finish_response = payload.get("message", "").lower()
        elif payload.get("action") == "APPKILLED":
            finish_response = 'APPKILLED';
    elif msg.topic == IAMALIVETOPIC:
        iamalive_queue.put('OK')


# Publish and Wait
def publish_and_wait(client, topic, message, timeout=5000, wait_for="ACK"):
    global ack_received, demo_start_received, demo_end_received, finish_received, ctg_received, ctg_results_received,ack_ctg_received,video_ack,video_stop
    #reset_global_flags()

    #ack_received = demo_start_received = demo_end_received = finish_received = ctg_received = video_ack= video_stop =  False 
    client.publish(topic, json.dumps(message))
    print(f"Published: {message} to {topic}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        if wait_for == "ACK" and ack_received:
            ack_received = False;
            return True
        if wait_for == "ACK_CTG" and ack_ctg_received:
            ack_ctg_received = False;
            return True
        if wait_for == "DEMO_START" and demo_start_received:
            demo_start_received = False;
            return True
        if wait_for == "DEMO_END" and demo_end_received:
            demo_end_received = False;
            return True
        if wait_for == "FINISH" and finish_received:
            finish_received = False;
            return True
        if wait_for == "CTG_END" and ctg_received:
            ctg_received = False;
            return True
        if wait_for == "CTG_RESULTS" and ctg_results_received:
            ctg_results_received = False;
            return True
        if wait_for == "ACK_VIDEO" and video_ack:
            video_ack = False;
            return True
        if wait_for == "VIDEO_END" and video_stop:
            video_stop = False;
            return True
        time.sleep(0.5)
    print(f"Timeout waiting for {wait_for} on message: {message}")
    return False


# Initialize MQTT Client
def init_mqtt_client():
    #global client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)
    client.loop_start()
    print("MQTT client initialized and connected to broker.")
    return client


# Function for Language Selection
def set_language(client, language):
    language_message = {
        "action": "LANGUAGE",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": "",
        "message":"",
        "language":language
    }
    while not publish_and_wait(client, DEMO_TOPIC, language_message, wait_for="FINISH"):
        print("Retrying LANGUAGE message...")
    print(f"Language set to {language}")

#{'exerciseName': 'ex1', 'progression': 0, 'exerciseId': 1, 'weekNumber': 47, 'year': 2024}
# Function for Exercise Demonstration



def send_exit(client):
    exit_message = {
        "action": "EXIT",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": "",
        "message": "",
        "language": "/"
    }
    client.publish(EXIT_TOPIC, json.dumps(exit_message))
    print(f"Published EXIT message: {exit_message}")

def start_exercise_demo(client, exercise):
    #reset_global_flags()

    if exercise['exerciseId'] == 1:
        exercise_name = f"VC holobalance_sitting_1 P{exercise['progression']}"
    elif exercise['exerciseId'] == 2:
        exercise_name = f"VC holobalance_sitting_2 P{exercise['progression']}"
    elif exercise['exerciseId'] == 3:
        exercise_name = f"VC holobalance_sitting_3 P{exercise['progression']}"
    elif exercise['exerciseId'] == 4:
        exercise_name = f"VC holobalance_standing_1 P{exercise['progression']}"
    elif exercise['exerciseId'] == 5:
        exercise_name = f"VC holobalance_standing_2 P{exercise['progression']}"
    elif exercise['exerciseId'] == 6:
        exercise_name = f"VC holobalance_standing_3 P{exercise['progression']}"
    elif exercise['exerciseId'] == 7:
        exercise_name = f"VC holobalance_standing_5 P{exercise['progression']}"
    elif exercise['exerciseId'] == 8:
        exercise_name = f"VC holobalance_walking_1 P{exercise['progression']}"
    elif exercise['exerciseId'] == 9:
        exercise_name = f"VC holobalance_walking_2 P{exercise['progression']}"
    elif exercise['exerciseId'] == 10:
        exercise_name = f"VC holobalance_walking_3 P{exercise['progression']}"
    elif exercise['exerciseId'] == 11:
        exercise_name = f"VC holobalance_stretching_1 P{exercise['progression']}"
    elif exercise['exerciseId'] == 12:
        exercise_name = f"VC holobalance_stretching_2 P{exercise['progression']}"
    elif exercise['exerciseId'] == 13:
        exercise_name = f"VC holobalance_stretching_3 P{exercise['progression']}"
    elif exercise['exerciseId'] == 14:
        exercise_name = f"VC holobalance_sitting_4 P{exercise['progression']}"
    elif exercise['exerciseId'] == 15:
        exercise_name = f"VC holobalance_sitting_5 P{exercise['progression']}"
    elif exercise['exerciseId'] == 16:
        exercise_name = f"VC holobalance_sitting_6 P{exercise['progression']}"
    elif exercise['exerciseId'] == 17:
        exercise_name = f"VC holobalance_sitting_7 P{exercise['progression']}"
    elif exercise['exerciseId'] == 18:
        exercise_name = f"VC holobalance_sitting_8 P{exercise['progression']}"
    elif exercise['exerciseId'] == 19:
        exercise_name = f"VC holobalance_standing_6 P{exercise['progression']}"
    elif exercise['exerciseId'] == 20:
        exercise_name = f"VC holobalance_standing_7 P{exercise['progression']}"
    elif exercise['exerciseId'] == 21:
        exercise_name = f"VC holobalance_standing_8 P{exercise['progression']}"
    elif exercise['exerciseId'] == 22:
        exercise_name = f"VC holobalance_walking_4 P{exercise['progression']}"
    elif exercise['exerciseId'] == 43:
        exercise_name = f"VC holobalance_standing_4 P{exercise['progression']}"
    elif exercise['exerciseId'] == 24:
        exercise_name = f"VC holobalance_Optokinetic_1 P{exercise['progression']}"
    elif exercise['exerciseId'] == 25:
        exercise_name = f"VC holobalance_Optokinetic_2 P{exercise['progression']}"
    elif exercise['exerciseId'] == 26:
        exercise_name = f"VC holobalance_Optokinetic_3 P{exercise['progression']}"
    elif exercise['exerciseId'] == 27:
        exercise_name = f"VC holobalance_Optokinetic_4 P{exercise['progression']}"


    demo_message = {
        "action": "START",
        "exercise": exercise_name,
        "timestamp": datetime.now().isoformat(),
        "code" : "",
        "message": "",
        "language":"/"
    }

    while not publish_and_wait(client, DEMO_TOPIC, demo_message, wait_for="DEMO_END"):
        print("Waiting for DEMO_END...")
    print(f"Exercise demonstration for {exercise_name} completed.")


def start_video(client,exercise):
    if exercise['exerciseId'] == 24:
         video_id=1
    elif exercise['exerciseId'] == 25:
         video_id=2
    elif exercise['exerciseId'] == 26:
         video_id=3
    elif exercise['exerciseId'] == 27:
        video_id=4
    start_video_message = {
        "action": "START_VIDEO",
        "exercise": "",
        "timestamp": datetime.now().isoformat(),
        "code" : "",
        "message": f"{video_id}",
        "language":"/"
    }
    print("video_ack = ", video_ack)
    while not publish_and_wait(client, MSG_TOPIC, start_video_message, wait_for="ACK_VIDEO"):
        print("Waiting for ACK_VIDEO...")
    print(f"Exercise demonstration for {start_video_message} completed.")

def stop_video(client,exercise):
    if exercise['exerciseId'] == 24:
         video_id=1
    elif exercise['exerciseId'] == 25:
         video_id=2
    elif exercise['exerciseId'] == 26:
         video_id=3
    elif exercise['exerciseId'] == 27:
        video_id=4
    stop_video_message = {
        "action": "STOP_VIDEO",
        "exercise": "",
        "timestamp": datetime.now().isoformat(),
        "code" : "",
        "message": f"{video_id}",
        "language":"/"
    }   
    
    while not publish_and_wait(client, MSG_TOPIC, stop_video_message, wait_for="VIDEO_END"):
        print("Waiting for VIDEO_END...")
    print(f"Exercise demonstration for {stop_video_message} completed.")


def start_exergames(client, exercise):
    #reset_global_flags()

    if exercise['exerciseId'] == 28:
        exercise_name = "holobalance_exergame_s2_sitting_1"
        msg= "0"
    elif exercise['exerciseId'] == 29:
        exercise_name = "holobalance_exergame_s2_sitting_2"
        msg= "0"
    elif exercise['exerciseId'] == 30:
        exercise_name = "holobalance_exergame_s2_standing_1"
        msg= "0"
    elif exercise['exerciseId'] == 31:
        exercise_name = "holobalance_exergame_s2_standing_2"
        msg= "0"
    elif exercise['exerciseId'] == 32:
        exercise_name = "holobalance_exergame_s2_standing_3"
        msg= "1"
    elif exercise['exerciseId'] == 33:
        exercise_name = "holobalance_exergame_s2_walking_1"
        msg= "0"
    elif exercise['exerciseId'] == 34:
        exercise_name = "holobalance_exergame_s2_walking_2"
        msg= "0"
    elif exercise['exerciseId'] == 35:
        exercise_name = "holobalance_exergame_s2_walking_3"
        msg= "1"
    elif exercise['exerciseId'] == 36:
        exercise_name = "holobalance_exergame_s2_walking_4"
        msg= "2"

    exergame_message = {
        "action": "START_CTG",
        "exercise": exercise_name,
        "timestamp": datetime.now().isoformat(),
        "code" : "",
        "message": msg,
        "language":"/"
    }


    while not publish_and_wait(client, DEMO_TOPIC, exergame_message, wait_for="ACK_CTG"):
        print("Waiting for CTG_END...")
    print(f"Exercise demonstration for {exercise_name} completed.")





def start_cognitive_games(client, exercise):
    global ctg_results_data,ctg_results_received
    #reset_global_flags()
    #reset_ctg()
    if exercise['exerciseId'] == 37:
        exercise_name = "holobalance_cognitive_s3_animal_feeding"
        msg= f"{exercise['progression']}"
    elif exercise['exerciseId'] == 38:
        exercise_name = "holobalance_cognitive_s3_bridge_crossing"
        msg= f"{exercise['progression']}"
    elif exercise['exerciseId'] == 39:
        exercise_name = "holobalance_cognitive_s3_catching_food"
        msg= f"{exercise['progression']}"
    elif exercise['exerciseId'] == 40:
        exercise_name = "holobalance_cognitive_s3_memory"
        msg= f"{exercise['progression']}"
    elif exercise['exerciseId'] == 41:
        exercise_name = "holobalance_cognitive_s3_preparing_animal_food"
        msg= f"{exercise['progression']}"
    elif exercise['exerciseId'] == 42:
        exercise_name = "holobalance_cognitive_s3_remember_previous"
        msg= f"{exercise['progression']}"
    

    cognitive_message = {
        "action": "START_CTG",
        "exercise": exercise_name,
        "timestamp": datetime.now().isoformat(),
        "code" : "",
        "message": msg,
        "language":"/"
    }


    #success = publish_and_wait(client, DEMO_TOPIC, cognitive_message, wait_for="CTG_RESULTS")
    success = publish_and_wait(client, DEMO_TOPIC, cognitive_message, wait_for="CTG_END")
    
    if success and ctg_results_data:
        print("Results received for cognitive game.")
        print("ctg_results_data =", ctg_results_data)

        try:
            #parsed_data = json.loads(ctg_results_data) # needs to be added back in when cognitive are fixed
            parsed_data = {'score':'0'}
            if isinstance(parsed_data, str):
                parsed_data = json.loads(parsed_data)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
            return None
    else:
        print("No results returned from cognitive game.")
        print("ctg_results_received =", ctg_results_received)
        print("ctg_results_data =", ctg_results_data)
        return None



# Function for Sending Oral Instructions
def send_voice_instructions(client, code):
    #reset_global_flags()
    oral_message = {
        "action": "SPEAK",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "message":"",
        "language":"/"
    }

    while not publish_and_wait(client, MSG_TOPIC, oral_message, wait_for="FINISH"):
        print("Retrying DEMO_START...")

def send_voice_instructions_ctg(client, code):
    #reset_global_flags()
    oral_message = {
        "action": "SPEAK_CTG",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "message":"",
        "language":"/"
    }
    # while not publish_and_wait(MSG_TOPIC, oral_message, wait_for="ACK"):
    #     print("Retrying SPEAK message...")
    while not publish_and_wait(client, MSG_TOPIC, oral_message, wait_for="FINISH"):
        print("Retrying DEMO_START...")
    
def send_message_with_speech_to_text(client, code, timeout=20):
    """
    Send an oral instruction message and wait for FINISH_RESPONSE.

    Args:
        code (str): Code to identify the message.
        timeout (int): Time to wait for a response before retrying (default: 20 seconds).

    Returns:
        str: The user's response ("yes" or "no").
    """
    global finish_response,finish_received
    finish_response = None;
    #reset_global_flags()

    # Message format stays the same as simple oral message
    oral_message = {
        "action": "SPEAK",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "message": "",
        "language": "/"
    }

    while True:
        # Send the message
        client.publish(MSG_TOPIC, json.dumps(oral_message))
        print(f"Published oral message: {oral_message}")
        start_time = time.time()

        # Wait for response
        while time.time() - start_time < timeout:
            if finish_response in ["yes", "no"]:
                print(f"Received FINISH_RESPONSE: {finish_response}")
                finish_received=False
                return finish_response  # Exit loop and return response
            elif finish_response == 'APPKILLED':
                print("App was killed, exiting...")
                finish_received=False
                return 'APPKILLED'
            time.sleep(0.5)
        time.sleep(5)

        # No response received, retrying
        print("No response received. Retrying...")


def send_message_with_speech_to_text_2(client, code, timeout=20):

    global finish_response ,finish_received
    finish_response = None;
    #reset_global_flags()

    # Message format stays the same as simple oral message
    oral_message = {
        "action": "SPEAK",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "message": "mild, moderate, severe?",
        "language": "/"
    }

    while True:
        # Send the message
        client.publish(MSG_TOPIC, json.dumps(oral_message))
        print(f"Published oral message: {oral_message}")
        start_time = time.time()

        # Wait for response
        while time.time() - start_time < timeout:
            print(finish_response)
            if finish_response in ["mild", "moderate","severe"]:
                print(f"Received FINISH_RESPONSE: {finish_response}")
                finish_received=False
                return finish_response  # Exit loop and return response
            time.sleep(0.5)
        time.sleep(5)

        # No response received, retrying
        print("No response received. Retrying...")


def send_message_with_speech_to_text_ctg(client, code, timeout=20):

    global finish_response,finish_received
    #reset_global_flags()
    finish_response = None;

    # Message format stays the same as simple oral message
    oral_message = {
        "action": "SPEAK_CTG",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "message": "",
        "language": "/"
    }

    while True:
        # Send the message
        client.publish(MSG_TOPIC, json.dumps(oral_message))
        print(f"Published oral message: {oral_message}")
        start_time = time.time()

        # Wait for response
        while time.time() - start_time < timeout:
            if finish_response in ["yes", "no"]:
                print(f"Received FINISH_RESPONSE: {finish_response}")
                finish_received=False
                return finish_response  # Exit loop and return response
            time.sleep(0.5)
        time.sleep(5)

        # No response received, retrying
        print("No response received. Retrying...")


def send_message_with_speech_to_text_ctg_2(client, code, timeout=20):

    global finish_response,finish_received
    #reset_global_flags()
    finish_response = None;

    # Message format stays the same as simple oral message
    oral_message = {
        "action": "SPEAK_CTG",
        "exercise": "/",
        "timestamp": datetime.now().isoformat(),
        "code": code,
        "message": "You have stopped too early, please try to continue the exercise. Yes or No?",
        "language": "/"
    }

    while True:
        # Send the message
        client.publish(MSG_TOPIC, json.dumps(oral_message))
        print(f"Published oral message: {oral_message}")
        start_time = time.time()

        # Wait for response
        while time.time() - start_time < timeout:
            if finish_response in ["low", "moderate","severe"]:
                print(f"Received FINISH_RESPONSE: {finish_response}")
                finish_received=False
                return finish_response  # Exit loop and return response
            time.sleep(0.5)
        time.sleep(5)

        # No response received, retrying
        print("No response received. Retrying...")
__all__ = [
    'app_connected' 
]
