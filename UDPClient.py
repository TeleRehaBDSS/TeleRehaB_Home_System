import socket
import paho.mqtt.client as mqtt
import threading
import FindMyIP as ip
import time

from shared_variables import (
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL
)
import configparser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Construct the paths for config and logo
CONFIG_PATH_2 = BASE_DIR / 'clinic.ini'
# Load config
config = configparser.ConfigParser()
config.read(CONFIG_PATH_2)
# Get clinic_id and strip quotes
clinic_id = config.get("CLINIC", "clinic_id").strip('"')
# MQTT Configuration

MQTT_TOPIC = f"TELEREHAB@{clinic_id}/DeviceStatus"
IP_TOPIC = f"TELEREHAB@{clinic_id}/EDGEIP";
MQTT_EXPECTED_MESSAGE = "up"

# Multicast Configuration
MULTICAST_GROUP = '224.1.1.1'
MULTICAST_PORT = 10001

# Flag to control broadcasting
broadcasting = True

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc));
    client.subscribe(f"TELEREHAB@{clinic_id}/EDGEIP")

def on_message(client, userdata, msg):
    global broadcasting
    message_payload = msg.payload.decode("utf-8")
    #print(f"Received MQTT message on topic [{msg.topic}]: {message_payload}")

    if msg.topic == IP_TOPIC:
        print('Received', message_payload, 'message, stopping IP sending.')
        broadcasting = False  # Stop broadcasting

def start_mqtt_listener():
    """Listens for MQTT messages and stops broadcasting when needed."""
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()  # Keep listening

def SendMyIP():
    """Continuously sends the local IP via UDP multicast until stopped."""
    global broadcasting

    local_ip = ip.internal()
    print(f"Broadcasting local IP: {local_ip}")

    # Start MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)
    client.loop_start()

    
    while broadcasting:
        client.publish(IP_TOPIC, local_ip)
        time.sleep(5)


    print("IP broadcasting stopped.")
    # Gracefully close
    client.disconnect()
    client.loop_stop()
