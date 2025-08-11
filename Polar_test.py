import asyncio
import struct
import json
import logging
import time
import threading
import queue
import paho.mqtt.client as mqtt
from bleak import BleakScanner, BleakClient
from shared_variables import MQTT_BROKER_HOST,MQTT_BROKER_PORT,MQTT_KEEP_ALIVE_INTERVAL
from pathlib import Path
import configparser
import os

BASE_DIR = Path(__file__).resolve().parent
# Construct the paths for config and logo
CONFIG_PATH_2 = BASE_DIR / 'clinic.ini'
# Load config
config = configparser.ConfigParser()
config.read(CONFIG_PATH_2)
# Get clinic_id and strip quotes
clinic_id = config.get("CLINIC", "clinic_id").strip('"')

POLAR_TOPIC = f"polar@{clinic_id}"

# BLE UUIDs
HR_MEASUREMENT_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
BLE_ADAPTER_INDEX = 0

# Globals
polar_data = []
polar_collecting = False
mqtt_client = None
polar_thread = None  
thread_started = False  
command_queue = queue.Queue()


def parse_heart_rate(data):
    flags = data[0]
    hr_format = flags & 0x01
    rr_interval_present = (flags >> 4) & 0x01
    offset = 1
    heart_rate = data[offset] if hr_format == 0 else struct.unpack_from("<H", data, offset)[0]
    offset += 1 if hr_format == 0 else 2
    rr_intervals = []
    if rr_interval_present:
        while offset + 1 < len(data):
            rr = struct.unpack_from("<H", data, offset)[0]
            rr_intervals.append(rr / 1024.0)
            offset += 2
    return heart_rate, rr_intervals


async def polar_manager():
    global polar_collecting, polar_data

    print("Searching for Polar H10...")
    polar = None
    while polar is None:
        devices = await BleakScanner.discover(adapter=BLE_ADAPTER_INDEX)
        polar = next((d for d in devices if d.name and "Polar H10" in d.name), None)
        if not polar:
            print("Polar H10 not found, retrying...")
            await asyncio.sleep(5)

    print(f"Found {polar.name} at {polar.address}")
    async with BleakClient(polar.address, adapter=BLE_ADAPTER_INDEX) as client:

        print("Connected to Polar H10.")

        async def listen_hr(sender, data):
            if polar_collecting:
                hr, rr = parse_heart_rate(data)
                polar_data.append((hr, rr))

        await client.start_notify(HR_MEASUREMENT_CHAR_UUID, listen_hr)

        try:
            while True:
                cmd = await asyncio.get_event_loop().run_in_executor(None, command_queue.get)
                if cmd == "start":
                    print("Starting data collection")
                    polar_data.clear()
                    polar_collecting = True
                elif cmd == "stop":
                    print("Stopping data collection")
                    polar_collecting = False
                    await asyncio.sleep(2)
                    await client.stop_notify(HR_MEASUREMENT_CHAR_UUID)
                    await save_polar_data()
                    if client.is_connected:
                        await client.disconnect()

                    break  # Exit to reconnect cleanly
        except Exception as e:
            print(f"⚠️ Polar manager error: {e}")


async def save_polar_data():
    global mqtt_client
    if not polar_data:
        print("⚠️ No Polar data collected.")
        result = {"error": "No data recorded"}
    else:
        hr_values = [hr for hr, _ in polar_data]
        rr_values = [rr for _, rr_list in polar_data for rr in rr_list]
        result = {
            "average_heart_rate": round(sum(hr_values) / len(hr_values), 2) if hr_values else 0,
            "average_rr_interval": round(sum(rr_values) / len(rr_values), 3) if rr_values else 0,
            "samples_collected": len(polar_data)
        }

    with open("polar_results.txt", "w") as f:
        json.dump(result, f, indent=4)
    print("Polar metrics saved to polar_results.txt")


def on_connect(client, userdata, flags, rc):
    print("🔗 Connected to MQTT with result code", rc)
    client.subscribe(POLAR_TOPIC)


def on_message(client, userdata, msg):
    global polar_thread, thread_started
    command = msg.payload.decode().strip().lower()
    print(f"Received: {msg.topic} = {command}")

    if command == "start":
        if not thread_started:
            command_queue.put("start")
            polar_thread = threading.Thread(target=asyncio.run, args=(polar_manager(),))
            polar_thread.start()
            thread_started = True
        else:
            print("⚠️ Polar manager already running. Ignoring duplicate start.")

    elif command == "stop":
        command_queue.put("stop")
        thread_started = False

    elif command == "polarout":
        print("Exiting Polar script.")
        os._exit(0)





def main():
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)
    mqtt_client.loop_forever()


if __name__ == "__main__":
    main()
