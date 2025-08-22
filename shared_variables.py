import multiprocessing as mp
from multiprocessing import Manager, Value
import time
import json
import FindMyIP as ip

local_ip = ip.internal()
#MQTT_BROKER_HOST = 'broker.emqx.io'  # Public MQTT broker for testing
MQTT_BROKER_HOST = local_ip
MQTT_BROKER_PORT = 1883
MQTT_KEEP_ALIVE_INTERVAL = 60


DEBUG = False;


# Define shared variables
manager = mp.Manager()
queueData = manager.Queue()
imu1Queue = manager.Queue()
imu2Queue = manager.Queue()
imu3Queue = manager.Queue()
imu4Queue = manager.Queue()
imu1FinalQueue = manager.Queue()
imu2FinalQueue = manager.Queue()
imu3FinalQueue = manager.Queue()
imu4FinalQueue = manager.Queue()
scheduleQueue = manager.Queue()

command_queue = mp.Queue() # For clinic_main -> scheduler
tick_queue = mp.Queue()    # For scheduler -> receive_imu_data


isFoundFirstTimestamp = mp.Value('b', False)
firstTimestamp = mp.Value('l', -1)
counter = 0
firstPacket = mp.Value('b', True)
lastDataTime = mp.Value('d',time.time())
startReceiving = mp.Value('b', False)
csv_file_path = manager.list()
write_to_files = mp.Value('b', False)
imus = manager.list()
jwt_token = ""
mqttState = mp.Value('c', b'I')  #  I -> "idle", R -> "receiving", S -> "stopReceiving", E -> "error"

#set True if you want to upload the results using API
enableConnectionToAPI = True
#set True to enable the metrics
enableMetrics = True
#set the time the metrics are calculated in sec
timeToCallMetrics = 15
#set True if you want to enable the interpolation
enableInterpolation = True

patientId = 1
sessionId = 5
exerciseId = 1
deviceApiKey = ""

#APIs
urlLogin = 'https://telerehab-develop.biomed.ntua.gr/api/Login'
urlProduceApiKey = 'https://telerehab-develop.biomed.ntua.gr/api/PatientDeviceSet/list'
urlUploadSensorData = 'https://telerehab-develop.biomed.ntua.gr/api/SensorData'
urlGetSchedule = 'https://telerehab-develop.biomed.ntua.gr/api/PatientSchedule/daily'
urlPerfomanceScore = 'https://telerehab-develop.biomed.ntua.gr/api/PatientSchedule/daily'

headers = {
    'accept': '*/*',
    'Content-Type': 'application/json-patch+json',
}
credentials = {
    "username": "testDoctor",
    "password": "TeleAdmin2023"
}

sensorDataToUpload = {
    "deviceId": "16", 
    "sessionId": sessionId,
    "exerciseId": exerciseId,   
    "data": {} 
}

feedbackData = {
    "Number of maxima greater than -0.65": 10,
    "Number of minima less than -0.7": 59,
    "Repetitions": 10,
    "Range of Motion": 0.23487970000000002,
    "Range of Motion (Yaw)": "359.10205910490043 degrees",
    "Rotation Speed (Tempo)": "0.40909090909090906 rotations per second",
    "Consistency (Std. Dev. of Peaks)": 0.010178870711628313
}

sensorDataToUpload["data"] = json.dumps(feedbackData)


