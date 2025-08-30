import os
import time
import cv2
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pyrealsense2 as rs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import paho.mqtt.client as mqtt
from shared_variables import MQTT_BROKER_HOST,MQTT_BROKER_PORT,MQTT_KEEP_ALIVE_INTERVAL
from pathlib import Path
import configparser

print('check01')

BASE_DIR = Path(__file__).resolve().parent
# Construct the paths for config and logo
CONFIG_PATH_2 = BASE_DIR / 'clinic.ini'
# Load config
config = configparser.ConfigParser()
config.read(CONFIG_PATH_2)
# Get clinic_id and strip quotes
clinic_id = config.get("CLINIC", "clinic_id").strip('"')

CAMERA_TOPIC = f"camera@{clinic_id}"
model_name = "movenet_lightning"
input_size = 192
MIN_CROP_KEYPOINT_SCORE = 0.2

KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

if "lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
elif "thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Unsupported model name.")
print('check02')
def movenet(input_image):
    model = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    return outputs['output_0'].numpy()

def init_crop_region(image_height, image_width):
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width
    return {
        'y_min': y_min, 'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height, 'width': box_width
    }

def crop_and_resize(image, crop_region):
    boxes = [[crop_region['y_min'], crop_region['x_min'], crop_region['y_max'], crop_region['x_max']]]
    return tf.image.crop_and_resize(image, box_indices=[0], boxes=boxes, crop_size=[input_size, input_size])

def run_inference(image, crop_region):
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(tf.expand_dims(image, axis=0), crop_region)
    keypoints_with_scores = movenet(input_image)
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.11):
    keypoints_all, keypoint_edges_all, edge_colors = [], [], []
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]
    kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (kpts_scores[edge_pair[0]] > keypoint_threshold and
            kpts_scores[edge_pair[1]] > keypoint_threshold):
            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)

    keypoints_xy = np.concatenate(keypoints_all, axis=0) if keypoints_all else np.zeros((0, 17, 2))
    edges_xy = np.stack(keypoint_edges_all, axis=0) if keypoint_edges_all else np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(image, keypoints_with_scores, crop_region=None, output_image_height=None):
    height, width, _ = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.imshow(image)

    keypoint_locs, keypoint_edges, edge_colors = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    line_segments = LineCollection(keypoint_edges, colors=edge_colors, linewidths=4)
    ax.add_collection(line_segments)
    if keypoint_locs.shape[0]:
        ax.scatter(keypoint_locs[:, 0], keypoint_locs[:, 1], s=60, color='#FF1493', zorder=3)

    if crop_region:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle((xmin, ymin), rec_width, rec_height, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_from_plot = image_from_plot.reshape((h, w, 4))[:, :, :3]
    plt.close(fig)

    if output_image_height:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(image_from_plot, (output_image_width, output_image_height), interpolation=cv2.INTER_CUBIC)
    return image_from_plot

def torso_angle(keypoints):
    left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']]
    right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']]
    left_hip = keypoints[KEYPOINT_DICT['left_hip']]
    right_hip = keypoints[KEYPOINT_DICT['right_hip']]

    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    vertical = np.array([0, -1])
    torso_vec = mid_shoulder[:2] - mid_hip[:2]
    torso_vec = torso_vec / np.linalg.norm(torso_vec)

    angle_rad = np.arccos(np.clip(np.dot(torso_vec, vertical), -1.0, 1.0))
    return np.degrees(angle_rad)

def head_tilt(keypoints):
    nose = keypoints[KEYPOINT_DICT['nose']]
    left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']]
    right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']]
    mid_shoulder = (left_shoulder + right_shoulder) / 2

    head_vector = nose[:2] - mid_shoulder[:2]
    angle = np.degrees(np.arctan2(-head_vector[1], head_vector[0]))
    return angle

def spine_deviation(keypoints):
    nose = keypoints[KEYPOINT_DICT['nose']]
    shoulders = (keypoints[KEYPOINT_DICT['left_shoulder']] + keypoints[KEYPOINT_DICT['right_shoulder']]) / 2
    hips = (keypoints[KEYPOINT_DICT['left_hip']] + keypoints[KEYPOINT_DICT['right_hip']]) / 2

    x_vals = [nose[0], shoulders[0], hips[0]]
    std_dev = np.std(x_vals)
    return std_dev

def detect_sitting_or_standing(keypoints):
    l_hip = keypoints[KEYPOINT_DICT['left_hip']]
    r_hip = keypoints[KEYPOINT_DICT['right_hip']]
    l_knee = keypoints[KEYPOINT_DICT['left_knee']]
    r_knee = keypoints[KEYPOINT_DICT['right_knee']]
    l_ankle = keypoints[KEYPOINT_DICT['left_ankle']]
    r_ankle = keypoints[KEYPOINT_DICT['right_ankle']]

    hip = (l_hip[1] + r_hip[1]) / 2
    knee = (l_knee[1] + r_knee[1]) / 2
    ankle = (l_ankle[1] + r_ankle[1]) / 2

    hip_to_knee = abs(hip - knee)
    if hip_to_knee < 0.15 and hip > knee:
        return "sitting"
    else:
        return "standing"


def legs_position(keypoints):
    left_knee = keypoints[KEYPOINT_DICT['left_knee']]
    right_knee = keypoints[KEYPOINT_DICT['right_knee']]
    left_ankle = keypoints[KEYPOINT_DICT['left_ankle']]
    right_ankle = keypoints[KEYPOINT_DICT['right_ankle']]

    avg_knee_y = (left_knee[1] + right_knee[1]) / 2
    avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
    leg_extension_ratio = abs(avg_ankle_y - avg_knee_y)
    return leg_extension_ratio

collect_data = False
running = True
start_time = None

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(CAMERA_TOPIC)

def on_message(client, userdata, msg):
    global collect_data, start_time, collected_torso, collected_head, collected_spine, collected_legs, running
    payload = msg.payload.decode().strip().strip('"')
    print("MQTT Message Received:", payload)
    if payload == "CAMERA_START":
        collect_data = True
        start_time = time.time()
        collected_torso, collected_head, collected_spine, collected_legs = [], [], [], []
    elif payload == "CAMERA_STOP":
        collect_data = False
    elif payload == "CAMERAOUT":
        running = False
print('check03')
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL)
client.loop_start()
print('check04')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

prev_crop_region = init_crop_region(480, 640)
frame_interval = 0.5
last_frame_time = time.time()
k = 0;
try:
    while running:
        if time.time() - last_frame_time < frame_interval:
            continue
        last_frame_time = time.time()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        if collect_data:
            tf_image = tf.convert_to_tensor(frame, dtype=tf.uint8)
            keypoints = run_inference(tf_image, prev_crop_region)
            prev_crop_region = init_crop_region(480, 640)
            keypoint_coords = keypoints[0, 0, :, :2]
            torso = torso_angle(keypoint_coords)
            head = head_tilt(keypoint_coords)
            spine = spine_deviation(keypoint_coords)
            legs = legs_position(keypoint_coords)

            collected_torso.append(torso)
            collected_head.append(head)
            collected_spine.append(spine)
            collected_legs.append(legs)

            if time.time() - start_time >= 7:
                avg_torso = sum(collected_torso) / len(collected_torso)
                avg_head = sum(collected_head) / len(collected_head)
                avg_spine = sum(collected_spine) / len(collected_spine)
                avg_legs = sum(collected_legs) / len(collected_legs)
                result = {
                    "avg_torso_angle": round(avg_torso, 2),
                    "torso_characterization": "forward" if avg_torso < 70 else "back" if avg_torso > 110 else "neutral",
                    "avg_head_tilt": round(avg_head, 2),
                    "head_characterization": "tilted" if not (-130 <= avg_head <= -105) else "neutral",
                    "avg_spine_deviation": round(avg_spine, 2),
                    "spine_characterization": "misaligned" if avg_spine > 0.26 else "aligned",
                    "avg_legs_position": round(avg_legs, 2),
                    "legs_characterization": "extended" if avg_legs > 0.14 else "neutral",
                    "posture": detect_sitting_or_standing(keypoint_coords)
                }
                try:
                    with open("posture_results.txt", "w") as f:
                        json.dump(result, f, indent=4)
                    print("Posture metrics saved.")
                    client.publish("CAMERAOUT", json.dumps(result))
                except Exception as e:
                    print(f"Failed to save posture results: {e}")
                collect_data = False

            output_frame = draw_prediction_on_image(frame, keypoints, crop_region=prev_crop_region, output_image_height=480)
            print(result);
            #cv2.imshow('Pose Estimation', output_frame)
        else:
            k = k + 1
            #cv2.imshow('Pose Estimation', frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
