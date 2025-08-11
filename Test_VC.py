import sys 
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt
import paho.mqtt.client as mqtt

MQTT_PORT = 1883
MQTT_TOPIC = "exercise/demo"

class MQTTExerciseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.client = mqtt.Client()
        self.connected = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Telerehab Demo Exercise")
        layout = QVBoxLayout()

       # MQTT Broker Connection
        broker_layout = QHBoxLayout()
        self.broker_ip = QLineEdit()
        self.broker_ip.setPlaceholderText("MQTT Broker IP")
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_mqtt)
        broker_layout.addWidget(self.broker_ip)
        broker_layout.addWidget(self.connect_btn)
        layout.addLayout(broker_layout)

         # Connection Status
        self.status_label = QLabel("Not Connected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.status_label)

        # Language Group
        lang_group = QGroupBox("Language Selection")
        lang_layout = QHBoxLayout()
        for lang in ['EN', 'GR', 'DE', 'PT', 'TH']:
            btn = QPushButton(lang)
            btn.setFixedSize(60, 40)
            btn.clicked.connect(lambda _, l=lang: self.send_language(l))
            lang_layout.addWidget(btn)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        # Exercise Configuration Group
        config_group = QGroupBox("Exercise Configuration")
        config_layout = QVBoxLayout()

        # Category Selection
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))
        self.category_cb = QComboBox()
        self.category_cb.addItems(['Sitting', 'Standing', 'Walking', 'Stretching', 'Exergames', 'Cognitive Games'])
        self.category_cb.currentTextChanged.connect(self.update_exercises)
        category_layout.addWidget(self.category_cb)
        config_layout.addLayout(category_layout)

        # Exercise Selection
        exercise_layout = QHBoxLayout()
        exercise_layout.addWidget(QLabel("Exercise:"))
        self.exercise_cb = QComboBox()
        exercise_layout.addWidget(self.exercise_cb)
        config_layout.addLayout(exercise_layout)

        # Progression Selection
        progression_layout = QHBoxLayout()
        progression_layout.addWidget(QLabel("Progression/Level:"))
        self.progression_cb = QComboBox()
        progression_layout.addWidget(self.progression_cb)
        config_layout.addLayout(progression_layout)

        # Start Button
        self.start_btn = QPushButton("Start Exercise")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.send_exercise)
        config_layout.addWidget(self.start_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        self.setLayout(layout)
        self.update_exercises()
        self.show()

    def toggle_connection(self):
        if self.connected:
            self.disconnect_mqtt()
        else:
            self.connect_mqtt()

    def connect_mqtt(self):
        try:
            self.client.connect(self.broker_ip.text().strip(), MQTT_PORT, 60)
            self.connected = True
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self.status_label.setText("Not Connected")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")


    def send_language(self, lang):
        if not self.connected:
            QMessageBox.warning(self, "Error", "MQTT broker not connected.")
            return

        payload = {
            "action": "LANGUAGE",
            "exercise": "/",
            "timestamp": datetime.now().isoformat(),
            "code": "",
            "message": "",
            "language": lang
        }
        self.publish(payload)
        self.log_message(f"Language set to {lang}")

    def send_exercise(self):
        if not self.connected:
            QMessageBox.warning(self, "Error", "MQTT broker not connected.")
            return

        category = self.category_cb.currentText()
        timestamp = datetime.now().isoformat()

        try:
            if category in ['Sitting', 'Standing', 'Walking', 'Stretching']:
                exercise_str = self.progression_cb.currentData()
                payload = {
                    "action": "START",
                    "exercise": exercise_str,
                    "timestamp": timestamp,
                    "code": "",
                    "message": "",
                    "language": "/"
                }
            elif category == 'Exergames':
                ex_data = self.exercise_cb.currentData()
                payload = {
                    "action": "START_CTG",
                    "exercise": ex_data['exercise_str'],
                    "timestamp": timestamp,
                    "code": "",
                    "message": ex_data['message'],
                    "language": "/"
                }
            elif category == 'Cognitive Games':
                exercise_str = self.exercise_cb.currentData()
                message_level = self.progression_cb.currentData()
                payload = {
                    "action": "START_CTG",
                    "exercise": exercise_str,
                    "timestamp": timestamp,
                    "code": "",
                    "message": message_level,
                    "language": "/"
                }

            self.publish(payload)
            self.log_message(f"Sent {category} exercise: {payload['exercise']}")
        except Exception as e:
            self.log_message(f"Error sending exercise: {str(e)}", error=True)
            QMessageBox.critical(self, "Error", f"Failed to send exercise: {str(e)}")

    def publish(self, payload):
        msg = json.dumps(payload, indent=2)
        self.client.publish(MQTT_TOPIC, msg)
        self.log_message(f"Published to {MQTT_TOPIC}:\n{msg}")


    def update_exercises(self):
        category = self.category_cb.currentText()
        self.exercise_cb.clear()
        #self.exercise_cb.currentTextChanged.disconnect()

        if category in ['Sitting', 'Standing', 'Walking', 'Stretching']:
            exercises = {
                'Sitting': {
                    'Sit – Yaw': ['P0', 'P1', 'P2', 'P3'],
                    'Sit - Pitch': ['P0', 'P1', 'P2', 'P3'],
                    'Sit - Bend': ['P0', 'P1', 'P2', 'P3'],
                    'Seated trunk rotation': ['P0', 'P1', 'P2'],
                    'Assisted toe raises': ['P0', 'P1'],
                    'Heel raises': ['P0', 'P1'],
                    'Seated marching': ['P0', 'P1', 'P2'],
                    'Sit to Stand': ['P0', 'P1', 'P2', 'P3']
                },
                'Standing': {
                    'Maintain Balance': ['P0', 'P1', 'P2', 'P3'],
                    'Maintain Balance Foam': ['P0', 'P1', 'P2', 'P3'],
                    'Bend and Reach': ['P0', 'P1'],
                    'Overhead Reach': ['P0', 'P1'],
                    'Turn': ['P0', 'P1'],
                    'Lateral weight shifts': ['P0', 'P1', 'P2', 'P3'],
                    'Limits of stability': ['P0', 'P1', 'P2', 'P3'],
                    'Forward reach': ['P0', 'P1']
                },
                'Walking': {
                    'Horizon': ['P0', 'P1'],
                    'Yaw': ['P0', 'P1', 'P2', 'P3'],
                    'Pitch': ['P0', 'P1', 'P2', 'P3'],
                    'Side stepping': ['P0', 'P1', 'P2']
                },
                'Stretching': {
                    'Hip external rotator': ['P0'],
                    'Lateral trunk flexion': ['P0'],
                    'Calf stretch': ['P0']
                }
            }
            for ex in exercises[category].keys():
                self.exercise_cb.addItem(ex, ex)
        elif category == 'Exergames':
            exergames = {
                'Sitting 1': {'exercise_str': 'holobalance_exergame_s2_sitting_1', 'message': '0'},
                'Sitting 2': {'exercise_str': 'holobalance_exergame_s2_sitting_2', 'message': '0'},
                'Standing 1': {'exercise_str': 'holobalance_exergame_s2_standing_1', 'message': '0'},
                'Standing 2': {'exercise_str': 'holobalance_exergame_s2_standing_2', 'message': '0'},
                'Standing 3': {'exercise_str': 'holobalance_exergame_s2_standing_3', 'message': '1'},
                'Walking 1': {'exercise_str': 'holobalance_exergame_s2_walking_1', 'message': '0'},
                'Walking 2': {'exercise_str': 'holobalance_exergame_s2_walking_2', 'message': '0'},
                'Walking 3': {'exercise_str': 'holobalance_exergame_s2_walking_3', 'message': '1'},
                'Walking 4': {'exercise_str': 'holobalance_exergame_s2_walking_4', 'message': '2'}
            }
            for ex_name, data in exergames.items():
                self.exercise_cb.addItem(ex_name, data)
        elif category == 'Cognitive Games':
            cognitive_games = {
                'Memory Game': 'holobalance_cognitive_s3_memory',
                'Catching Food': 'holobalance_cognitive_s3_catching_food',
                'Remember Previous': 'holobalance_cognitive_s3_remember_previous',
                'Bridge Crossing': 'holobalance_cognitive_s3_bridge_crossing',
                'Animal Feeding': 'holobalance_cognitive_s3_animal_feeding',
                'Preparing Animal Food': 'holobalance_cognitive_s3_preparing_animal_food'
            }
            for game_name, ex_str in cognitive_games.items():
                self.exercise_cb.addItem(game_name, ex_str)

        self.update_progressions()
        self.exercise_cb.currentTextChanged.connect(self.update_progressions)

    def update_progressions(self):
        category = self.category_cb.currentText()
        self.progression_cb.clear()

        if category in ['Sitting', 'Standing', 'Walking', 'Stretching']:
            exercises = {
                'Sitting': {
                    'Sit – Yaw': ['P0', 'P1', 'P2', 'P3'],
                    'Sit - Pitch': ['P0', 'P1', 'P2', 'P3'],
                    'Sit - Bend': ['P0', 'P1', 'P2', 'P3'],
                    'Seated trunk rotation': ['P0', 'P1', 'P2'],
                    'Assisted toe raises': ['P0', 'P1'],
                    'Heel raises': ['P0', 'P1'],
                    'Seated marching': ['P0', 'P1', 'P2'],
                    'Sit to Stand': ['P0', 'P1', 'P2', 'P3']
                },
                'Standing': {
                    'Maintain Balance': ['P0', 'P1', 'P2', 'P3'],
                    'Maintain Balance Foam': ['P0', 'P1', 'P2', 'P3'],
                    'Bend and Reach': ['P0', 'P1'],
                    'Overhead Reach': ['P0', 'P1'],
                    'Turn': ['P0', 'P1'],
                    'Lateral weight shifts': ['P0', 'P1', 'P2', 'P3'],
                    'Limits of stability': ['P0', 'P1', 'P2', 'P3'],
                    'Forward reach': ['P0', 'P1']
                },
                'Walking': {
                    'Horizon': ['P0', 'P1'],
                    'Yaw': ['P0', 'P1', 'P2', 'P3'],
                    'Pitch': ['P0', 'P1', 'P2', 'P3'],
                    'Side stepping': ['P0', 'P1', 'P2']
                },
                'Stretching': {
                    'Hip external rotator': ['P0'],
                    'Lateral trunk flexion': ['P0'],
                    'Calf stretch': ['P0']
                }
            }
            exercise = self.exercise_cb.currentText()
            if not exercise:
                return
            progressions = exercises[category][exercise]
            category_key = category.lower()
            exercise_number = list(exercises[category].keys()).index(exercise) + 1

            for prog in progressions:
                ex_str = f"VC holobalance_{category_key}_{exercise_number} {prog}"
                self.progression_cb.addItem(prog, ex_str)
        elif category == 'Exergames':
            self.progression_cb.setEnabled(False)
            self.progression_cb.addItem("N/A", "")
        elif category == 'Cognitive Games':
            self.progression_cb.setEnabled(True)
            for level in range(5):
                self.progression_cb.addItem(f"Level {level+1}", str(level))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MQTTExerciseApp()
    sys.exit(app.exec_())