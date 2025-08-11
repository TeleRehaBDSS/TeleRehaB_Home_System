import requests
import configparser
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from pathlib import Path

# Get the directory where the script is located
BASE_DIR = Path(__file__).resolve().parent

# Construct the paths for config and logo
CONFIG_PATH = BASE_DIR / 'config.ini'
LOGO_PATH = BASE_DIR / 'logo.png'

# Load API key from config file
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
api_key = config['API'].get('key_doctor', '')

# Define the API endpoints
login_url = 'https://telerehab-develop.biomed.ntua.gr/api/Login'
data_url = 'https://telerehab-develop.biomed.ntua.gr/api/PatientDeviceSet/list'

# Function to login and get the token
def login(username, password):
    login_payload = {
        'username': username,
        'password': password
    }
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json-patch+json'
    }
    response = requests.post(login_url, json=login_payload, headers=headers)
    if response.status_code == 200:
        token = response.json().get('message')
        config['API']['key_doctor'] = token
        with open(CONFIG_PATH, 'w') as configfile:
            config.write(configfile)
        return token
    else:
        messagebox.showerror("Error", "Login failed")
        return None

# Function to get data using the token
def get_data():
    token = config['API']['key_doctor']
    headers = {
        'accept': '*/*',
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(data_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        messagebox.showerror("Error", "Failed to retrieve data")
        return None

# Create the GUI
def create_gui():
    def handle_login():
        username = username_entry.get()
        password = password_entry.get()
        token = login(username, password)
        if token:
            messagebox.showinfo("Success", "Logged in successfully")

    def handle_get_data():
        data = get_data()
        if data:
            patient_ids = [str(item['patientId']) for item in data]
            api_keys = {str(item['patientId']): item['apiKey'] for item in data}
            patient_combobox['values'] = patient_ids
            patient_combobox.api_keys = api_keys

    def handle_save():
        selected_patient_id = patient_combobox.get()
        api_key = patient_combobox.api_keys.get(selected_patient_id)
        if api_key:
            config['API']['key_edge'] = api_key
            with open(CONFIG_PATH, 'w') as configfile:
                config.write(configfile)
            api_key_text.delete(1.0, tk.END)
            api_key_text.insert(tk.END, api_key)
            messagebox.showinfo("Success", f"API key for patient ID {selected_patient_id} saved successfully")

    root = tk.Tk()
    root.title("Register Edge Computer")

    # Load and display logo
    logo = Image.open(LOGO_PATH)  # Replace with the correct path to your logo
    logo = logo.resize((300, 200), Image.LANCZOS)
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(root, image=logo)
    logo_label.grid(row=0, columnspan=2)

    tk.Label(root, text="Username:").grid(row=1, column=0)
    username_entry = tk.Entry(root)
    username_entry.grid(row=1, column=1)

    tk.Label(root, text="Password:").grid(row=2, column=0)
    password_entry = tk.Entry(root, show="*")
    password_entry.grid(row=2, column=1)

    login_button = tk.Button(root, text="Login", command=handle_login)
    login_button.grid(row=3, columnspan=2)

    data_button = tk.Button(root, text="Get Data", command=handle_get_data)
    data_button.grid(row=4, columnspan=2)

    tk.Label(root, text="Select Patient ID:").grid(row=5, column=0)
    patient_combobox = ttk.Combobox(root)
    patient_combobox.grid(row=5, column=1)

    save_button = tk.Button(root, text="Save API Key", command=handle_save)
    save_button.grid(row=6, columnspan=2)

    tk.Label(root, text="API Key:").grid(row=7, column=0)
    api_key_text = tk.Text(root, height=1, width=50)
    api_key_text.grid(row=7, column=1)

    root.mainloop()

if __name__ == '__main__':
    create_gui()
