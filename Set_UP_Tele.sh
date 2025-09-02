#!/bin/bash

# ==============================================================================
# TeleRehaB Home System - Full Service Setup Script (from GitHub Repo)
#
# This script automates the entire process of setting up the TeleRehaB Home
# System on any modern Ubuntu system.
#
# It performs the following actions:
#   1. Asks the user to select a language for the application.
#   2. Installs all prerequisites, including build tools and RustDesk.
#   3. Builds and installs the Intel RealSense SDK from source if not present.
#   4. Ensures Python 3.10 and tkinter are installed.
#   5. Installs the project to a standard location (/opt/TeleRehaB_Home_System).
#   6. Clones or updates the project repository from GitHub.
#   7. Installs Python dependencies into a dedicated virtual environment.
#   8. Sets up a virtual screen for remote monitoring.
#   9. Safely configures passwordless shutdown for the heartbeat service.
#   10. Creates a robust, auto-updating run script (update_repo.sh).
#   11. Creates and configures systemd services with a safe, sequential startup order.
#   12. Creates a clean, authoritative Mosquitto config to prevent conflicts.
#   13. Disables Wayland for better graphical compatibility.
#   14. Creates a desktop application shortcut for the "Add Patient" GUI.
#   15. Enables and starts all services.
#   16. Verifies that all services are running.
#
# Usage:
#   1. Save this file as setup_service.sh
#   2. Make it executable: chmod +x setup_service.sh
#   3. Run it with sudo:   sudo ./setup_service.sh
# ==============================================================================

# --- Script Setup ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration Variables ---
GIT_REPO_URL="https://github.com/TeleRehaBDSS/TeleRehaB_Home_System"
REPO_NAME="TeleRehaB_Home_System"
PARENT_PATH="/opt" # Standard location for third-party software
PROJECT_PATH="$PARENT_PATH/$REPO_NAME"

# Get the username of the user running the script (even with sudo)
if [ "$SUDO_USER" ]; then
    CURRENT_USER=$SUDO_USER
else
    echo "This script needs to be run with sudo."
    echo "Usage: sudo ./setup_service.sh"
    exit 1
fi
USER_HOME=$(getent passwd "$CURRENT_USER" | cut -d: -f6)

# --- Main Script Logic ---

echo "--- TeleRehaB Home System Full Setup ---"
echo "Project will be installed in: $PROJECT_PATH"
echo

# 1. Get Language Choice from User
# ================================
while true; do
    read -p "Please choose a language (DE, EN, GR, TH, PT): " SELECTED_LANGUAGE
    SELECTED_LANGUAGE=${SELECTED_LANGUAGE^^}
    case $SELECTED_LANGUAGE in
        DE|EN|GR|TH|PT)
            echo "Language set to: $SELECTED_LANGUAGE"
            break
            ;;
        *)
            echo "Invalid selection. Please try again."
            ;;
    esac
done
echo

# 2. Install Prerequisites & Ensure Python 3.10 is available
# =========================================================
echo "Checking for and disabling problematic software repositories..."
# This function finds and comments out lines in apt sources files to prevent update errors
disable_repo() {
    local domain="$1"
    local source_file=$(grep -l "$domain" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true)
    if [ -n "$source_file" ]; then
        echo "Found problematic repository for '$domain' in $source_file. Disabling it to prevent errors..."
        sed -i -e "s|^deb.*$domain.*|# &|" "$source_file"
    fi
}

# List of known problematic domains from user logs
disable_repo "download.konghq.com"
disable_repo "apt.packages.shiftkey.dev"
disable_repo "mirror.mwt.me/shiftkey-desktop"
disable_repo "realsense-hw-public.s3.amazonaws.com"

echo "Updating package list and installing initial prerequisites..."
apt-get update
# General tools, MQTT, virtual screen, build dependencies, and wget for downloading
apt-get install --yes git mosquitto mosquitto-clients xserver-xorg-video-dummy cmake g++ libssl-dev libusb-1.0-0-dev pkg-config wget ca-certificates

echo "Manually adding deadsnakes PPA to install Python 3.10..."
# This bypasses the need for the potentially broken add-apt-repository command
echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes-ppa.list
# Add the GPG key for the repository
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776

echo "Updating package list again with new repository..."
apt-get update

echo "Installing Python 3.10..."
apt-get install --yes python3.10 python3.10-venv python3.10-tk

echo "Prerequisites and Python 3.10 are ready."
echo

# 3. Install RustDesk for Remote Access
# ======================================
echo "Installing RustDesk for remote access..."
if ! command -v rustdesk &> /dev/null; then
    RUSTDESK_VERSION="1.2.6" # Using a recent, stable version
    RUSTDESK_DEB="rustdesk-${RUSTDESK_VERSION}-x86_64.deb"
    RUSTDESK_URL="https://github.com/rustdesk/rustdesk/releases/download/${RUSTDESK_VERSION}/${RUSTDESK_DEB}"
    
    echo "Downloading RustDesk..."
    wget -O "/tmp/${RUSTDESK_DEB}" "$RUSTDESK_URL"
    
    echo "Installing RustDesk package..."
    # Use apt to handle dependencies automatically
    apt install --yes "/tmp/${RUSTDESK_DEB}"
    
    echo "Cleaning up downloaded file..."
    rm "/tmp/${RUSTDESK_DEB}"
    echo "RustDesk installed successfully."
else
    echo "RustDesk is already installed. Skipping."
fi
echo

# 4. Build and Install Intel RealSense SDK
# ========================================
REALSENSE_LIB_FILE="/usr/local/lib/librealsense2.so"
if [ ! -f "$REALSENSE_LIB_FILE" ]; then
    echo "Intel RealSense SDK not found. Building from source (this may take a while)..."
    REALSENSE_BUILD_DIR="$USER_HOME/realsense_build"
    
    # Ensure a clean state and correct permissions for the build directory
    rm -rf "$REALSENSE_BUILD_DIR"
    sudo -u "$CURRENT_USER" mkdir -p "$REALSENSE_BUILD_DIR"
    
    echo "Cloning librealsense repository..."
    # Clone the repository into the newly created directory
    sudo -u "$CURRENT_USER" git clone https://github.com/IntelRealSense/librealsense.git "$REALSENSE_BUILD_DIR"
    
    cd "$REALSENSE_BUILD_DIR"
    
    echo "Building SDK..."
    sudo -u "$CURRENT_USER" mkdir -p build && cd build
    sudo -u "$CURRENT_USER" cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false
    sudo -u "$CURRENT_USER" make -j$(nproc)
    
    echo "Installing SDK..."
    make install
    
    echo "Cleaning up build files..."
    cd "$USER_HOME"
    rm -rf "$REALSENSE_BUILD_DIR"
    
    # Update the dynamic linker cache
    ldconfig
    echo "Intel RealSense SDK installed successfully."
else
    echo "Intel RealSense SDK already installed. Skipping build."
fi
echo

# 5. Clone or Update the Repository
# =================================
echo "Cloning or updating the project repository..."
if [ -d "$PROJECT_PATH" ]; then
    echo "Directory '$PROJECT_PATH' already exists. Pulling latest changes..."
    cd "$PROJECT_PATH"
    # Clean up potential lock file before pulling
    rm -f .git/index.lock
    sudo -u "$CURRENT_USER" git pull
else
    echo "Cloning repository into a temporary location..."
    # Create a temporary directory in the user's home to avoid permission issues
    TEMP_CLONE_DIR=$(sudo -u "$CURRENT_USER" mktemp -d -p "$USER_HOME")
    
    # Clone as the user into the temporary directory
    sudo -u "$CURRENT_USER" git clone "$GIT_REPO_URL" "$TEMP_CLONE_DIR"
    
    echo "Moving project to $PROJECT_PATH..."
    # Move the temporary directory to the final destination as root
    mv "$TEMP_CLONE_DIR" "$PROJECT_PATH"
fi

# Ensure final ownership is correct regardless of path taken
chown -R "$CURRENT_USER":"$CURRENT_USER" "$PROJECT_PATH"
echo "Repository is up to date."
echo

# 6. Setup Python Virtual Environment and Install Dependencies
# =========================================================
VENV_PATH="$PROJECT_PATH/venv"
PYTHON_EXEC="$VENV_PATH/bin/python"
REQUIREMENTS_FILE="$PROJECT_PATH/requirements.txt"

echo "Setting up Python virtual environment..."
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating one with python3.10..."
    sudo -u "$CURRENT_USER" python3.10 -m venv "$VENV_PATH"
else
    echo "Virtual environment already exists."
fi

echo "Installing dependencies from requirements.txt..."
sudo -u "$CURRENT_USER" "$PYTHON_EXEC" -m pip install -r "$REQUIREMENTS_FILE"

echo "Forcing reinstallation of TensorFlow and NumPy to fix potential conflicts..."
# This command fixes the "module 'tensorflow' has no attribute '__version__'" error
sudo -u "$CURRENT_USER" "$PYTHON_EXEC" -m pip install --upgrade --force-reinstall tensorflow==2.14.0 keras==2.14.0
# This command fixes the NumPy 1.x vs 2.x conflict
sudo -u "$CURRENT_USER" "$PYTHON_EXEC" -m pip install --upgrade --force-reinstall numpy==1.26.4

echo "Dependencies installed successfully."
echo

# 7. Setup Virtual Screen for Remote Monitoring
# =============================================
XORG_BAK_FILE="/root/xorg.conf"
XORG_WRK_FILE="/usr/share/X11/xorg.conf.d/xorg.conf"
CHSCREEN_SCRIPT_PATH="/root/chScreen.sh"

echo "Setting up virtual screen configuration..."

(
cat <<'EOF'
Section "Device"
    Identifier  "Configured Video Device"
    Driver      "dummy"
EndSection
Section "Monitor"
    Identifier  "Configured Monitor"
    HorizSync 31.5-48.5
    VertRefresh 50-70
EndSection
Section "Screen"
    Identifier  "Default Screen"
    Monitor     "Configured Monitor"
    Device      "Configured Video Device"
    DefaultDepth 24
    SubSection "Display"
    Depth 24
    Modes "1024x800"
    EndSubSection
EndSection
EOF
) > "$XORG_BAK_FILE"

(
cat <<'EOF'
#!/bin/bash
XORG_BAK_FILE="/root/xorg.conf"
XORG_WRK_FILE="/usr/share/X11/xorg.conf.d/xorg.conf"
DM=$(basename "$(cat /etc/X11/default-display-manager 2>/dev/null || echo 'gdm3')")
check_monitor_connected() {
    for card in /sys/class/drm/card*; do
        for conn in "$card"/*; do
            if [ -f "$conn/status" ]; then
                status=$(cat "$conn/status")
                if [ "$status" = "connected" ]; then
                    return 0
                fi
            fi
        done
    done
    return 1
}
switch_display() {
    if check_monitor_connected; then
        echo "Monitor Connected. Disabling virtual screen."
        if [ -f "$XORG_WRK_FILE" ]; then
            rm -f "$XORG_WRK_FILE"
            sleep 2
            systemctl restart "$DM"
        fi
    else
        echo "Monitor Disconnected. Enabling virtual screen."
        if ! [ -f "$XORG_WRK_FILE" ]; then
            cp "$XORG_BAK_FILE" "$XORG_WRK_FILE"
            sleep 2
            systemctl restart "$DM"
        fi
    fi
}
switch_display
EOF
) > "$CHSCREEN_SCRIPT_PATH"

chmod +x "$CHSCREEN_SCRIPT_PATH"

CRON_JOB="*/1 * * * * $CHSCREEN_SCRIPT_PATH"
(crontab -l -u root 2>/dev/null | grep -Fv "$CHSCREEN_SCRIPT_PATH" ; echo "$CRON_JOB") | crontab -u root -

echo "Virtual screen setup complete. Cron job installed."
echo

# 8. Configure Passwordless Shutdown (Safe Method)
# =================================================
SUDOERS_FILE="/etc/sudoers.d/99-mqtt-shutdown"
echo "Configuring passwordless shutdown for the service..."
(
cat <<EOF
# Allow user $CURRENT_USER to run the poweroff command without a password.
# This is required for the MQTT Heartbeat service.
$CURRENT_USER ALL=NOPASSWD: /bin/systemctl poweroff
EOF
) > "$SUDOERS_FILE"
chmod 0440 "$SUDOERS_FILE"
echo "Passwordless shutdown configured successfully."
echo

# 9. Create the Auto-Update Script
# ==========================================
UPDATE_SCRIPT_PATH="/opt/update_repo.sh"
echo "Creating the auto-update script at $UPDATE_SCRIPT_PATH..."
(
cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$PROJECT_PATH"
cd "\$APP_DIR"
# Forcefully remove lock file to prevent errors from previous crashes
rm -f .git/index.lock
git config --global --add safe.directory "\$APP_DIR" || true
git remote set-url origin $GIT_REPO_URL || true
git fetch --prune origin
git reset --hard origin/main
git update-index --skip-worktree clinic.ini || true
git update-index --skip-worktree config.ini || true
git pull --rebase --autostash origin main
echo "Applying custom language configuration: $SELECTED_LANGUAGE"
# Specifically replace the default "EN" language with the selected one
sed -i "s/set_language(client, \"EN\")/set_language(client, \"$SELECTED_LANGUAGE\")/" clinic_main.py
echo "Repository update complete."
EOF
) > "$UPDATE_SCRIPT_PATH"
chmod +x "$UPDATE_SCRIPT_PATH"
echo "Auto-update script created."
echo

# 10. Create and Configure systemd Service Files
# =================================================
echo "Creating systemd service files with safe startup order..."

# --- Service 1: telerehab-update.service (Runs once at boot) ---
UPDATE_SERVICE_FILE="/etc/systemd/system/telerehab-update.service"
(
cat <<EOF
[Unit]
Description=TeleRehaB Home System - Update Repository
After=network-online.target
[Service]
Type=oneshot
User=$CURRENT_USER
ExecStart=/bin/bash -lc '$UPDATE_SCRIPT_PATH'
[Install]
WantedBy=multi-user.target
EOF
) > "$UPDATE_SERVICE_FILE"

# --- Service 2: mqtt-heartbeat.service (Starts after update) ---
MQTT_SERVICE_FILE="/etc/systemd/system/mqtt-heartbeat.service"
(
cat <<EOF
[Unit]
Description=MQTT Heartbeat Checker
After=telerehab-update.service mosquitto.service
Requires=telerehab-update.service mosquitto.service
[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_PATH
ExecStartPre=/bin/sleep 30
ExecStart=$PYTHON_EXEC $PROJECT_PATH/mqtt_healthcheck_standalone.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/mqtt-heartbeat.out
StandardError=append:/var/log/mqtt-heartbeat.err
[Install]
WantedBy=multi-user.target
EOF
) > "$MQTT_SERVICE_FILE"

# --- Service 3: telerehab-main.service (Starts after heartbeat) ---
MAIN_APP_SERVICE_FILE="/etc/systemd/system/telerehab-main.service"
(
cat <<EOF
[Unit]
Description=TeleRehaB Home System - Main App
After=mqtt-heartbeat.service
Requires=mqtt-heartbeat.service
[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_PATH
ExecStart=$PYTHON_EXEC $PROJECT_PATH/clinic_main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/$CURRENT_USER/.Xauthority"
StandardOutput=append:/var/log/telerehab-main.log
StandardError=append:/var/log/telerehab-main.log
[Install]
WantedBy=multi-user.target
EOF
) > "$MAIN_APP_SERVICE_FILE"

# --- Service 4: telerehab-depth-camera.service (Starts after main) ---
DEPTH_CAMERA_SERVICE_FILE="/etc/systemd/system/telerehab-depth-camera.service"
(
cat <<EOF
[Unit]
Description=TeleRehaB Home System - Depth Camera
After=telerehab-main.service
Requires=telerehab-main.service
[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_PATH
ExecStart=$PYTHON_EXEC $PROJECT_PATH/depth_camera_main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/$CURRENT_USER/.Xauthority"
StandardOutput=append:/var/log/telerehab-depth-camera.log
StandardError=append:/var/log/telerehab-depth-camera.log
[Install]
WantedBy=multi-user.target
EOF
) > "$DEPTH_CAMERA_SERVICE_FILE"

# --- Service 5: telerehab-polar.service (Starts after main) ---
POLAR_SERVICE_FILE="/etc/systemd/system/telerehab-polar.service"
(
cat <<EOF
[Unit]
Description=TeleRehaB Home System - Polar Sensor
After=telerehab-main.service
Requires=telerehab-main.service
[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_PATH
ExecStart=$PYTHON_EXEC $PROJECT_PATH/Polar_test.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/$CURRENT_USER/.Xauthority"
StandardOutput=append:/var/log/telerehab-polar.log
StandardError=append:/var/log/telerehab-polar.log
[Install]
WantedBy=multi-user.target
EOF
) > "$POLAR_SERVICE_FILE"

echo "All application service files created."
echo

# 11. Configure Mosquitto
# =======================
MOSQUITTO_MAIN_CONF="/etc/mosquitto/mosquitto.conf"

echo "Configuring Mosquitto to allow anonymous network connections and avoid conflicts..."
# Create a new, clean configuration file, overwriting any defaults.
(
cat <<EOF
# This file is automatically generated by the setup script.
# Default settings for persistence and logging.
pid_file /run/mosquitto/mosquitto.pid
persistence true
persistence_location /var/lib/mosquitto/
log_dest file /var/log/mosquitto/mosquitto.log

# Custom listener configuration.
# Allow connections from any IP address on the standard port.
listener 1883 0.0.0.0
# Allow clients to connect without a username/password.
allow_anonymous true
EOF
) > "$MOSQUITTO_MAIN_CONF"
echo "Mosquitto configuration complete."
echo

# 12. Create Desktop Application Shortcut
# ======================================
DESKTOP_FILE_PATH="/usr/share/applications/add-patient.desktop"
echo "Creating desktop application shortcut..."
(
cat <<EOF
[Desktop Entry]
Version=1.0
Name=Add Patient
Comment=Launch the Telerehab DSS application
Exec=$PYTHON_EXEC $PROJECT_PATH/frontEnd_register.py
Icon=$PROJECT_PATH/logo.png
Terminal=false
Type=Application
Categories=Application;Development;
EOF
) > "$DESKTOP_FILE_PATH"

update-desktop-database
echo "Desktop shortcut created."
echo

# 13. Disable Wayland for Better Compatibility
# ============================================
GDM_CUSTOM_CONF="/etc/gdm3/custom.conf"
if [ -f "$GDM_CUSTOM_CONF" ]; then
    echo "Disabling Wayland for better compatibility with remote access and GUI apps..."
    # Comment out any existing WaylandEnable line to ensure ours is the only active one
    sed -i -E 's/^(WaylandEnable=.*)/#\1/' "$GDM_CUSTOM_CONF"
    # Add our line under the [daemon] section
    sed -i '/\[daemon\]/a WaylandEnable=false' "$GDM_CUSTOM_CONF"
    echo "Wayland disabled. A reboot is required for this change to take effect."
else
    echo "GDM3 custom.conf not found. Skipping Wayland configuration."
fi
echo

# 14. Enable and Start All Services
# =================================
echo "Enabling and (re)starting Mosquitto MQTT broker with new config..."
systemctl enable mosquitto
systemctl restart mosquitto

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling all application services to start on boot..."
systemctl enable telerehab-update
systemctl enable mqtt-heartbeat
systemctl enable telerehab-main
systemctl enable telerehab-depth-camera
systemctl enable telerehab-polar

echo "Starting all application services now..."
# The update service runs once and exits. The others will start sequentially after it.
systemctl start telerehab-update
systemctl start mqtt-heartbeat
systemctl start telerehab-main
systemctl start telerehab-depth-camera
systemctl start telerehab-polar
echo

# 15. Run Initial Screen Check
# ============================
echo "Performing initial check for connected monitors..."
$CHSCREEN_SCRIPT_PATH
echo

# 16. Verify the Service Status
# ===============================
echo "--- Verification ---"
echo "Service setup is complete. Checking status of all services..."
echo

echo "--> Status of mosquitto.service:"
systemctl status mosquitto --no-pager
echo "-------------------------------------"
echo

echo "--> Status of telerehab-update.service (should be inactive/dead after running):"
systemctl status telerehab-update --no-pager
echo "-------------------------------------"
echo

echo "--> Status of mqtt-heartbeat.service:"
systemctl status mqtt-heartbeat --no-pager
echo "-------------------------------------"
echo

echo "--> Status of telerehab-main.service:"
systemctl status telerehab-main --no-pager
echo "-------------------------------------"
echo

echo "--> Status of telerehab-depth-camera.service:"
systemctl status telerehab-depth-camera --no-pager
echo "-------------------------------------"
echo

echo "--> Status of telerehab-polar.service:"
systemctl status telerehab-polar --no-pager
echo "------------------------------------"
echo

echo
echo "--- Success! ---"
echo "All services are now active and running."
echo "----------------"

