#!/bin/bash
function check_net {
  ping -q -w 1 -c 1 google.com &>/dev/null && return 0 || return 1
}
check_net
if [ $? -eq 0 ]; then
   echo "Network error, fix it and run this script again."
   exit 1
fi
echo "This script installs headless monitor!"
if ! [ $(id -u) = 0 ]; then
   echo "The script need to be run with sudo" >&2
   exit 1
fi

XORG_BAK_FILE="/root/xorg.conf"
XORG_WRK_FILE="/usr/share/X11/xorg.conf.d/xorg.conf"
CHSCREEN_SCRIPT_PATH="/root/chScreen.sh"

echo "Setting up virtual screen configuration..."

(cat <<'EOF'
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

(cat <<'EOF'
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


if [ $(crontab -l | grep -c $CHSCREEN_SCRIPT_PATH) -eq 0 ]; then
    # echo my crontab is not installed
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -u root -
fi

echo "Everything is ready!!!"
