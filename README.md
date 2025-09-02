# TeleRehaB Home System — Technician Installation (v2.5)

**Audience:** Trained technicians installing/configuring TeleRehaB in a patient’s home.

---

## Contents
- [Overview](#overview)
- [Disclaimer](#disclaimer)
- [Support](#support)
- [Pre-Installation Checklist](#pre-installation-checklist)
- [Part 1: Dedicated Network Setup](#part-1-dedicated-network-setup)
- [Part 2: Edge Computer Setup](#part-2-edge-computer-setup)
  - [Ubuntu 22.04 LTS](#ubuntu-2204-lts)
  - [Automated Setup Script](#automated-setup-script)
  - [RustDesk Configuration](#rustdesk-configuration)
  - [Virtual Screen Setup](#virtual-screen-setup)
- [Part 3: Mobile / Tablet Setup](#part-3-mobile--tablet-setup)
  - [Initial Device Setup](#initial-device-setup)
  - [Install Apps (.apk)](#install-apps-apk)
  - [Keep-Screen-On Config](#keep-screen-on-config)
- [Part 4: Final Assembly & Patient Handoff](#part-4-final-assembly--patient-handoff)
  - [Hardware Connections](#hardware-connections)
  - [Register Patient to Device Set](#register-patient-to-device-set)
  - [System Test & Patient Instructions](#system-test--patient-instructions)
  - [Important Notes](#important-notes)
- [Version](#version)

---

## Overview
This README gives step-by-step hardware and software setup for the TeleRehaB Decision Support System (DSS): **Edge Computer**, **Mobile/Tablet**, and **peripheral sensors**.

## Disclaimer
Use only under clinician guidance and prescription. For assessed patients only. Unauthorized use is prohibited.

## Support
- **Panagiotis Lionas**
- **Vasilis Tsakanikas**

---

## Pre-Installation Checklist

### Hardware Kit
- Edge Computer (NUC or similar) + power
- Access Point (TP-Link TL-WA1201) + power
- USB Depth Camera
- Monitor (for setup), USB mouse, USB keyboard
- BLE USB adapter (for Polar)
- Mobile phone or tablet (designated for home use)
- Bootable USB (≥8 GB) with **Ubuntu 22.04 LTS** installer

### Software & Credentials
- Technician credentials for **Add Patient** app
- RustDesk relay server details (ID/Relay/Key):  
  https://public.3.basecamp.com/p/y18PZsWYGXrJxUApzJbKzedq

---

## Part 1: Dedicated Network Setup
1. Connect **TP-Link AP** WAN to a spare LAN on the home router (Ethernet).
2. Power on AP; wait ~1 min. Default **Access Point Mode** will broadcast Wi-Fi.
3. Note SSID and Password from the AP sticker (bottom).  
4. Ref manual (if needed):  
   https://static.tp-link.com/upload/manual/2024/202406/20240611/7106511220_TL-WA1201(EU)_QIG_V1.pdf

---

## Part 2: Edge Computer Setup

### Ubuntu 22.04 LTS
1. Connect monitor/mouse/keyboard.
2. Boot from Ubuntu USB ➜ **Install Ubuntu**.
3. Connect to the **dedicated AP Wi-Fi**.
4. **Installation type:** Erase disk and install Ubuntu.
5. **Standard credentials:**
   - Username: `telerehab`
   - Password: `telerehab`
6. Reboot; remove USB.

### Automated Setup Script
After first login:

```bash
# 1) Open Terminal
# 2) Download setup script to Desktop
wget -O ~/Desktop/Set_UP_Tele.sh https://raw.githubusercontent.com/TeleRehaBDSS/TeleRehaB_Home_System/main/Set_UP_Tele.sh

# 3) Make executable and run with sudo
cd ~/Desktop
sudo chmod +x Set_UP_Tele.sh
sudo ./Set_UP_Tele.sh
```

> Let the script finish before proceeding. It installs all required software (incl. RustDesk).

### RustDesk Configuration
1. Open relay details page:  
   https://public.3.basecamp.com/p/y18PZsWYGXrJxUApzJbKzedq/vault/9019437575
2. Launch **RustDesk** → **Settings → Network** → **Unlock** → set **ID Server**, **Relay Server**, **Key**.  
3. **Reboot** the edge PC (required to apply and get a **permanent ID**).  
4. Re-open RustDesk; **record the permanent ID**.  
5. **Unattended access:** Settings → Security → Unlock → set permanent password: `telerehaB1`; **Permission profile:** *Full Access*.

### Virtual Screen Setup
```bash
wget -O ~/Desktop/screensetup.sh https://raw.githubusercontent.com/TeleRehaBDSS/TeleRehaB_Home_System/refs/heads/main/screensetup.sh
cd ~/Desktop
sudo chmod +x screensetup.sh
sudo ./screensetup.sh
```

---

## Part 3: Mobile / Tablet Setup

### Initial Device Setup
- Connect to the **same AP Wi-Fi** as the edge PC.
- Enable **Bluetooth**, **Location**, **Wi-Fi**.

### Install Apps (.apk)
Allow installs from unknown sources where needed.

1. **TeleRehab VC App (Virtual Coach)**  
   - High-Tech (Mobile): https://public.3.basecamp.com/p/y18PZsWYGXrJxUApzJbKzedq/vault/8987552890  
   - Low-Tech (Tablet): https://public.3.basecamp.com/p/y18PZsWYGXrJxUApzJbKzedq/vault/8987653747  
   - Grant permissions: Phone, Location, Nearby Devices, Camera, Mic, Notifications  
   - Battery: **Unrestricted**

2. **TeleRehab Session Starter (IMU App)**  
   - https://public.3.basecamp.com/p/y18PZsWYGXrJxUApzJbKzedq/vault/8989267871  
   - Permissions: Phone, Location, Nearby Devices, Notifications  
   - Battery: **Unrestricted**

3. **RustDesk (Android)**  
   - https://public.3.basecamp.com/p/y18PZsWYGXrJxUApzJbKzedq/vault/8987552890  
   - Get server creds: vault/9019437575  
   - App Settings → **ID/Relay Server** → set values → **OK**  
   - Enable **Start on boot**  
   - Set permanent password: `telerehaB1`  

> Note: Stop the RustDesk service after installation. The patient will start it manually only for troubleshooting.

### Keep-Screen-On Config
From Google Play:

- **Screen Alive**  
  - Grant modify system settings  
  - Set **Always** (keep screen on)
- **AutoStart App Manager**  
  - Grant “Display over other apps”  
  - Enable **AutoStart**, add **Screen Alive**, then **reboot** device

---

## Part 4: Final Assembly & Patient Handoff

### Hardware Connections
- Plug **USB depth camera** into a **USB3 (blue)** port on edge PC.
- Plug **BLE adapter** into a USB port.

### Register Patient to Device Set
**On Edge PC**
1. Launch **Add Patients** app.
2. Login with clinician credentials → **Get DATA**.
3. Select patient → **Save API Key** → **OK** → note **Patient ID**.

**On Mobile/Tablet**
1. Open **TeleRehab VC App**.
2. Enter the noted **Patient ID** in settings/login.
3. Tap **Connect**.

### System Test & Patient Instructions
1. Instruct patient to **wear sensors**.
2. On mobile/tablet, open **Session Starter (IMU App)**; place device in mount/stand.
3. Power on edge PC; explain ~**50s** boot.
4. Run **one test exercise**; verify connectivity.
5. Perform a **clean shutdown** of the edge PC.

### Important Notes
- Keep **both** Edge PC and Mobile/Tablet on the **dedicated AP Wi-Fi** only.
- AP must stay connected to the home router with **internet access**.
- **Do not** switch either device to other Wi-Fi networks. Advise patient not to change network settings.

---

## Version
**2.5 — Technician Installation Guide (adapted for README).**

*Source: “TeleRehab System — Technician Installation Guide” v2.5.*


### Patient Quick-Start (Give this to the patient)
- Turn on the **white Wi‑Fi box** (AP) and keep it near the TV/router.
- Turn on the **small PC** (edge computer). Wait ~50 seconds.
- Open the **TeleRehab app** on the phone/tablet and press **Connect**.
- Follow the voice/video instructions. Wear sensors as shown by the clinician.
- If something looks wrong, call support (see numbers above).

---

## Part 5: Troubleshooting & Diagnostics

### Quick checks (most issues)
- **Power:** AP and edge PC LEDs on? Phone has battery?  
- **Wi‑Fi:** Phone/tablet and edge PC on the **same AP SSID**?  
- **Internet:** AP Ethernet cable plugged into home router LAN?  
- **RustDesk:** Can you see the edge PC online? If not, reboot edge PC + phone.

### Edge PC diagnostics (Ubuntu)
```bash
# Network basics
nmcli device status
nmcli dev wifi
ping -c 4 8.8.8.8
curl ifconfig.me

# See IPs and routes
ip a
ip route

# USB camera + BLE dongle presence
lsusb

# Kernel events when (re)plugging devices
dmesg --ctime | tail -n 50

# List video devices (if v4l2-utils is installed)
v4l2-ctl --list-devices 2>/dev/null || echo "Install v4l2-utils for camera diagnostics"
```

### Android quick checks
- **Permissions:** Settings → Apps → TeleRehab apps → Permissions: allow all requested.  
- **Battery:** Set to **Unrestricted**.  
- **Bluetooth & Location:** Turn **ON**.  
- **Wi‑Fi:** Must be the AP SSID (not home Wi‑Fi).

### RustDesk (remote support)
- On **Android**: open RustDesk → confirm **ID/Relay Server** and **Key** are set → note the **ID**.  
- On **Edge PC**: open RustDesk → Settings → Network → confirm servers/Key → note the **permanent ID**.  
- If IDs change or show **Offline**, **reboot** both devices.

### Common issues → quick fixes
- **Phone can’t connect to app** → Toggle Wi‑Fi off/on, then reopen app.  
- **No sensors detected** → Ensure Bluetooth **ON**, re-seat BLE USB on edge PC, reboot phone.  
- **No camera feed** → Try other USB3 port; check `lsusb`; reboot edge PC.  
- **Remote help fails** → Verify AP has internet (ping), confirm RustDesk server fields.

---

## Part 6: Maintenance & Updates

### Routine (monthly)
```bash
# Edge PC
sudo apt update && sudo apt upgrade -y
sudo reboot
```

- Re‑run the **Set_UP_Tele.sh** when notified of updates.  
- Keep **RustDesk** current on both devices.  
- Android: apply OS/app updates; re‑check permissions afterwards.

### Backups
- Note and store: **AP SSID/password**, **RustDesk IDs**, **Patient ID**.  
- Keep the **Ubuntu installer USB** in the kit.

---

## Part 7: Security & Privacy

- Use the dedicated AP; avoid mixing with home Wi‑Fi.  
- Do not share patient IDs publicly.  
- Use provided passwords only; change if exposure is suspected.  
- Remote access strictly for clinical/technical support with patient consent.  
- Follow local data protection policies and clinical guidance.

---

## Part 8: Uninstall / Recovery

### Remove app autostarts (Android)
- Settings → Apps → (each TeleRehab app) → Disable Autostart/Background if needed.  
- Uninstall via Play Store or App settings.

### Edge PC reset options
```bash
# Stop suspected user services
systemctl --failed
# Example: view logs (replace <service> with actual name if applicable)
journalctl -u <service> -e --no-pager
```

- To fully reimage: boot from Ubuntu USB → **Erase disk and install Ubuntu** → rerun setup scripts.

### AP factory reset
- Use the pinhole **RESET** button (press ~8–10s). Reconfigure SSID and password.

---

## Part 9: Handoff Checklist (Technician)

- [ ] AP powered, Ethernet to router OK  
- [ ] Edge PC boots and auto‑connects to AP  
- [ ] Phone/tablet on AP; Battery **Unrestricted**; permissions granted  
- [ ] TeleRehab apps launch and connect  
- [ ] Sensors detected; test exercise runs  
- [ ] RustDesk IDs recorded (PC + phone)  
- [ ] Patient trained with **Quick‑Start** above  
- [ ] Support contacts shared

```text
Record here:
AP SSID: ______________________    AP Pass: ______________________
Edge PC RustDesk ID: __________    Android RustDesk ID: __________
Patient ID: ____________________   Install Date: _________________
Technician: ____________________   Notes: ________________________
```

---

## Part 10: Ports & Network Notes (info)

- Internet via home router → Ethernet to AP → Wi‑Fi to devices.  
- Remote support uses RustDesk relay/ID servers (outbound HTTPS/TCP).  
- If the home router blocks outbound traffic, ask the ISP to allow standard outbound ports.

---

## License & Credits

© TeleRehaB Consortium. Internal deployment guide for clinical pilots.
