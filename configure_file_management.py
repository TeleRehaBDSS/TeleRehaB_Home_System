from shared_variables import imus

def read_configure_file():
    global imus
    with open('/home/uoi/Documents/GitHub/Telerehab_UOI/WP3_v1/imu_mqtt/configure.txt', 'r') as file:
        content = file.read()
        lines = content.splitlines()
        for line in lines:
            parts = line.split()
            if(parts[0] == 'IMU'):
                imus.extend(parts[1:]) 