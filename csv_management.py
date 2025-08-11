import csv
import os
from datetime import datetime
from shared_variables import firstPacket, csv_file_path, imus

def create_csv_files(imus, csv_file_path, isGyroscope):
    currentTime = datetime.now()
    currentTime = currentTime.strftime("%Y-%m-%d_%H:%M:%S")
    folderPath = f"results/{currentTime}" 

    for imu in range(len(imus)):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        csv_file_path.append(f"{folderPath}/{imus[imu]}_{currentTime}.csv")
        if(isGyroscope):
            headers = ["Device", "MAC", "Timestamp", "Time(03:00)", "Elapsed(s)", "X(number)", "Y(number)", "Z (number)"]
        else:
            headers = ["Device", "MAC", "Timestamp", "Time(03:00)", "Elapsed(s)", "W(number)", "X(number)", "Y (number)", "Z (number)"]
        with open(csv_file_path[imu], 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)

def write_in_a_specific_file(deviceData, csv_file_path):
    global firstPacket
    device = ""
    isGyroscope = False

    if(len(deviceData) > 0):
        firstData = deviceData[0].split()
        device = firstData[1] 
        if("num" not in firstData[len(firstData) - 1].replace('"', '').strip()):   # not the header
            if(len(firstData) == 8 ):
                isGyroscope = True
            elif(len(firstData) == 9 ):
                isGyroscope = False
    
    if (firstPacket.value == True):
        firstPacket.value = False
        create_csv_files(imus, csv_file_path, isGyroscope)
     
    for allData in deviceData:
        device = csv_file_path[imus.index(allData.split()[1])]
        with open(device, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            allData = allData[1:len(allData)-1]
            csv_writer.writerow(allData.split())

def write_in_files(imu1List, imu2List, imu3List, imu4List):
    write_in_a_specific_file(imu1List, csv_file_path)
    write_in_a_specific_file(imu2List, csv_file_path)
    write_in_a_specific_file(imu3List, csv_file_path)
    write_in_a_specific_file(imu4List, csv_file_path)