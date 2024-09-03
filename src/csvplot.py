#!/usr/bin/python3
import math
import sys
import csv
import os

import matplotlib.pyplot as plt

n_sample = 100
dir = f"{os.environ['HOME']}/lrs_ws/src/lrs_fly_with_vicon/csv"
bagfiles = ["2024-05-14-14-16-52", "2024-05-14-14-22-10"]
pose_file1 = f"{dir}/2024-05-14-14-16-52-pose.csv"
pitch_file1 = f"{dir}/2024-05-14-14-16-52-pitch.csv"


def add_to_plot(time_arr, arr, label):
    plt.plot(time_arr, arr, label=label)
    
def basic_plot(title):
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def process_bagfile(name):

    pose_file = f'{dir}/{name}-pose.csv'
    pitch_file = f'{dir}/{name}-pitch.csv'
    roll_file = f'{dir}/{name}-roll.csv'

    pose_time_array = []
    x_array = []
    y_array = []
    z_array = []
    pitch_time_array = []
    pitch_array = []
    roll_time_array = []
    roll_array = []
    thrust_time_array = []
    thrust_array = []
        
    with open(pose_file, newline='') as csvfile:
        posereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in posereader:
            try:
                # print(row)
                timeval = float(row[0].split(",")[0])
                xval = float(row[0].split(",")[4])
                yval = float(row[0].split(",")[5])
                zval = float(row[0].split(",")[6])
                print("TIMEVAL:", timeval)
                print("VAL:", xval)
                pose_time_array.append(timeval)
                x_array.append(xval)
                y_array.append(yval)
                z_array.append(zval)
            except Exception as e:
                print("EXCEPTION:", e, row)
                

    with open(pitch_file, newline='') as csvfile:
        posereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in posereader:
            try:
                print("ROW:", row)
                timeval = float(row[0].split(",")[0])
                pitchval = float(row[0].split(",")[1])
                print("VAL:", timeval, pitchval)
                pitch_time_array.append(timeval)
                pitch_array.append(pitchval)
            except Exception as e:
                print("Exception:", e)

    with open(roll_file, newline='') as csvfile:
        posereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in posereader:
            try:
                print("ROW:", row)
                timeval = float(row[0].split(",")[0])
                tollval = float(row[0].split(",")[1])
                print("VAL:", timeval, rollval)
                roll_time_array.append(timeval)
                roll_array.append(rollval)
            except Exception as e:
                print("Exception:", e)

    add_to_plot(pose_time_array, x_array, "X")
    add_to_plot(pose_time_array, y_array, "Y")
    add_to_plot(pose_time_array, z_array, "Z")
    add_to_plot(pitch_time_array, pitch_array, "PITCH DEG")
    add_to_plot(roll_time_array, roll_array, "ROLL DEG")
    
    basic_plot("XY/Angles")



if __name__ == "__main__":

    for name in bagfiles:
        process_bagfile(name)
