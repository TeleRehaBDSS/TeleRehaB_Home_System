import numpy as np
import pandas as pd
import joblib
import sys

def give_score(metrics,id):
    if id ==1 or id==2 : # Head Left-Right & Head Up-Down
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score=0
        else :
            if metrics["total_metrics"]["std_range_degrees"] < 0.2 :
                score = 2
            else :
                score = 3                                                                                                      

    elif id == 3 or id == 6 : # Sitting and Standing Bend Over
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["number_of_movements"] < 3:
            score = 1
        elif metrics["total_metrics"]["number_of_movements"] >3 and metrics["total_metrics"]["movement_std_time"]<0.3:
            score = 2
        else:
            score = 3
    
    elif id == 43:
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["number_of_movements"] < 3:
            score = 1
        elif metrics["total_metrics"]["number_of_movements"] >3 and metrics["total_metrics"]["std_duration_seconds"]<0.3:
            score = 2
        else:
            score = 3

    elif id == 4 or id == 5: #Standing Balances
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 3
        elif metrics["total_metrics"]["number_of_movements"] <= 2:
            score = 2
        elif metrics["total_metrics"]["number_of_movements"] <= 3 :
            score = 1
        else:
            score = 0

    elif id == 7 : 
        if metrics["total_metrics"]["Total Movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["Total Movements"] <=18:
            score = 1
        elif metrics["total_metrics"]["Total Movements"] <=32:
            score = 2
        else :
            score = 3
    elif id == 8: #Forward Walking
        score = 3
    elif id == 9: #Forward Walking Yaw
        score = 3
    elif id == 10: #Forward Walking Tilt
        score = 3
    elif id == 14: #Trunk Rotation
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["number_of_movements"] <= 10:
            score = 1
        elif metrics["total_metrics"]["number_of_movements"] >= 20:
            score = 3
        else :
            score =2
    elif id == 15: #Toe Raises
        if metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] == 0 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] == 0 :
            score = 0
        elif metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] <= 7 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] <= 7:
            score = 1
        elif (metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] >= 15 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] <= 15) and (metrics["total_metrics"]["RIGHT LEG"]["mean_combined_range_degrees"] <= 0.5 or metrics["total_metrics"]["LEFT LEG"]["mean_combined_range_degrees"] <= 0.5):
            score = 2
        else :
            score = 3

    elif id == 16: #Heel Raises
        if metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] == 0 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] == 0 :
            score = 0
        elif metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] <= 7 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] <= 7:
            score = 1
        elif (metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] >= 15 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] <= 15) and (metrics["total_metrics"]["RIGHT LEG"]["mean_combined_range_degrees"] <= 0.5 or metrics["total_metrics"]["LEFT LEG"]["mean_combined_range_degrees"] <= 0.5):
            score = 2
        else :
            score = 3

    elif id == 17: #Seated Marching on the spot
        if metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] == 0 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] <= 5 and metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] <= 5:
            score = 1
        elif (metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] >= 10 and metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] >= 10) and (metrics["total_metrics"]["RIGHT LEG"]["mean_combined_range_degrees"] <= 0.5 or metrics["total_metrics"]["LEFT LEG"]["mean_combined_range_degrees"] <= 0.5):
            score = 2
        else :
            score = 3
    elif id == 18: # Sit to Stand
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["number_of_movements"] <= 10:
            score = 1
        elif metrics["total_metrics"]["number_of_movements"] >= 20 and metrics["total_metrics"]["mean_range_degrees"] <= 0.5:
            score = 2
        else :
            score =3

    elif id == 19: #Lateral Weight Shifts
        if metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] == 0 or metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] <= 2 and metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] <= 2:
            score = 1
        elif (metrics["total_metrics"]["RIGHT LEG"]["number_of_movements"] >= 5 and metrics["total_metrics"]["LEFT LEG"]["number_of_movements"] >= 5) :
            score = 2
        else :
            score = 3

    elif id == 20: #Limits of Stability
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["number_of_movements"] <= 5:
            score = 1
        elif metrics["total_metrics"]["number_of_movements"] >= 5 and metrics["total_metrics"]["mean_combined_range_degrees"] <= 0.5:
            score = 2
        else :
            score =3

    elif id == 21: #Forward Reach
        if metrics["total_metrics"]["number_of_movements"] == 0:
            score = 0
        elif metrics["total_metrics"]["number_of_movements"] <= 10:
            score = 1
        elif metrics["total_metrics"]["number_of_movements"] >= 20 and metrics["total_metrics"]["mean_combined_range_degrees"] <= 0.5:
            score = 2
        else :
            score =3

    elif id == 22: #Side Stepping
        score = 3
    elif id == 20: #Walking Horizontal Head Turns
        score = 3
    else :
        score = 3
    return score

from scoring import give_score  


sys.modules['__main__'].give_score = give_score

def give_score_AI(metrics, id):
    # Load the function from the joblib file
    loaded_give_score = joblib.load('give_score_AI.joblib')
    return loaded_give_score(metrics, id);


