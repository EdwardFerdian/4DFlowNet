"""
This script mainly contains functions needed for logging
Author: Edward Ferdian
Date:   27/02/2019
"""
from datetime import datetime
from time import time

def calculate_time_elapsed(start):
    '''
        This function calculates the time elapsed
        Input:  
            start = start time
        Output: 
            hrs, mins, secs = time elapsed in hours, minutes, seconds format
    '''
    end = time()
    hrs = (end-start)//60//60
    mins = ((end-start) - hrs*60*60)//60
    secs = int((end-start) - mins*60 - hrs*60*60)

    return hrs, mins, secs

def log_to_file(filepath, msg):
    with open(filepath, 'a') as f:
        f.write(msg)