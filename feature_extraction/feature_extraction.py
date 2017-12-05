################################################################################
'''
                                KAGGLE FEATURE EXTRACTOR
    Author: Harsh Bhate
    Date: November 29, 2017
    Version: 1.01
    Description:
'''
################################################################################

import numpy as np
import os
import librosa
import librosa.display as display
import pickle
import sys
import time
import matplotlib.pyplot as plt

################################################################################
'''Declare File address here'''

dataset_root = "/home/uvuser/Documents/train/train/audio"
feature_dataset_root = "/home/uvuser/Documents/train/train/audio/pkls"
################################################################################

'''List all the audio folders in the database and returns them'''
def folder_list (dataset_root):
    files = []
    for folder in os.listdir(dataset_root):
        if (folder != "_background_noise"):
            files.append(folder)
        else:
            continue
    print "\t\tTHE FILE NAMES ARE EXTRACTED"
    for folder in files:
        print "\n\tFolder Name:",folder
    time.sleep(5)
    return files

'''Checking Audio Files'''
def check_track (clip,sr):
    max_length = (sr*1)
    length_of_clip = len(clip)
    #Checking the clip for dubious values
    if (np.isnan(np.amin(clip)) or np.isnan(np.amax(clip))):
        return np.zeros(max_length)
    #Checking if length of the clip is 2s or not
    if (length_of_clip > max_length):
        print "\n\tThe length of the clip is larger than 1s. Shortening to 1s"
        return clip[0:max_length]
    elif (length_of_clip < max_length):
        print "\n\tThe length of the clip is smaller than 1s. Appending to 1s"
        diff = max_length - length_of_clip
        clip = np.hstack ([clip,np.zeros(diff)])
        return clip
    else:
        print "\n\tThe clip is perfect!"
        return clip

'''PKL name'''
def pkl_namer(folder):
    pkl_name =  feature_dataset_root+"/"+folder+".pkl"
    print "\n\t\t PICKLE Destination:",pkl_name
    return pkl_name

'''PKL Write'''
def pkl_dump(pkl_filename,picture):
    with open(pkl_filename,'a') as f:
        pickle.dump(picture,f)

def file_runner(files):
    os.chdir (dataset_root) #Changing to the root of the dataset
    for folder in files:
        print "\n\t\tCURRENT FOLDER:",folder
        pkl_filename = pkl_namer(folder)
        os.chdir(folder)
        for filename in os.listdir(dataset_root+"/"+folder):
            if (filename.endswith(".wav") or filename.endswith(".mp3")):
                print "\tCurrent Filename:",filename
                picture = feature_extractor(filename)
                pkl_dump(pkl_filename,picture)
            else:
                continue
        os.chdir("..")
    print "\n\t\tFeature Extraction is done!"

'''Function to extract features'''
def feature_extractor(track):
    #Loading the audio track
    clip,sr = librosa.core.load(track,sr=8000)
    song = check_track(clip,sr)

    #Exctracting Features
    '''
    Feature Specification:
        Rate: 8000 Hz
        Scale Type: Mel
        Audio Track Length: 2 s
        Frame Size: 20 ms
        STFT Hop Length: 100 (System Default:512)
                        Audio Track Length
        (    Hop Length: ---------------------- )
                            Frame Size
        Frequency Bins: 128
        Number of MFCC: 64
        Number of Deltas: 2
    '''
    #MelSpectogram
    S = librosa.feature.melspectrogram(song, sr=sr,hop_length = 100,n_mels=192)
    log_S = librosa.logamplitude(S,ref_power = np.max)
    #display.specshow(log_S, y_axis='log')
    #plt.show()
    #MFCC and Deltas
    mfcc1 = librosa.feature.mfcc(y=song, sr=8000, S=log_S, n_mfcc=64)
    mfcc2 = librosa.feature.delta(mfcc1)
    mfcc3 = librosa.feature.delta(mfcc1,order=2)
    mfcc = np.vstack ([mfcc1,mfcc2,mfcc3])
    #librosa.display.specshow(mfcc, y_axis='log')
    #plt.show()
    #Array Manipulations
    log_S = log_S[:,:,np.newaxis]				#2-D to 3-D array
    log_S = log_S[:,:,:,np.newaxis]              #3-D to 4-D array
    log_S = np.transpose(log_S,(3,2,0,1))
    #print "\n\tShape of the Mel Spectorgram:",np.shape(log_S)
    mfcc = mfcc[:,:,np.newaxis]				#2-D to 3-D array
    mfcc = mfcc[:,:,:,np.newaxis]              #3-D to 4-D array
    mfcc = np.transpose(mfcc,(3,2,0,1))
    #print "\n\tShape of the MFCC Deltas:",np.shape(mfcc)
    picture = np.stack([log_S,mfcc],axis=1)
    #print "\n\tShape of picture",np.shape(picture)
    return picture
################################################################################

if __name__ == "__main__":
    files = folder_list(dataset_root)
    file_runner(files)
