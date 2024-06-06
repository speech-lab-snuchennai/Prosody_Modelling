#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:24:29 2023

@author: speechlab
"""

import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import seaborn as sns
import statistics
# from flask import *
import os
import time
import scipy.io.wavfile as wav
from g2p_en import G2p
import pandas as pd
import itertools
import more_itertools as mit
from itertools import groupby
from statistics import mean
import requests
import json
import wave
import requests
from parselmouth.praat import call
# from transformers import pipeline
# from jiwer import wer
import requests
import json
import os
from tabulate import tabulate
# import speech_recognition as sr
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# from ai4bharat.transliteration import XlitEngine
from itertools import zip_longest
import sys


# ddd_phoneme, ddd_syllable, ddd_word, ddd_intensity, ddd_break, ddd_tobi


def full_code(filename, language):
    f1 = filename.split('\\')
    file = f1[-1].split('.')[0]
    file_pit = filename.split('/')[-1].split('.')[0]
    print(file_pit)
    os.system("ch_wave " + filename + " -F 16000 -otype wav -o temp.wav")
    os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
    ###############################################################################
    ###############################################################################
    ##########################Speech to Text#######################################
    if language == 'English':
        headersList = {
            "Authorization": f"Token {token}"
        }
        files = {
            'file': open('temp.wav', 'rb'),
            'language': (None, 'english'),
            'vtt': (None, 'true'),
        }
        response = requests.post('https://asr.iitm.ac.in/api/asr/', files=files,
                                 headers=headersList)
        text = response.json().get("transcript")
        print(text)
        outf = open("text.txt", "w")
        for lines in text:
            outf.write(lines)
            # outf.write("\n")
        outf.close()
        os.system("tr -d '[:punct:]' <text.txt >temp.txt")
        ###############################################################################
        ###############################################################################
        #####################PHONEME SEGMENTATION######################################
        os.system("rm -r data")
        os.system("mkdir data")
        # os.system("cp -r temp.wav data/")
        os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
        # os.system("cp temp.wav data/")
        os.system("mv temp.txt data")
        # os.system("mv temp.wav data")
        # os.system("rm *.wav")
        # os.system("./scripts/ortho_to_phonetic1 data/temp.txt phoneset_smp >data/temp.lab")
        # texts = ["She had your dark suit in greasy wash water all year"]
        with open("data/temp.txt") as f:
            texts = f.read()
            # print(texts)
        g2p = G2p()
        out = g2p(texts)
        # res = [ele for ele in out if ele.strip()]
        outf = open("file.txt", "w")
        for lines in out:
            outf.write(lines)
            outf.write("\n")
        outf.close()
        os.system("./scripts/map_eng file.txt english_map")
        os.system("ls -v data/*.wav >data/wavlist")
        os.system("ls -v data/*.lab >data/lablist")
        os.system("./scripts/create_maptable data/wavlist")
        os.system("HCopy -C config_files/speech_config_file -S data/maptable")
        os.system("cat data/*.lab | sort | uniq >data/wordlist")
        os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
        os.system("./scripts/mlf data/lablist >data/mlf")
        os.system("HVite -a -o SW -I data/mlf -H all_hmmE dictE data/wordlist data/temp.mfc")
        os.system("./scripts/phoneme data/temp.rec")
        ###############################################################################
        ###############################################################################
        ########################SYLLABLE SEGMENTATION##################################
        os.system("tr -d '\n' <data/temp.lab >data/temp")
        os.system("./scripts/ortho_to_phonetic_cv_eng data/temp con_c_v_cv_list_eng >data/t.led")
        os.system("cp data/temp.rec data/temp_syl.lab")
        os.system("HLEd data/t.led data/temp_syl.lab")
        os.system("./scripts/ortho_to_phonetic_syllable_eng data/temp con_c_v_cv_list_eng >data/tempp")
        os.system("./scripts/syllable_from_char_eng data/tempp v_list_eng c_list_eng CV_list_eng >data/temp.led")
        os.system("HLEd data/temp.led data/temp_syl.lab")
        os.system("./scripts/syllable data/temp_syl.lab")
        ###############################################################################
        ###############################################################################
        ##########################WORD SEGMENTATION####################################
        # with open("data/temp.txt") as f:
        #     texts =  f.read()
        #     # print(texts)
        # g2p = G2p()
        # out = g2p(texts)
        # # res = [ele for ele in out if ele.strip()]
        # outf=open("file1.txt", "w")
        # for lines in out:
        #     outf.write(lines)
        #     outf.write("\n")
        # outf.close()
        # os.system("./scripts/map_eng file.txt english_map")
        # os.system("./scripts/word_segment data/temp.lab data/temp.txt")
        os.system("./scripts/word_segment_modified data/temp.rec data/temp.txt")
    elif language == 'Tamil':
        headersList = {
            "Authorization": f"Token {token}"
        }
        files = {
            'file': open('temp.wav', 'rb'),
            'language': (None, 'tamil'),
            'vtt': (None, 'true'),
        }
        response = requests.post('https://asr.iitm.ac.in/api/asr/', files=files,
                                 headers=headersList)
        text = response.json().get("transcript")
        print(text)
        outf = open("te.txt", "w")
        for lines in text:
            outf.write(lines)
        outf.close()
        os.system("tr -d '[:punct:]' <te.txt >t.txt")
        ###############################################################################
        ###############################################################################
        #####################PHONEME SEGMENTATION######################################
        os.system("rm -r data")
        os.system("mkdir data")
        # os.system("cp -r temp.wav data/")
        os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
        # os.system("mv temp.txt data")
        # os.system("mv temp.wav data")
        # os.system("rm *.wav")
        # os.system("./scripts/ortho_to_phonetic1 data/temp.txt phoneset_smp >data/temp.lab")
        # texts = ["She had your dark suit in greasy wash water all year"]
        os.system("sed -i 's/ /sil/g' t.txt")
        os.system("perl scripts/vuv.pl t.txt")
        os.system("cp -r lists/out_word temp.txt")
        os.system("sed -i 's/sil/ /g' temp.txt")
        os.system("sed -i 's/sil/SIL/g' lists/out_word")
        os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word phonelist_tamil >p")
        os.system("./scripts/startsil p")
        os.system("mv temp.txt data")
        # os.system("./scripts/tam_eng file.txt tamil_map")
        os.system("ls -v data/*.wav >data/wavlist")
        os.system("ls -v data/*.lab >data/lablist")
        os.system("./scripts/create_maptable data/wavlist")
        os.system("HCopy -C config_files/speech_config_file -S data/maptable")
        os.system("cat data/*.lab | sort | uniq >data/wordlist")
        os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
        os.system("./scripts/mlf data/lablist >data/mlf")
        os.system("HVite -a -o SW -I data/mlf -H all_hmmT dictT data/wordlist data/temp.mfc")
        os.system("./scripts/phoneme data/temp.rec")
        ###############################################################################
        ###############################################################################
        ########################SYLLABLE SEGMENTATION##################################
        # # os.system("./scripts/create_syl.sh c_list v_list data/temp.rec")
        # os.system("./scripts/syllabification_phone_only_tamil data/temp.rec v_list")
        # # os.system("./scripts/syllabification_phone_only_modified data/temp.rec v_list")
        # # os.system("./scripts/try1 data/temp.rec v_list c_list CV_list")
        # # os.system("./scripts/syl_lab data/temp.rec c_list v_list CV_list ")
        os.system("tr -d '\n' <data/temp.lab >data/temp")
        # os.system("sed -i 's/SIL/ /g' data/temp")
        os.system("./scripts/ortho_to_phonetic_cv_tam data/temp con_c_v_cv_list_tam >data/t.led")
        os.system("cp data/temp.rec data/temp_syl.lab")
        os.system("HLEd data/t.led data/temp_syl.lab")
        os.system("./scripts/ortho_to_phonetic_syllable_tam data/temp con_c_v_cv_list_tam >data/tempp")
        os.system("./scripts/syllable_from_char_tam data/tempp v_list_tam c_list_tam CV_list_tam >data/temp.led")
        os.system("HLEd data/temp.led data/temp_syl.lab")
        os.system("./scripts/syllable data/temp_syl.lab")
        # os.system("./scripts/syllable data/temp_syl.lab")
        ###############################################################################
        ###############################################################################
        ##########################WORD SEGMENTATION####################################
        # with open("data/temp.txt") as f:
        #     texts =  f.read()
        #     # print(texts)
        # g2p = G2p()
        # out = g2p(texts)
        # # res = [ele for ele in out if ele.strip()]
        # outf=open("file1.txt", "w")
        # for lines in out:
        #     outf.write(lines)
        #     outf.write("\n")
        # outf.close()
        # os.system("./scripts/map_eng file.txt english_map")
        # os.system("./scripts/word_segment data/temp.lab data/temp.txt")
        os.system("./scripts/word_segment_modified_tamil data/temp.rec data/temp.txt")
        # os.system("./scripts/word_segment data/temp.lab data/temp.txt")
    elif language == 'Hindi':
        headersList = {
            "Authorization": f"Token {token}"
        }
        files = {
            'file': open('temp.wav', 'rb'),
            'language': (None, 'hindi'),
            'vtt': (None, 'true'),
        }
        response = requests.post('https://asr.iitm.ac.in/api/asr/', files=files,
                                 headers=headersList)
        text = response.json().get("transcript")
        print(text)
        outf = open("te.txt", "w")
        for lines in text:
            outf.write(lines)
        outf.close()
        os.system("tr -d '[:punct:]' <te.txt >temp.txt")

        ###############################################################################
        ###############################################################################
        #####################PHONEME SEGMENTATION######################################
        os.system("rm -r data")
        os.system("mkdir data")
        # os.system("cp -r temp.wav data/")
        os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
        # os.system("mv temp.txt data")
        # os.system("mv temp.wav data")
        # os.system("rm *.wav")

        os.system("./scripts/text_word temp.txt")
        os.system("tr -d [[:punct:]] <Perl/wordpronunciation >data/t_wrd.txt")
        os.system("sed -i 's/0//g' data/t_wrd.txt")
        os.system("sed -i 's/ //g' data/t_wrd.txt")
        os.system("tr '\n' ' ' <data/t_wrd.txt >data/tt_wrd.txt")
        os.system("cp -r data/tt_wrd.txt data/temp.txt")
        os.system("sed -i 's/ /SIL/g' data/tt_wrd.txt")
        os.system(
            "./scripts/ortho_to_phonetic1_phoneme_HIN data/tt_wrd.txt phonelist_hindi >>data/te")
        os.system(
            'awk \'{ for (i=1; i<=NF; i++) if ($i == "e") $i = "ee" } 1\' data/te >data/tem')
        os.system("./scripts/startsil_HIN data/tem")
        # os.system("mv temp.wav data")
        os.system("ls -v data/*.wav >data/wavlist")
        os.system("ls -v data/*.lab >data/lablist")
        os.system("./scripts/create_maptable data/wavlist")
        os.system("HCopy -C config_files/speech_config_file -S data/maptable")
        os.system("cat data/*.lab | sort | uniq >data/wordlist")
        os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
        os.system("./scripts/mlf data/lablist >data/mlf")
        os.system("HVite -a -o SW -I data/mlf -H all_hmmH dictH data/wordlist data/temp.mfc")
        os.system("./scripts/phoneme data/temp.rec")
        ###############################################################################
        ###############################################################################
        ########################SYLLABLE SEGMENTATION##################################
        # # os.system("./scripts/create_syl.sh c_list v_list data/temp.rec")
        # os.system("./scripts/syllabification_phone_only data/temp.rec v_list")
        # # os.system("./scripts/syllabification_phone_only_modified data/temp.rec v_list")
        # # os.system("./scripts/try1 data/temp.rec v_list c_list CV_list")
        # # os.system("./scripts/syl_lab data/temp.rec c_list v_list CV_list ")
        # os.system("./scripts/syllable data/temp_syl.lab")
        os.system("tr -d '\n' <data/temp.lab >data/temp")
        os.system("./scripts/ortho_to_phonetic_cv_HIN data/temp con_c_v_cv_H_list >data/t.led")
        os.system("cp data/temp.rec data/temp_syl.lab")
        os.system("HLEd data/t.led data/temp_syl.lab")
        os.system("./scripts/ortho_to_phonetic_syllable_HIN data/temp con_c_v_cv_H_list >data/tempp")
        os.system("./scripts/syllable_from_char_HIN data/tempp v_H_list c_H_list CV_H_list >data/temp.led")
        os.system("HLEd data/temp.led data/temp_syl.lab")
        os.system("./scripts/syllable data/temp_syl.lab")
        ###############################################################################
        ###############################################################################
        ##########################WORD SEGMENTATION####################################
        # with open("data/temp.txt") as f:
        #     texts =  f.read()
        #     # print(texts)
        # g2p = G2p()
        # out = g2p(texts)
        # # res = [ele for ele in out if ele.strip()]
        # outf=open("file1.txt", "w")
        # for lines in out:
        #     outf.write(lines)
        #     outf.write("\n")
        # outf.close()
        # os.system("./scripts/map_eng file.txt english_map")
        # os.system("./scripts/word_segment data/temp.lab data/temp.txt")
        os.system("./scripts/word_segment_modified_HIN data/temp.rec data/temp.txt")
    fp = open('data/temp_ph.lab', "r")
    lines = fp.readlines()
    starttimes = []
    endtimes = []
    phoneme = []
    for x in lines:
        starttimes.append(x.split()[0])
        endtimes.append(x.split()[1])
        phoneme.append(x.split()[2])
    fp.close()
    # print(starttimes)
    # print(endtimes)
    floats, floats_e = [eval(x) for x in starttimes], [eval(x) for x in endtimes]
    # print()

    seg = []
    for s in floats:
        seg.append(s / 10000000)
    # print(seg)
    seg_e = []
    for s in floats_e:
        seg_e.append(s / 10000000)
    # print(seg_e)
    d_phoneme = []
    for n, v, p in zip(seg, seg_e, phoneme):
        d_phoneme.append("{} {} {}".format(n, v, p))
    p_table = open("data/Phoneme_t.txt", "w")
    for lines in d_phoneme:
        p_table.write(lines)
        p_table.write("\n")
    p_table.close()
    filename_phoneme = "data/Phoneme_t.txt"
    ddd_phoneme = pd.read_csv(filename_phoneme, header=None, sep="\s{1,}")
    ddd_phoneme.columns = ['Starttime', 'Endtime', 'Phoneme']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # df1.to_csv(filename)
    print(ddd_phoneme)
    out_phoneme = file + ".phn"
    out_phoneme1 = open(out_phoneme, "w")
    for lines in str(ddd_phoneme):
        out_phoneme1.write(lines)
        # outf.write("\n")
    out_phoneme1.close()

    fs = open('data/temp_syl_syl.lab', "r")
    lines1 = fs.readlines()
    starttimes1 = []
    endtimes1 = []
    syllable = []
    for x1 in lines1:
        starttimes1.append(x1.split()[0])
        endtimes1.append(x1.split()[1])
        syllable.append(x1.split()[2])
    fs.close()
    # print(starttimes1)
    floats1, floats1_e = [eval(x1) for x1 in starttimes1], [eval(x1) for x1 in endtimes1]
    # print(floats1)
    syl_id = [x1 for x1 in syllable]
    seg1 = []
    for s1 in floats1:
        seg1.append(s1 / 10000000)
    # print(seg1)
    seg1_e = []
    for s1 in floats1_e:
        seg1_e.append(s1 / 10000000)
    d_syllable = []
    for n1, v1, p1 in zip(seg1, seg1_e, syllable):
        d_syllable.append("{} {} {}".format(n1, v1, p1))
    s_table = open("data/syllable_t.txt", "w")
    for lines in d_syllable:
        s_table.write(lines)
        s_table.write("\n")
    s_table.close()
    filename_syllable = "data/syllable_t.txt"
    ddd_syllable = pd.read_csv(filename_syllable, header=None, sep="\s{1,}")
    ddd_syllable.columns = ['Starttime', 'Endtime', 'Syllable']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # df1.to_csv(filename)
    print(ddd_syllable)
    out_syllable = file + ".syl"
    out_syllable1 = open(out_syllable, "w")
    for lines in str(ddd_syllable):
        out_syllable1.write(lines)
        # outf.write("\n")
    out_syllable1.close()

    fss = open('data/temp_word.lab', "r")
    lines2 = fss.readlines()
    starttimes2 = []
    endtimes2 = []
    word = []
    for x2 in lines2:
        starttimes2.append(x2.split()[0])
        endtimes2.append(x2.split()[1])
        word.append(x2.split()[2])
    fss.close()
    # print(starttimes2)
    floats2, floats2_e = [eval(x1) for x1 in starttimes2], [eval(x1) for x1 in endtimes2]
    # print(floats2)
    seg2 = []
    for s2 in floats2:
        seg2.append(s2 / 10000000)
    # print(seg2)
    seg2_e = []
    for s2 in floats2_e:
        seg2_e.append(s2 / 10000000)
    d_word = []
    for n2, v2, p2 in zip(seg2, seg2_e, word):
        d_word.append("{} {} {}".format(n2, v2, p2))
    w_table = open("data/word_t.txt", "w")
    for lines in d_word:
        w_table.write(lines)
        w_table.write("\n")
    w_table.close()
    filename_word = "data/word_t.txt"
    ddd_word = pd.read_csv(filename_word, header=None, sep="\s{1,}")
    ddd_word.columns = ['Starttime', 'Endtime', 'Word']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # df1.to_csv(filename)
    print(ddd_word)
    out_word = file + ".wrd"
    out_word1 = open(out_word, "w")
    for lines in str(ddd_word):
        out_word1.write(lines)
        # outf.write("\n")
    out_word1.close()

    ###############################################################################
    ###############################################################################
    ######################RELATIVE INTENSITY INDEX#################################
    def stride_trick(a, stride_length, stride_step):

        """

        apply framing using the stride trick from numpy.


        Args:

            a (array) : signal array.

            stride_length (int) : length of the stride.

            stride_step (int) : stride step.


        Returns:

            blocked/framed array.

        """

        nrows = ((a.size - stride_length) // stride_step) + 1

        n = a.strides[0]

        return np.lib.stride_tricks.as_strided(a,

                                               shape=(nrows, stride_length),

                                               strides=(stride_step * n, n))

    def relativ_intensity(signal, fs):
        win_len = 0.020
        win_step = 0.020
        if win_len < win_step:
            print("ParameterError: win_len must be larger than win_hop.")
        frame_length = win_len * fs
        frame_step = win_step * fs
        frames_overlap = frame_length - frame_step

        # compute number of frames and left sample in order to pad if needed to make

        # sure all frames have equal number of samples  without truncating any samples

        # from the original signal

        rest_samples = np.abs(N - frames_overlap) % np.abs(frame_length - frames_overlap)

        pad_signal = np.append(snd, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

        # apply stride trick

        frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
        e = np.sum(np.abs(np.fft.rfft(a=frames, n=len(frames))) ** 2, axis=-1) / len(frames) ** 2
        log_energy = e / N  # find the Intensity
        energy = scipy.signal.medfilt(log_energy, 5)
        energy = np.repeat(energy, frame_length)
        return energy

    ###############################################################################
    ###############################################################################
    ######################BREAK INDICES#################################
    def vad_s(signal, fs):
        wlen = 320
        inc = 160

        N = len(signal)
        if N <= wlen:
            nf = 1  # If the signal length is less than the length of one frame , Then the number of frames is defined as 1

        else:
            nf = int(np.ceil((1.0 * N - wlen + inc) / inc))  # otherwise , Calculate the total length of the frame

        pad_length = int((nf - 1) * inc + wlen)  # The total flattened length of all frames added up
        zeros = np.zeros(
            (pad_length - N,))  # Not enough length to use 0 fill , Be similar to FFT Extended array operation in
        pad_signal = np.concatenate((signal, zeros))  # The filled signal is recorded as pad_signal
        indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (
        wlen, 1)).T  # It is equivalent to extracting the time points of all frames , obtain nf*nw Matrix of length
        # print(indices[:2])
        indices = np.array(indices, dtype=np.int32)  # take indices Convert to matrix
        frames = pad_signal[indices]  # Get the frame signal
        windown = np.hanning(wlen)
        time2 = np.arange(0, nf) * (inc * 1.0 / fs)
        ste = np.zeros(nf)
        sf=[]
        for i in range(0, nf):
            a = frames[i:i + 1]
            b = a[0] * windown
            spectrum = np.fft.fft(b)

        # Magnitude spectrum (ignoring negative frequencies)
            magnitude_spectrum = np.abs(spectrum)

        # Compute geometric and arithmetic means
            geometric_mean =np.exp((2/N) * np.mean(10 * np.log10(magnitude_spectrum/N)))
            arithmetic_mean = (2/N) * np.mean(magnitude_spectrum)

        # Calculate spectral flatness
            sf1= geometric_mean / arithmetic_mean
            sf.append(sf1)
        print(sf)
        threshold = 25 * statistics.median(sf) if sf else 0
        vad = [i < threshold for i in sf]


        # print(vad)
        return vad, time2

    ###############################################################################
    ###############################################################################
    ############################ MAIN #############################################
    # signal = signal/max(abs(signal));
    # N1 = len(signal)         # Normalize the Signal
    # time = np.arange(0, N1) / fs
    fs, signal = wav.read("temp.wav")
    snd = parselmouth.Sound("temp.wav")
    N = len(snd)
    fs = snd.sampling_frequency
    # print(fs)
    signal = signal / max(abs(signal));
    t = np.arange(0, len(signal)) / fs
    # print(fs)
    energy = relativ_intensity(signal, fs)
    pitchEn = snd.to_pitch()
    pitch_values = pitchEn.selected_array['frequency']
    time_points = pitchEn.xs()
    f0 = pitch_values[pitch_values != 0]
    average_pitch = sum(f0) / len(f0)
    print(average_pitch)
    vad, time2 = vad_s(signal, fs)
    er = energy / max(abs(energy))  #### Relative Intensity Index
    t1 = [i / fs for i in range(0, len(er))]
    # splitted = [list(j) for i, j in groupby(er)]
    # print(len(splitted))
    #         d_intensity=[]
    # 	# k=[]
    #         I_seg1=[]
    #         for s1 in floats1:
    #             I_seg1.append((s1/10000000)*fs)
    #         floatss_I = [int(x) for x in I_seg1]
    #         for i in range(0,len(floatss_I)-1):
    #             s1 = er[floatss_I[i]:floatss_I[i+1]]
    #     		# print(syllable)
    #             splitted = [list(j) for i, j in groupby(s1)]
    #     		# print(splitted)
    #             k = []
    #             for idx,row in enumerate(splitted):
    #                 if all(n==n for n in row):
    #                     if row[0] >= 0 and row[0] < 0.1:
    #                 	# print(row[0],"1")
    #                	 # plt.axhline(y=row[0],c="g")
    #                         k.append(row[0])

    #                     elif row[0] >= 0.1 and row[0] < 0.4:

    #                         k.append(row[0])

    #                     elif row[0] >=0.4 and row[0] < 0.7:
    #                 # print(row[0],"3")
    #                 # plt.axhline(y=row[0],c="y")
    #                     	k.append(row[0])

    #                     elif row[0] >= 0.7:
    #                 # print(row[0],"4")
    #                 # plt.axhline(y=row[0],c="black")
    #                     	k.append(row[0])

    #             st = floatss_I[i]/fs
    #             et   = floatss_I[i+1]/fs
    #             min_energy = min(k)
    #             max_energy = max(k)
    #             Ave_energy = mean(k)
    #             d_intensity.append([st,et,min_energy,max_energy,Ave_energy])
    #         out_intensity=open("data/ff_intensity.txt", "w")
    #         for lines in str(np.array(d_intensity)):
    #             out_intensity.write(lines)
    #             # outf.write("\n")
    #         out_intensity.close()
    #         os.system("tr -d '\n' <data/ff_intensity.txt >data/ff1_intensity.txt")
    #         os.system("tr -s '[]' '\n' <data/ff1_intensity.txt >data/ff2_intensity.txt")
    #         os.system("awk 'NF' data/ff2_intensity.txt >data/ff3_intensity.txt")
    #         f_intensity=open('data/ff3_intensity.txt',"r")
    #         lines6=f_intensity.readlines()
    #         starttimes_intensity=[]
    #         endtimes_intensity=[]
    #         Min_Energy=[]
    #         Max_Energy=[]
    #         Average_Energy=[]
    #         for x6 in lines6:
    #             starttimes_intensity.append(x6.split()[0])
    #             endtimes_intensity.append(x6.split()[1])
    #             Min_Energy.append(x6.split()[2])
    #             Max_Energy.append(x6.split()[3])
    #             Average_Energy.append(x6.split()[4])
    #         f_intensity.close()
    #         # print(starttimes2)
    #         #tobi_I = [item.strip("'") for item in tobi]
    #         # print(breaks)
    #         floats_intensity,floats_e_intensity, mi_e, ma_e, av_e =[eval(x) for x in starttimes_intensity],[eval(x) for x in endtimes_intensity],[eval(x) for x in Min_Energy],[eval(x) for x in Max_Energy],[eval(x) for x in Average_Energy]
    #         d_rel_intensity=[]
    #         for n5,v5,p5,q5,r5,t5 in zip(floats_intensity,floats_e_intensity,syllable,mi_e,ma_e,av_e):
    #             d_rel_intensity.append("{} {} {} {} {} {}".format(n5,v5,p5,q5,r5,t5))
    #         # print(d_word)
    #         intensity_table=open("data/intensity_t.txt", "w")
    #         for lines in d_rel_intensity:
    #             intensity_table.write(lines)
    #             intensity_table.write("\n")
    #         intensity_table.close()
    #         filename_intensity = "data/intensity_t.txt"
    #         ddd_intensity = pd.read_csv(filename_intensity, header=None, sep="\s{1,}", engine='python')
    #         ddd_intensity.columns = ['Starttime' ,'Endtime' ,'Syllables' ,'Min_Energy' ,'Max_Energy' ,'Average_Energy']
    #         # df1.to_csv(filename)
    #         #print(df6)
    #         pd.set_option('display.max_columns',None)
    #         pd.set_option('display.max_rows',None)
    #         out_intensity_index = file + ".intensity"
    #         out_intensity_index1=open('../speech_new/static/' + out_intensity_index, "w")
    #         for lines in str(ddd_intensity):
    #             out_intensity_index1.write(lines)
    #     		# outf.write("\n")
    #         out_intensity_index1.close()

    ###############################################################################
    ###############################################################################
    ########################DISPLAY THE RESULTS####################################
    sns.set()  # Use seaborn's default style to make attractive graphs
    plt.rcParams['figure.dpi'] = 320  # Show images nicely
    plt.figure(figsize=(15, 10))
    plt.subplot(7, 1, 1)
    plt.plot(t, signal)
    frame1 = plt.gca()
    for xlabel_i in frame1.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    plt.ylabel("Amplitude", fontsize=8)
    # plt.xlabel("Samples/sec")
    plt.title("Time-Domain Signal", fontsize=10)
    plt.grid(False)
    plt.subplot(7, 1, 2)
    plt.plot(t1, er)
    frame2 = plt.gca()
    for xlabel_i in frame2.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    plt.ylabel("Amplitude", fontsize=8)
    # plt.xlabel("sec--->")
    plt.title("Relative Intensity Index", fontsize=10)
    d_intensity = []
    # k=[]
    I_seg1 = []
    for s1 in floats1:
        I_seg1.append((s1 / 10000000)*fs)
    I_seg1_e = []
    for s2 in floats1_e:
        I_seg1_e.append((s2 / 10000000)*fs)
    floatss_I = [int(x) for x in I_seg1]
    floatss_E = [int(x) for x in I_seg1_e]
    for i1, j1, k1 in zip_longest(floatss_I, floatss_E, syllable, fillvalue=None):
        print(i1)
    # for i in range(0, len(floatss_I) - 1):
        s1 = er[i1:j1]
        # print(syllable)
        splitted = [list(j) for i, j in groupby(s1)]
        # print(splitted)
        k = []
        for idx, row in enumerate(splitted):
            if all(n == n for n in row):
                if row[0] >= 0 and row[0] < 0.1:
                    # print(row[0],"1")
                    # plt.axhline(y=row[0],c="g")
                    k.append(row[0])


                elif row[0] >= 0.1 and row[0] < 0.4:

                    k.append(row[0])

                elif row[0] >= 0.4 and row[0] < 0.7:
                    # print(row[0],"3")
                    # plt.axhline(y=row[0],c="y")
                    k.append(row[0])

                elif row[0] >= 0.7:
                    # print(row[0],"4")
                    # plt.axhline(y=row[0],c="black")
                    k.append(row[0])

        st = i1 / fs
        et = j1/ fs
        min_energy = min(k)
        max_energy = max(k)
        Ave_energy = mean(k)
        if max_energy >= 0.9:
            x = i1/ fs
            y = max_energy
            plt.text(x, y, '5', {'color': 'r', 'fontsize': 15})
        elif max_energy < 0.9 and max_energy >= 0.55:
            x = i1/ fs
            y = max_energy
            plt.text(x, y, '4', {'color': 'r', 'fontsize': 15})
        elif max_energy < 0.55 and max_energy >= 0.4:
            x = i1/ fs
            y = max_energy
            plt.text(x, y, '3', {'color': 'r', 'fontsize': 15})
        elif max_energy < 0.4 and max_energy >= 0.2:
            x = i1 / fs
            y = max_energy
            plt.text(x, y, '2', {'color': 'r', 'fontsize': 15})
        else:
            x = i1 / fs
            y = max_energy
            plt.text(x, y, '1', {'color': 'r', 'fontsize': 15})
        d_intensity.append([st, et, min_energy, max_energy, Ave_energy])
    out_intensity = open("data/ff_intensity.txt", "w")
    for lines in str(np.array(d_intensity)):
        out_intensity.write(lines)
        # outf.write("\n")
    out_intensity.close()
    os.system("tr -d '\n' <data/ff_intensity.txt >data/ff1_intensity.txt")
    os.system("tr -s '[]' '\n' <data/ff1_intensity.txt >data/ff2_intensity.txt")
    os.system("awk 'NF' data/ff2_intensity.txt >data/ff3_intensity.txt")
    f_intensity = open('data/ff3_intensity.txt', "r")
    lines6 = f_intensity.readlines()
    starttimes_intensity = []
    endtimes_intensity = []
    Min_Energy = []
    Max_Energy = []
    Average_Energy = []
    for x6 in lines6:
        starttimes_intensity.append(x6.split()[0])
        endtimes_intensity.append(x6.split()[1])
        Min_Energy.append(x6.split()[2])
        Max_Energy.append(x6.split()[3])
        Average_Energy.append(x6.split()[4])
    f_intensity.close()
    # print(starttimes2)
    # tobi_I = [item.strip("'") for item in tobi]
    # print(breaks)
    floats_intensity, floats_e_intensity, mi_e, ma_e, av_e = [eval(x) for x in starttimes_intensity], [eval(x) for x in
                                                                                                       endtimes_intensity], [
                                                                 eval(x) for x in Min_Energy], [eval(x) for x in
                                                                                                Max_Energy], [eval(x)
                                                                                                              for x in
                                                                                                              Average_Energy]
    d_rel_intensity = []
    for n5, v5, p5, q5, r5, t5 in zip(floats_intensity, floats_e_intensity, syllable, mi_e, ma_e, av_e):
        d_rel_intensity.append("{} {} {} {} {} {}".format(n5, v5, p5, q5, r5, t5))
    # print(d_word)
    intensity_table = open("data/intensity_t.txt", "w")
    for lines in d_rel_intensity:
        intensity_table.write(lines)
        intensity_table.write("\n")
    intensity_table.close()
    filename_intensity = "data/intensity_t.txt"
    ddd_intensity = pd.read_csv(filename_intensity, header=None, sep="\s{1,}", engine='python')
    ddd_intensity.columns = ['Starttime', 'Endtime', 'Syllables', 'Min_Energy', 'Max_Energy', 'Average_Energy']
    # df1.to_csv(filename)
    print(ddd_intensity)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    out_intensity_index = file + ".intensity"
    out_intensity_index1 = open(out_intensity_index, "w")
    for lines in str(ddd_intensity):
        out_intensity_index1.write(lines)
    # outf.write("\n")
    out_intensity_index1.close()

    plt.grid(False)
    plt.subplot(7, 1, 3)
    plt.plot(snd.xs(), snd.values.T)
    frame3 = plt.gca()
    for xlabel_i in frame3.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    plt.ylabel("Amplitude", fontsize=8)
    # plt.xlabel("Samples/sec")
    plt.title("Phoneme Segmentation", fontsize=10)
    for s_l in seg:
        plt.axvline(x=s_l, color='black')
    plt.subplot(7, 1, 4)
    plt.plot(snd.xs(), snd.values.T)
    frame3 = plt.gca()
    for xlabel_i in frame3.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    plt.ylabel("Amplitude", fontsize=8)
    # plt.xlabel("Samples/sec")
    plt.title("Syllable Segmentation", fontsize=10)
    for s_l1 in seg1:
        plt.axvline(x=s_l1, color='black')
    plt.subplot(7, 1, 7)
    plt.plot(snd.xs(), snd.values.T)
    plt.ylabel("Amplitude", fontsize=8)
    plt.xlabel("sec --->")
    plt.title("Word Segmentation", fontsize=10)
    for s_l2 in seg2:
        plt.axvline(x=s_l2, color='black')
    plt.subplot(7, 1, 6)
    plt.plot(t, signal)
    plt.plot(time2, vad, c="black")
    frame4 = plt.gca()
    for xlabel_i in frame4.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    plt.ylabel("Amplitude", fontsize=8)
    # plt.xlabel("Samples/sec")
    plt.title("Break Indices", fontsize=9)
    splitted = [list(g) for i, g in itertools.groupby(vad, lambda x: x == True)]
    # print(splitted)
    # indices = np.where(vad == False)
    # print(indices)
    indices = [i for i, x in enumerate(vad) if not x]

    gg = [list(group) for group in mit.consecutive_groups(indices)]
    d_break = []
    for idx, row in enumerate(splitted):
        if all(n == False for n in row):
            # print(len(row))
            if len(row) >= 10:
                # print("LSIL")
                for idx1, row1 in enumerate(gg):
                    if len(row) == len(row1):
                        t = row1[0]
                        # print(t)
                        x = time2[t]
                        # print(x)
                        st = x
                        t1 = row1[len(row1) - 1]
                        et = time2[t1]
                        y = 0.25
                        plt.text(x, y, '3', {'color': 'r', 'fontsize': 15})
                        d_break.append([st, et, "LSIL"])
            elif len(row) >= 3:
                # print("MSIL")
                for idx1, row1 in enumerate(gg):
                    if len(row) == len(row1):
                        t = row1[0]
                        # print(t)
                        x = time2[t]
                        # print(x)
                        st = x
                        t1 = row1[len(row1) - 1]
                        et = time2[t1]
                        y = 0.25
                        plt.text(x, y, '2', {'color': 'r', 'fontsize': 15})
                        d_break.append([st, et, "MSIL"])
            elif len(row) < 3:
                # print("SSIL")
                for idx1, row1 in enumerate(gg):
                    if len(row) == len(row1):
                        t = row1[0]
                        # print(t)
                        x = time2[t]
                        # print(x)
                        st = x
                        t1 = row1[len(row1) - 1]
                        et = time2[t1]
                        y = 0.25
                        plt.text(x, y, '1', {'color': 'r', 'fontsize': 15})
                        d_break.append([st, et, "SSIL"])
    test_list = np.array(d_break)
    res_duplicate = list(set(map(lambda i: tuple(sorted(i)), test_list)))
    # print(res_duplicate)
    dd_break = sorted(res_duplicate, key=lambda x: x[1], reverse=False)
    dd_break1 = pd.DataFrame(dd_break)
    dd_break1.columns = ['Starttime', 'Endtime', 'Break_Indices']
    print(dd_break1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    out_break = file + ".b"
    out_break1=open(out_break, "w")
    for lines in str(dd_break1):
        out_break1.write(lines)
		# outf.write("\n")
    out_break1.close()
    plt.subplot(7, 1, 5)
    pitch_values = pitchEn.selected_array['frequency']
    # pitch_values[pitch_values==0] = np.nan
    # print(pitchEn)
    plt.plot(pitchEn.xs(), pitch_values, '.', markersize=2.5)
    plt.ylim(0, pitchEn.ceiling)
    frame3 = plt.gca()
    for xlabel_i in frame3.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    plt.ylabel("freq[Hz]", fontsize=8)
    # plt.xlabel("Samples/sec")
    plt.title("Pitch Contour", fontsize=10)
    plt.grid(True)
    d_tobi = []
    d1_tobi=[]
    p_seg1 = []
    for s1 in floats1:
        p_seg1.append((s1 / 10000000))
    # print(p_seg1)
    # floatss = [int(x) for x in p_seg1]
    # print(floatss)
    p_seg1_e = []
    for s1 in floats1_e:
        p_seg1_e.append((s1 / 10000000))
    # print(p_seg1_e)plt.subplot(7, 1, 5)
    

    # floatss_e = [int(x) for x in p_seg1_e]
    # print(floatss_e)
    dynamic_ranges1 = []
    dynamic_ranges2 = []
    k1 = 1
    for i, j, k in zip_longest(p_seg1, p_seg1_e, syllable, fillvalue=None):
        print(i, j, k)
        start_index = np.argmax(time_points >= i)
        end_index = np.argmax(time_points >= j)
        spliced_pitch_values = pitch_values[start_index:end_index]
        spliced_time_points = time_points[start_index:end_index]
        res = spliced_pitch_values[spliced_pitch_values != 0]
        snd1 = snd.extract_part(from_time=i, to_time=j)
        # print(res)
        # print(res)
        # plt.subplot(211),plt.plot(snd1.xs(),snd1.values.T)
        # plt.subplot(212),plt.plot(res,'.')
        # kr=KernelReg(pitch_values, len(pitch_values), 'c
        # ss = savgol_filter(x=res, window_length=5,  polyorder=3)
        # print(ss)
        # ss = pyasl.smooth(pitch_values,5,'hanning')
        # print(ss)
        # plt.subplot(313),plt.plot(ss)
        ff = []
        ff1 = []

        l = [x for x in res if ~np.isnan(x)]
        if l == []:
            dy_r1 = 0
        else:
            dy_r1 = max(l) - min(l)
            dynamic_ranges1.append(dy_r1)
        with open(file + 'dynamic1.txt', 'w') as doc1:
            for item in dynamic_ranges1:
                doc1.write(str(item) + '\n')

        # print(len(l))
        # if len(l) != 0:
        #     l1 = l
        # else:
        #     l1 = 0
        #     continue
        # # print(len(l))
        len_f = len(l)
        # print(len_f)
        if len_f > 1:
            xin = np.arange(0, len_f)
            yin = l
        elif len_f == 1 or len_f < 1:
            yin = 0
            xin = 0
            pf = 0
            continue
        # print(xin)
        # yin = [3,4,2,3,4,5,6,7,8,9,10,11,12,13,14]
        # xin = range(0,len(yin))
           
   
       # print(df6)
       
       # print(file)
       # print(xin)
       # yin = [3,4,2,3,4,5,6,7,8,9,10,11,12,13,14]
       # xin = range(0,len(yin))
     
        
        pf = np.polyfit(xin, yin, 3)
        # print(pf)
        yin_smooth = np.polyval(pf, xin)
        dy_r2 = max(yin_smooth) - min(yin_smooth)
        print(yin_smooth)
        # print(yin_smooth[0])
        dynamic_ranges2.append(dy_r2)
        with open(file + 'dynamic2.txt', 'w') as doc2:
            for item in dynamic_ranges2:
                doc2.write(str(item) + '\n')
        if dy_r2 < 10:
            # fig, axs = plt.subplots(3, 1)
            x1 = j 
            y1 = pitchEn.ceiling / 4
            st1 = i
            et1 = j
            print(st1)
            # axs[0].plot(snd1.xs(), snd1.values.T)
            # axs[0].set_title(syl_id[k1 - 1])
            # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
            # # Plot yin_smooth
            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
            # axs[2].legend(['flat'], loc='center', fontsize=20)
            d_tobi.append([st1, et1, 'flat'])
            plt.text(x1,y1,'flat',{'color':'r','fontsize':7})
            print("flat")
            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/flat'
            # os.makedirs(save_dir, exist_ok=True)
            # # plt.figure(k1)  # Activate the figure
            # # plt.tight_layout()
            # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
            # print(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
            # k1 = k1 + 1

        else:
            # print(yin_smooth)
            for ss in range(0, len(yin_smooth) - 1):
                ff.append(yin_smooth[ss + 1] - yin_smooth[ss])
                ff1 = abs(max(yin_smooth) - min(yin_smooth))
            # print(ff)
            # ff.sort(reverse=True)
            # # # print(ff)
            splitted = [list(g) for i, g in itertools.groupby(ff, lambda xs: xs < 0)]
            # print(splitted)
            # print(len(splitted))
            index = []
            len_f = []
            f_p_i = None
            l_p_i = None
            f_n_i = None
            l_n_i = None
            for s, value in enumerate(ff):
                if value > 0:
                    if f_p_i is None:
                        f_p_i = s
                    l_p_i = s
                else:
                    if f_n_i is None:
                        f_n_i = s
                    l_n_i = s
            #print(yin_smooth[f_p_i])
            #print(yin_smooth[f_n_i])
            #print(yin_smooth[(f_n_i) + 1])
            #print(yin_smooth[(l_n_i) + 1])

            for idx, row in enumerate(splitted):

                if all(n > 0 for n in row):
                    if idx == 0:
                        # p1 = len(row)
                        index.append('p1')
                        len_f.append(len(row))
                        # print(row[0])
                        # print(row[-1])
                        # pit_f.append(abs(row[0]-row[-1]))
                        # print(pit_f)
                    elif idx == 1:
                        # p2= len(row)
                        index.append("p2")
                        len_f.append(len(row))
                    elif idx == 2:
                        # p3=len(row)
                        index.append("p3")
                        len_f.append(len(row))
                        # print(p3)
                else:
                    if idx == 0:
                        # q1=len(row)
                        index.append("q1")
                        len_f.append(len(row))
                    elif idx == 1:
                        # q2=len(row)
                        index.append("q2")
                        len_f.append(len(row))
                    elif idx == 2:
                        # q3=len(row)
                        index.append("q3")
                        len_f.append(len(row))
            # print(index)
            # print(len_f)
            a = np.concatenate((index, len_f))
            print(a)
            # input()
            # print(a)
            l_a = len(a)
            # print(l_a)
            # x1 = floatss[i+1]/fs
            # print(x)
            # y = np.array()
            # print(y)
            # fig, axs = plt.subplots(3, 1)
            if l_a == 2:
                # Create a new figure and subplots outside the condition

                if a[0] == 'p1' and 10 <= ff1 < 60:
                    x1 = j  
                    y1 = pitchEn.ceiling / 4
                    st1 = i
                    et1 = j
                    t1 = np.arange(0, len(yin_smooth))
                    d_tobi.append([st1, et1, 'S_H'])
                    plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})

                    #         axs[0].plot(snd1.xs(), snd1.values.T)
                    #         #axs[0].set_title(syl_id[k1-1])
                    # # Plot pitch values
                    #         axs[1].plot(pitchE.xs(), pitch_values, '.', markersize=2.5)
                    # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    #
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # # Plot yin_smooth
                    # axs[2].plot(t1, yin_smooth)
                    # axs[2].legend(['S_H'], loc='center', fontsize=20)
                    
                    # Add text annotation
                    # axs[1].text(0.11, 200, 'H*', {'color': 'r', 'fontsize': 20}, ha='center', va='center')
                    # axs[1].legend(['H*'], loc='center', fontsize=20)
                    # print("S_H")
                    # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_H'
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'p1' and 60 <= ff1 < 100:
                    x1 = j 
                    y1 = pitchEn.ceiling / 4
                    st1 = i
                    et1 = j
                    t1 = np.arange(0, len(yin_smooth))

                    #         axs[0].plot(snd1.xs(), snd1.values.T)
                    #         #axs[0].set_title(syl_id[k1-1])
                    # # Plot pitch values
                    #         axs[1].plot(pitchE.xs(), pitch_values, '.', markersize=2.5)
                    # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    #
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    # axs[2].legend(['M_H'], loc='center', fontsize=20)
                    d_tobi.append([st1, et1, 'M_H'])
                    plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                    # Add text annotation
                    # axs[1].text(0.11, 200, 'H*', {'color': 'r', 'fontsize': 20}, ha='center', va='center')
                    # axs[1].legend(['H*'], loc='center', fontsize=20)
                    # print("M_H")
                    # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_H'
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'p1' and ff1 >= 100:
                    x1 = j
                    y1 = pitchEn.ceiling / 4
                    st1 = i
                    et1 = j
                    t1 = np.arange(0, len(yin_smooth))

                    #         axs[0].plot(snd1.xs(), snd1.values.T)
                    #         #axs[0].set_title(syl_id[k1-1])
                    # # Plot pitch values
                    #         axs[1].plot(pitchE.xs(), pitch_values, '.', markersize=2.5)
                    # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    #
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    # axs[2].legend(['B_H'], loc='center', fontsize=20)
                    d_tobi.append([st1, et1, 'B_H'])
                    plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                    # Add text annotation
                    # axs[1].text(0.11, 200, 'H*', {'color': 'r', 'fontsize': 20}, ha='center', va='center')
                    # axs[1].legend(['H*'], loc='center', fontsize=20)
                    # print("B_H")
                    # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_H'
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1

                    # plt.savefig('plot1.png')
                elif a[0] == 'q1' and 10 <= ff1 < 60:
                    x1 = j 
                    y1 = pitchEn.ceiling / 4
                    st1 = i
                    et1 = j

                    #         axs[0].plot(snd1.xs(), snd1.values.T)
                    #         #axs[0].set_title(syl_id[k1-1])
                    # # Plot pitch values
                    #         axs[1].plot(pitchE.xs(), pitch_values, '.', markersize=2.5)
                    # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    # Add text annotation
                    # axs[1].text(0.11, 200, 'L*', {'color': 'r', 'fontsize': 20}, ha='center', va='center')
                    #
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    # axs[2].legend(['S_L'], loc='center', fontsize=20)
                    d_tobi.append([st1, et1, 'S_L'])
                    plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                    # print("S_L")
                    # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_L'
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                    # plt.savefig('plot2.png')
                    # d_tobi.append([st1,et1,'L*'])
                elif a[0] == 'q1' and 60 <= ff1 < 100:
                    x1 = j 
                    y1 = pitchEn.ceiling / 4
                    st1 = i
                    et1 = j
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    # axs[2].legend(['M_L'], loc='center', fontsize=20)
                    d_tobi.append([st1, et1, 'M_L'])
                    plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                    # print("M_L")
                    # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_L'
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'q1' and ff1 >= 100:
                    x1 = j
                    y1 = pitchEn.ceiling / 4
                    st1 = i
                    et1 = j
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # # Plot yin_smooth
                    # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                    # axs[2].legend(['B_L'], loc='center', fontsize=20)
                    d_tobi.append([st1, et1, 'B_L'])
                    plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                    # print("B_L")
                    # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_L'
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1

            elif l_a == 4:
                if a[0] == 'p1' and a[1] == 'q2':
                    if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['S_L'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'S_L'])
                            plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                            # print("S_HLL")
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_L'

                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_hat'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_hat'])
                                plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_hat'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_HLL'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_HLL'])
                                plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HLL'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['M_L'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'M_L'])
                            plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_L'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_hat'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_hat'])
                                plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_hat'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_HLL'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_HLL'])
                                plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HLL'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) < int(a[3]) and ff1 >= 100:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['B_L'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'B_L'])
                            plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_L'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_hat'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_hat'])
                                plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_hat'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_HLL'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_HLL'])
                                plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HLL'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        k1 = k1 + 1

                    elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                        if int(a[3]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['S_H'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'S_H'])
                            plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_H'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_hat'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_hat'])
                                plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_hat'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_HHL'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_HHL'])
                                plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HHL'

                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['M_H'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'M_H'])
                            plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_H'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_hat'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_hat'])
                                plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_hat'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_HHL'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_HHL'])
                                plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HHL'
                       
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) > int(a[3]) and ff1 >= 100:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['B_H'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'B_H'])
                            plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_H'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_hat'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_hat'])
                                plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_hat'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_HHL'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_HHL'])
                                plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HHL'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) == int(a[3]):
                        if 10 <= ff1 < 60:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['S_hat'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'S_hat'])
                            plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_hat'
                        elif 60 <= ff1 < 100:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['M_hat'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'M_hat'])
                            plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_hat'
                        elif ff1 >= 100:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['B_hat'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'B_hat'])
                            plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_hat'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                elif a[0] == 'q1' and a[1] == 'p2':
                    if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['S_H'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'S_H'])
                            plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_H'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_bucket'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_bucket'])
                                plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_bucket'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_LHH'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_LHH'])
                                plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LHH'

                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['M_H'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'M_H'])
                            plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_H'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_bucket'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_bucket'])
                                plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_bucket'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_LHH'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_LHH'])
                                plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LHH'
                       
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) < int(a[3]) and ff1 >= 100:
                        if int(a[2]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend('B_H', loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'B_H'])
                            plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_H'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_bucket'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_bucket'])
                                plt.text(x1,y1,'B_bucket',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_bucket'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend('B_LHH', loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_LHH'])
                                plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LHH'
                       
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                        if int(a[3]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['S_L'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'S_L'])
                            plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_L'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_bucket'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_bucket'])
                                plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_bucket'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['S_LLH'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'S_LLH'])
                                plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LLH'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                        if int(a[3]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['M_L'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'M_L'])
                            plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_L'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_bucket'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_bucket'])
                                plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_bucket'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['M_LLH'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'M_LLH'])
                                plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LLH'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1
                    elif int(a[2]) > int(a[3]) and ff1 > 100:
                        if int(a[3]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['B_L'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'B_L'])
                            plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_L'
                        else:
                            thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                            thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                            print(thresh_p)
                            print(thresh_n)
                            if thresh_p == thresh_n:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_bucket'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_bucket'])
                                plt.text(x1,y1,'B_bucket',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_bucket'
                            else:
                                st1 = i
                                et1 = j
                                x1 = j 
                                y1 = pitchEn.ceiling / 4
                                # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                                # axs[2].legend(['B_LLH'], loc='center', fontsize=20)
                                d_tobi.append([st1, et1, 'B_LLH'])
                                plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                                # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LLH'

                       
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1

                    elif int(a[2]) == int(a[3]):
                        if 10 <= ff1 < 60:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['S_bucket'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'S_bucket'])
                            plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                            print("bucket")
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_bucket'
                        elif 60 <= ff1 < 100:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['M_bucket'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'M_bucket'])
                            plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                            print("bucket")
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_bucket'
                        elif ff1 >= 100:
                            st1 = i
                            et1 = j
                            x1 = j 
                            y1 = pitchEn.ceiling / 4
                            # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                            # axs[2].legend(['B_bucket'], loc='center', fontsize=20)
                            d_tobi.append([st1, et1, 'B_bucket'])
                            plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                            print("bucket")
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_bucket'
                        
                        # axs[0].plot(snd1.xs(), snd1.values.T)
                        # axs[0].set_title(syl_id[k1 - 1])
                        # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                        # os.makedirs(save_dir, exist_ok=True)
                        # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                        # k1 = k1 + 1

            elif l_a == 6:
                if a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 10 <= ff1 < 60:
                    if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_LHH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_LHH'])
                        plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LHH'
                    elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_LLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_LLH'])
                        plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LLH'
                    elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_HLL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_HLL'])
                        plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HLL'
                    elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_HHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_HHL'])
                        plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HHL'
                    elif int(a[3]) <= 2 and int(a[5]) <= 2:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_L'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_L'])
                        plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_L'
                    else:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_HLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_HLH'])
                        plt.text(x1,y1,'M_HLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HLH'

                    
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 60 <= ff1 < 100:
                    if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_LHH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_LHH'])
                        plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LHH'
                    elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_LLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_LLH'])
                        plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LLH'
                    elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_HLL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_HLL'])
                        plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HLL'
                    elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_HHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_HHL'])
                        plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HHL'
                    elif int(a[3]) <= 2 and int(a[5]) <= 2:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_L'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_L'])
                        plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_L'
                    else:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_HLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_HLH'])
                        plt.text(x1,y1,'M_HLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HLH'

                   
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and ff1 >= 100:
                    if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_LHH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_LHH'])
                        plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LHH'
                    elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_LLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_LLH'])
                        plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LLH'
                    elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_HLL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_HLL'])
                        plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HLL'
                    elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_HHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_HHL'])
                        plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HHL'
                    elif int(a[3]) <= 2 and int(a[5]) <= 2:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_L'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_L'])
                        plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_L'
                    else:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_HLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_HLH'])
                        plt.text(x1,y1,'B_HLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HLH'

                    
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 10 <= ff1 < 60:
                    if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_HLL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_HLL'])
                        plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HLL'
                    elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_HHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_HHL'])
                        plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_HHL'
                    elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_LHH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_LHH'])
                        plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LHH'
                    elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_LLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_LLH'])
                        plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LLH'
                    elif int(a[3]) <= 2 and int(a[5]) <= 2:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_H'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_H'])
                        plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_H'
                    else:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['S_LHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'S_LHL'])
                        plt.text(x1,y1,'S_LHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/S_LHL'

                    
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 60 <= ff1 < 100:
                    if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_HLL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_HLL'])
                        plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HLL'
                    elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_HHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_HHL'])
                        plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_HHL'
                    elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_LHH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_LHH'])
                        plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LHH'
                    elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_LLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_LLH'])
                        plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LLH'
                    elif int(a[3]) <= 2 and int(a[5]) <= 2:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_H'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_H'])
                        plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_H'
                    else:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['M_LHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'M_LHL'])
                        plt.text(x1,y1,'M_LHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/M_LHL'
                    
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1
                elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and ff1 >= 100:
                    if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_HLL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_HLL'])
                        plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HLL'
                    elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_HHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_HHL'])
                        plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HHL'
                    elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_LHH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_LHH'])
                        plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LHH'
                    elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_LLH'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_LLH'])
                        plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LLH'
                    elif int(a[3]) <= 2 and int(a[5]) <= 2:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_H'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_H'])
                        plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_H'
                    else:
                        st1 = i
                        et1 = j
                        x1 = j 
                        y1 = pitchEn.ceiling / 4
                        # axs[2].plot([i for i in range(len(yin_smooth))], yin_smooth)
                        # axs[2].legend(['B_LHL'], loc='center', fontsize=20)
                        d_tobi.append([st1, et1, 'B_LHL'])
                        plt.text(x1,y1,'B_LHL',{'color':'r','fontsize':7})
                        # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_LHL'
                    
                    # axs[0].plot(snd1.xs(), snd1.values.T)
                    # axs[0].set_title(syl_id[k1 - 1])
                    # axs[1].plot(spliced_time_points, spliced_pitch_values, '.', markersize=2.5)
                    # os.makedirs(save_dir, exist_ok=True)
                    # plt.savefig(os.path.join(save_dir, f'{file_pit}_{k1}.png'))
                    # k1 = k1 + 1  # d_tobi.append([st1,et1,'LH%'])
                 #d1_tobi.append(d_tobi)
                # print(d1_tobi)

    # axs[0].set_title('Pitch Values')
    # axs[1].set_title('Yin Smooth')
    print(d_tobi)
    dd_tobi = pd.DataFrame(d_tobi)
    dd_tobi.columns = ['Starttime', 'Endtime', 'Labels']
    print(dd_tobi)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    out_pitch = file + ".pitch"
    out_pitch1 = open(out_pitch, "w")
    for lines in str(dd_tobi):
        out_pitch1.write(lines)
    # outf.write("\n")
    out_pitch1.close()
    plt.show()
    os.system("rm *.wav")


# json_data = {
#     "phenome":ddd_phoneme,
#     "syllable":out_syllable,
#     "word":out_word,
#     "intensity_index":out_intensity_index,
#     "break":out_break,
#     "pitch":out_pitch}

# return jsonify(json_data)
token = "1e0a93be5b86ed90c160f0a54a506b5a58ac3f87f4f744010b8f0d2cf838eae5"
# filename = sys.argv[1]
filename = '/media/speechlab/Expansion/hindi_fem_wav/text_01007.wav'
# filename = '/media/speechlab/Expansion/hindi_male_mono/wav/male_01009.wav'
language = "Hindi"
full_code(filename, language)
