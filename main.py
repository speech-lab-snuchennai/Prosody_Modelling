import itertools
import os
import statistics
import time
from itertools import groupby, zip_longest
from statistics import mean

import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import pandas as pd
import parselmouth
import scipy
import scipy.io.wavfile as wav
import seaborn as sns
from flask import *
from flask import Flask, render_template, request
from g2p_en import G2p

# Intialize the app object
app = Flask(__name__, template_folder="./web/templates", static_folder="./web/static")

def run_lang_all(language, text_file, wave_file):
    lang = language.lower()
    with open(text_file.filename, 'r') as f_t:
        text_content = f_t.read()
    with open('te.txt', 'w') as output_file:
        output_file.write(text_content)
    os.system("tr -d '[:punct:]' <te.txt >temp.txt")
    os.system("rm -r data")
    os.system("mkdir data")
    os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
    os.system("rm ../speech_new/web/static/*.jpg")
    os.system("rm ../speech_new/web/static/*.phn")
    os.system("rm ../speech_new/web/static/*.syl")
    os.system("rm ../speech_new/web/static/*.wrd")
    os.system("rm ../speech_new/web/static/*.intensity")
    os.system("rm ../speech_new/web/static/*.b")
    os.system("rm ../speech_new/web/static/*.pitch")
    os.system("python3 unified_python_parser/Unified_Parser_smt_lab_IITM/word.py")
    os.system("tr -d [[:punct:]] <Perl/wordpronunciation >data/t_wrd.txt")
    os.system("sed -i 's/0//g' data/t_wrd.txt")
    os.system("sed -i 's/ //g' data/t_wrd.txt")
    os.system("tr '\n' ' ' <data/t_wrd.txt >data/tt_wrd.txt")
    os.system("cp -r data/tt_wrd.txt data/temp.txt")
    os.system("sed -i 's/ /SIL/g' data/tt_wrd.txt")
    os.system(f"./scripts/ortho_to_phonetic1_phoneme data/tt_wrd.txt dep_list/phonelist_{lang} >>data/te")
    os.system("awk '{ for (i=1; i<=NF; i++) if ($i == \"e\") $i = \"ee\" } 1' data/te >data/tem")
    os.system("./scripts/startsil_HIN data/tem")
    os.system("ls -v data/*.wav >data/wavlist")
    os.system("ls -v data/*.lab >data/lablist")
    os.system("./scripts/create_maptable data/wavlist")
    os.system("HCopy -C config_files/speech_config_file -S data/maptable")
    os.system("cat data/*.lab | sort | uniq >data/wordlist")
    os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
    os.system("./scripts/mlf data/lablist >data/mlf")
    os.system(f"HVite -a -o SW -I data/mlf -H dep_models/all_hmm_{lang} dep_list/dict_{lang} data/wordlist data/temp.mfc")
    os.system("./scripts/phoneme data/temp.rec")
    os.system("tr -d '\n' <data/temp.lab >data/temp")
    os.system(f"./scripts/ortho_to_phonetic_cv_{lang} data/temp dep_list/con_c_v_cv_{lang}_list >data/t.led")
    os.system("cp data/temp.rec data/temp_syl.lab")
    os.system("HLEd data/t.led data/temp_syl.lab")
    os.system(f"./scripts/ortho_to_phonetic_syllable data/temp dep_list/con_c_v_cv_{lang}_list >data/tempp")
    os.system(f"./scripts/syllable_from_char data/tempp dep_list/v_{lang}_list dep_list/c_{lang}_list dep_list/cv_{lang}_list >data/temp.led")
    os.system("HLEd data/temp.led data/temp_syl.lab")
    os.system("./scripts/syllable data/temp_syl.lab")
    os.system("./scripts/word_segment_modified data/temp.rec data/temp.txt")

# region Routes that handle rendering of templates
@app.route("/")
def main():
    return render_template("index.html")


@app.route("/demos.html")
def demo():

    return render_template("demos.html")


@app.route("/complete_demo.html")
def complete():
    # logging.debug("succesfully logged")
    return render_template("complete_demo.html")


@app.route("/phoneme.html")
def phoneme():
    return render_template("phoneme.html")


@app.route("/syllable.html")
def syllable():
    return render_template("syllable.html")


@app.route("/word.html")
def word():
    return render_template("word.html")


@app.route("/Tobi_version_I.html")
def Tobi_version_II():
    return render_template("Tobi_version_I.html")


@app.route("/Relative_Intensity.html")
def Relative_Intensity():
    return render_template("Relative_Intensity.html")


@app.route("/Break_Indices.html")
def Break_Indices():
    return render_template("Break_Indices.html")


@app.route("/contact.html")
def contact():
    return render_template("contact.html")


@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/projects.html")
def proj():
    return render_template("projects.html")


@app.route("/activities.html")
def act():
    return render_template("activities.html")


@app.route("/members.html")
def member():
    return render_template("members.html")


@app.route("/publications.html")
def pub():
    return render_template("publications.html")


@app.route('/Relative_intensity',methods = ['GET','POST'])
def Relative_Intesnity_Index():
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        text_file.save(text_file.filename)
        wave_file.save(wave_file.filename)

        f1 =wave_file.filename.split('\\')
        file = f1[-1].split('.')[0]
        os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
        os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
        val=request.form.get('languages')
        language= str(val)
        print(language)

        if language.lower() == 'english':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_eng data/temp dep_list/con_c_v_cv_list_eng >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_eng data/temp dep_list/con_c_v_cv_list_eng >data/tempp")
            os.system("./scripts/syllable_from_char_eng data/tempp dep_list/v_list_eng dep_list/c_list_eng dep_list/CV_list_eng >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")

        elif language.lower() =='tamil':
            with open(text_file.filename, 'r') as f_t:
                text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
                output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >t.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/sil/ /g' temp.txt")
            os.system("sed -i 's/sil/SIL/g' lists/out_word")
            os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
            os.system("./scripts/startsil p")
            os.system("mv temp.txt data")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_tam data/temp dep_list/con_c_v_cv_list_tam >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_tam data/temp dep_list/con_c_v_cv_list_tam >data/tempp")
            os.system("./scripts/syllable_from_char_tam data/tempp dep_list/v_list_tam dep_list/c_list_tam dep_list/CV_list_tam >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")

        else:
            run_lang_all(language,text_file,wave_file)
            # run_lang_relative(language,text_file,wave_file)


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
        floats1, floats1_e = [eval(x1) for x1 in starttimes1], [eval(x1) for x1 in endtimes1]
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

        ######################RELATIVE INTENSITY INDEX#################################
        def stride_trick(a, stride_length, stride_step):
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
            rest_samples = np.abs(N - frames_overlap) % np.abs(frame_length - frames_overlap)
            pad_signal = np.append(snd, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))
            frames = stride_trick(pad_signal, int(frame_length), int(frame_step))               # apply stride trick
            e = np.sum(np.abs(np.fft.rfft(a=frames, n=len(frames))) ** 2, axis=-1) / len(frames) ** 2
            log_energy = e / N  # find the Intensity
            energy = scipy.signal.medfilt(log_energy, 5)
            energy = np.repeat(energy, frame_length)
            return energy
        ############################ MAIN #############################################
        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        energy = relativ_intensity(signal, fs)
        pitchEn = snd.to_pitch()
        pitch_values = pitchEn.selected_array['frequency']
        time_points = pitchEn.xs()
        f0 = pitch_values[pitch_values != 0]
        average_pitch = sum(f0) / len(f0)
        print(average_pitch)
        # vad, time2 = vad_s(signal, fs)
        er = energy / max(abs(energy))  #### Relative Intensity Index
        t1 = [i / fs for i in range(0, len(er))]
        ########################DISPLAY THE RESULTS####################################
        sns.set() # Use seaborn's default style to make attractive graphs
        plt.rcParams['figure.dpi'] = 320 # Show images nicely
        plt.figure(figsize=(10,5))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude",fontsize=8)
        plt.title("Time-Domain Signal",fontsize=10)
        for s_l1 in seg1:
            plt.axvline(x=s_l1, color='black')
        plt.grid(False)
        plt.subplot(2, 1, 2)
        plt.plot(t1, er)
        frame2 = plt.gca()
        for xlabel_i in frame2.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
        plt.title("Relative Intensity Index", fontsize=10)
        d_intensity = []
        I_seg1 = []
        I_seg1_e = []
        for s1 in floats1:
            I_seg1.append((s1 / 10000000) * fs)
        for s2 in floats1_e:
            I_seg1_e.append((s2 / 10000000)*fs)
        floatss_I = [int(x) for x in I_seg1]
        floatss_E = [int(x) for x in I_seg1_e]
        for i1, j1, k1 in zip_longest(floatss_I, floatss_E, syllable, fillvalue=None):
            s1 = er[i1:j1]
            splitted = [list(j) for i, j in groupby(s1)]
            k = []
            for idx, row in enumerate(splitted):
                if all(n == n for n in row):
                    if row[0] >= 0 and row[0] < 0.1:
                        k.append(row[0])
                    elif row[0] >= 0.1 and row[0] < 0.4:
                        k.append(row[0])
                    elif row[0] >= 0.4 and row[0] < 0.7:
                        k.append(row[0])
                    elif row[0] >= 0.7:
                        k.append(row[0])

            st = i1 / fs
            et = j1 / fs
            min_energy = min(k)
            max_energy = max(k)
            Ave_energy = mean(k)
            if max_energy >= 0.9:
                x = i1 / fs
                y = max_energy
                plt.text(x, y, '5', {'color': 'r', 'fontsize': 15})
            elif max_energy < 0.9 and max_energy >= 0.55:
                x = i1 / fs
                y = max_energy
                plt.text(x, y, '4', {'color': 'r', 'fontsize': 15})
            elif max_energy < 0.55 and max_energy >= 0.4:
                x = i1 / fs
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
        floats_intensity, floats_e_intensity, mi_e, ma_e, av_e = [eval(x) for x in starttimes_intensity], [eval(x) for x in
                                                                                                           endtimes_intensity], [
                                                                     eval(x) for x in Min_Energy], [eval(x) for x in
                                                                                                    Max_Energy], [eval(x)
                                                                                                                  for x in
                                                                                                                  Average_Energy]
        d_rel_intensity = []
        for n5, v5, p5, q5, r5, t5 in zip(floats_intensity, floats_e_intensity, syllable, mi_e, ma_e, av_e):
            d_rel_intensity.append("{} {} {} {} {} {}".format(n5, v5, p5, q5, r5, t5))
        intensity_table = open("data/intensity_t.txt", "w")
        for lines in d_rel_intensity:
            intensity_table.write(lines)
            intensity_table.write("\n")
        intensity_table.close()
        filename_intensity = "data/intensity_t.txt"
        ddd_intensity = pd.read_csv(filename_intensity, header=None, sep="\s{1,}", engine='python')
        ddd_intensity.columns = ['Starttime', 'Endtime', 'Syllables', 'Min_Energy', 'Max_Energy', 'Average_Energy']
        print(ddd_intensity)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        out_intensity_index = file + ".intensity"
        out_intensity_index1=open('../speech_new/web/static/' + out_intensity_index, "w")
        for lines in str(ddd_intensity):
            out_intensity_index1.write(lines)
        out_intensity_index1.close()
        plt.grid(False)
        new_graph_name = "graph" +str(time.time()) +".jpg"
        for filename in os.listdir('../speech_new/web/static'):
            if filename.startswith('graph_'):
                os.remove('../speech_new/web/static/'+filename)
        plt.savefig('../speech_new/web/static/' + new_graph_name)
        plt.show()
        os.system("rm -f *.wav *.txt *.p *.phn p phn")


    return render_template('table.html',name=file, data=ddd_intensity, graph=new_graph_name, file1=out_intensity_index)


@app.route("/Break_Indices", methods=["GET", "POST"])
def Break_indices():
    if request.method == "POST":
        f = request.files["file"]
        f.save(f.filename)
        f1 = f.filename.split("\\")
        file = f1[-1].split(".")[0]
        file_pit = f.filename.split("/")[-1].split(".")[0]
        print(f.filename)
        os.system("ch_wave " + f.filename + " -F 16000 -otype wav -o temp.wav")
        ##########################Speech to Text#######################################
        def vad_s(signal, fs):
            wlen = 320
            inc = 160

            N = len(signal)
            if N <= wlen:
                nf = 1  # If the signal length is less than the length of one frame , Then the number of frames is defined as 1
            else:
                nf = int(
                    np.ceil((1.0 * N - wlen + inc) / inc)
                )  # otherwise , Calculate the total length of the frame

            pad_length = int(
                (nf - 1) * inc + wlen
            )  # The total flattened length of all frames added up
            zeros = np.zeros(
                (pad_length - N,)
            )  # Not enough length to use 0 fill , Be similar to FFT Extended array operation in
            pad_signal = np.concatenate(
                (signal, zeros)
            )  # The filled signal is recorded as pad_signal
            indices = (
                np.tile(np.arange(0, wlen), (nf, 1))
                + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
            )  # It is equivalent to extracting the time points of all frames , obtain nf*nw Matrix of length
            indices = np.array(
                indices, dtype=np.int32
            )  # take indices Convert to matrix
            frames = pad_signal[indices]  # Get the frame signal
            windown = np.hanning(wlen)
            time2 = np.arange(0, nf) * (inc * 1.0 / fs)
            ste = np.zeros(nf)
            for i in range(0, nf):
                a = frames[i : i + 1]
                b = a[0] * windown
                c = np.square(b)
                ste[i] = np.sum(c)
                log_energy = 10 * np.log10(ste / N)
                p1 = np.exp((2 / N) * log_energy)  # geometric mean
                # print(log_energy)
                # print(p1)
                p2 = (2 / N) * ste  # arithmatic mean
                sf = p1 / p2
            vad = sf < 25 * statistics.median(sf)
            return vad, time2

        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        pitchEn = snd.to_pitch()
        pitch_values = pitchEn.selected_array["frequency"]
        time_points = pitchEn.xs()
        f0 = pitch_values[pitch_values != 0]
        average_pitch = sum(f0) / len(f0)
        print(average_pitch)
        vad, time2 = vad_s(signal, fs)
        ############Display Plot############################################
        sns.set()  # Use seaborn's default style to make attractive graphs
        plt.rcParams["figure.dpi"] = 320  # Show images nicely
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(t, signal)
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
        plt.title("Time-Domain Signal", fontsize=10)
        plt.subplot(212)
        plt.plot(t, signal)
        plt.plot(time2, vad, c="black")
        frame4 = plt.gca()
        for xlabel_i in frame4.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
        plt.title("Break Indices", fontsize=9)
        splitted = [list(g) for i, g in itertools.groupby(vad, lambda x: x == True)]
        indices = [i for i, x in enumerate(vad) if not x]

        gg = [list(group) for group in mit.consecutive_groups(indices)]
        d_break = []
        for idx, row in enumerate(splitted):
            if all(n == False for n in row):
                if len(row) >= 10:
                    for idx1, row1 in enumerate(gg):
                        if len(row) == len(row1):
                            t = row1[0]
                            x = time2[t]
                            st = x
                            t1 = row1[len(row1) - 1]
                            et = time2[t1]
                            y = 0.25
                            plt.text(x, y, "3", {"color": "r", "fontsize": 15})
                            d_break.append([st, et, "LSIL"])
                elif len(row) >= 3:
                    for idx1, row1 in enumerate(gg):
                        if len(row) == len(row1):
                            t = row1[0]
                            x = time2[t]
                            st = x
                            t1 = row1[len(row1) - 1]
                            et = time2[t1]
                            y = 0.25
                            plt.text(x, y, "2", {"color": "r", "fontsize": 15})
                            d_break.append([st, et, "MSIL"])
                elif len(row) < 3:
                    for idx1, row1 in enumerate(gg):
                        if len(row) == len(row1):
                            t = row1[0]
                            x = time2[t]
                            st = x
                            t1 = row1[len(row1) - 1]
                            et = time2[t1]
                            y = 0.25
                            plt.text(x, y, "1", {"color": "r", "fontsize": 15})
                            d_break.append([st, et, "SSIL"])
        test_list = np.array(d_break)
        res_duplicate = list(set(map(lambda i: tuple(sorted(i)), test_list)))
        dd_break = sorted(res_duplicate, key=lambda x: x[1], reverse=False)
        out_break = open("data/ff_break.txt", "w")
        for lines in str(np.array(dd_break)):
            out_break.write(lines)
        out_break.close()
        os.system("tr -s '[],' ' ' <data/ff_break.txt >data/ff1_break.txt")
        f_break = open("data/ff1_break.txt", "r")
        lines2 = f_break.readlines()
        starttimes_b = []
        endtimes_b = []
        breaks = []
        for x2 in lines2:
            starttimes_b.append(x2.split()[0])
            endtimes_b.append(x2.split()[1])
            breaks.append(x2.split()[2])
        f_break.close()
        breaks_indices = [item.strip("'") for item in breaks]
        floats_b, floats_e_b = [eval(x) for x in starttimes_b], [
            eval(x) for x in endtimes_b
        ]
        d_break_indices = []
        for n3, v3, p3 in zip(floats_b, floats_e_b, breaks_indices):
            d_break_indices.append("{} {} {}".format(n3, v3, p3))

        b_table = open("data/break_t.txt", "w")
        for lines in d_break_indices:
            b_table.write(lines)
            b_table.write("\n")
        b_table.close()
        filename_intensity = "data/break_t.txt"
        ddd_break = pd.read_csv(filename_intensity, header=None, sep="\s{1,}")
        ddd_break.columns = ["Starttime", "Endtime", "Break_Indices"]
        print(ddd_break)
        out_break_index = file + ".break"
        out_break_index1 = open("../speech_new/web/static/" + out_break_index, "w")
        for lines in str(ddd_break):
            out_break_index1.write(lines)
        out_break_index1.close()
        new_graph_name = "graph" + str(time.time()) + ".jpg"
        for filename in os.listdir("../speech_new/web/static"):
            if filename.startswith("graph_"):
                os.remove("../speech_new/web/static/" + filename)
        plt.savefig("../speech_new/web/static/" + new_graph_name)

        plt.show()
        os.system("rm -f *.wav *.txt *.p *.phn p phn")

    return render_template(
        "table.html",
        name=file,
        data=ddd_break,
        graph=new_graph_name,
        file1=out_break_index,
    )


@app.route('/Tobi_version_I',methods = ['GET','POST'])
def Tobi_annotator():
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        text_file.save(text_file.filename)
        wave_file.save(wave_file.filename)
        f1 =wave_file.filename.split('\\')
        file = f1[-1].split('.')[0]
        os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
        os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
        val=request.form.get('languages')
        language= str(val)
        print(language)

        if language.lower() == 'english':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()

            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_eng data/temp dep_list/con_c_v_cv_list_eng >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_eng data/temp dep_list/con_c_v_cv_list_eng >data/tempp")
            os.system("./scripts/syllable_from_char_eng data/tempp dep_list/v_list_eng dep_list/c_list_eng dep_list/CV_list_eng >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")

        elif language =='Tamil':
            with open(text_file.filename, 'r') as f_t:
                text_content = f_t.read()

            with open('te.txt', 'w') as output_file:
                output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >t.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("rm ../speech_new/web/static/*.jhttp://10.23.13.12:5000pg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/sil/ /g' temp.txt")
            os.system("sed -i 's/sil/SIL/g' lists/out_word")
            os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
            os.system("./scripts/startsil p")
            os.system("mv temp.txt data")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_tam data/temp dep_list/con_c_v_cv_list_tam >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_tam data/temp dep_list/con_c_v_cv_list_tam >data/tempp")
            os.system("./scripts/syllable_from_char_tam data/tempp dep_list/v_list_tam dep_list/c_list_tam dep_list/CV_list_tam >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")

        else:
            # run_lang_relative(language,text_file,wave_file)

            run_lang_all(language,text_file,wave_file)

        fs=open('data/temp_syl_syl.lab',"r")
        lines1=fs.readlines()
        starttimes1=[]
        endtimes1=[]
        syllable=[]
        for x1 in lines1:
              starttimes1.append(x1.split()[0])
              endtimes1.append(x1.split()[1])
              syllable.append(x1.split()[2])
        fs.close()
        floats1,floats1_e = [eval(x1) for x1 in starttimes1],[eval(x1) for x1 in endtimes1]
        seg1=[]
        for s1 in floats1:
            seg1.append(s1/10000000)
        seg1_e=[]
        ############################ MAIN #############################################
        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        pitchEn = snd.to_pitch()
        pitch_values = pitchEn.selected_array['frequency']
        time_points = pitchEn.xs()
        f0 = pitch_values[pitch_values != 0]
        average_pitch = sum(f0) / len(f0)
        print(average_pitch)

        sns.set() # Use seaborn's default style to make attractive graphs
        plt.rcParams['figure.dpi'] = 320 # Show images nicely
        plt.figure(figsize=(10,5))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude",fontsize=8)
        plt.title("Time-Domain Signal",fontsize=10)
        plt.grid(False)
        plt.subplot(2, 1, 2)
        pitch_values = pitchEn.selected_array['frequency']
        plt.plot(pitchEn.xs(), pitch_values, '.', markersize=2.5)
        plt.ylim(0, pitchEn.ceiling)
        frame3 = plt.gca()
        for xlabel_i in frame3.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("freq[Hz]", fontsize=8)
        plt.title("Pitch Contour", fontsize=10)
        plt.grid(True)
        d_tobi = []
        d1_tobi=[]
        p_seg1 = []
        for s1 in floats1:
            p_seg1.append((s1 / 10000000))
        p_seg1_e = []
        for s1 in floats1_e:
            p_seg1_e.append((s1 / 10000000))


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

            len_f = len(l)
            if len_f > 1:
                xin = np.arange(0, len_f)
                yin = l
            elif len_f == 1 or len_f < 1:
                yin = 0
                xin = 0
                pf = 0
                continue

            pf = np.polyfit(xin, yin, 3)
            yin_smooth = np.polyval(pf, xin)
            dy_r2 = max(yin_smooth) - min(yin_smooth)
            print(yin_smooth)
            dynamic_ranges2.append(dy_r2)
            with open(file + 'dynamic2.txt', 'w') as doc2:
                for item in dynamic_ranges2:
                    doc2.write(str(item) + '\n')
            if dy_r2 < 10:
                x1 = j
                y1 = pitchEn.ceiling / 4
                st1 = i
                et1 = j
                print(st1)
                d_tobi.append([st1, et1, 'flat'])
                plt.text(x1,y1,'flat',{'color':'r','fontsize':7})
                print("flat")

            else:
                for ss in range(0, len(yin_smooth) - 1):
                    ff.append(yin_smooth[ss + 1] - yin_smooth[ss])
                    ff1 = abs(max(yin_smooth) - min(yin_smooth))
                splitted = [list(g) for i, g in itertools.groupby(ff, lambda xs: xs < 0)]
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

                for idx, row in enumerate(splitted):

                    if all(n > 0 for n in row):
                        if idx == 0:
                            index.append('p1')
                            len_f.append(len(row))
                        elif idx == 1:
                            index.append("p2")
                            len_f.append(len(row))
                        elif idx == 2:
                            index.append("p3")
                            len_f.append(len(row))
                    else:
                        if idx == 0:
                            index.append("q1")
                            len_f.append(len(row))
                        elif idx == 1:
                            index.append("q2")
                            len_f.append(len(row))
                        elif idx == 2:
                            index.append("q3")
                            len_f.append(len(row))
                a = np.concatenate((index, len_f))
                print(a)
                l_a = len(a)
                if l_a == 2:

                    if a[0] == 'p1' and 10 <= ff1 < 60:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        t1 = np.arange(0, len(yin_smooth))
                        d_tobi.append([st1, et1, 'S_H'])
                        plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})


                    elif a[0] == 'p1' and 60 <= ff1 < 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        t1 = np.arange(0, len(yin_smooth))

                        d_tobi.append([st1, et1, 'M_H'])
                        plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                    elif a[0] == 'p1' and ff1 >= 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        t1 = np.arange(0, len(yin_smooth))

                        d_tobi.append([st1, et1, 'B_H'])
                        plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})

                    elif a[0] == 'q1' and 10 <= ff1 < 60:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j

                        d_tobi.append([st1, et1, 'S_L'])
                        plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                    elif a[0] == 'q1' and 60 <= ff1 < 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi.append([st1, et1, 'M_L'])
                        plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                    elif a[0] == 'q1' and ff1 >= 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi.append([st1, et1, 'B_L'])
                        plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})

                elif l_a == 4:
                    if a[0] == 'p1' and a[1] == 'q2':
                        if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_L'])
                                plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})

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
                                    d_tobi.append([st1, et1, 'S_hat'])
                                    plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_HLL'])
                                    plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})

                        elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_L'])
                                plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'M_hat'])
                                    plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_HLL'])
                                    plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})

                        elif int(a[2]) < int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_L'])
                                plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'B_hat'])
                                    plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_HLL'])
                                    plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})

                            k1 = k1 + 1

                        elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_H'])
                                plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'S_hat'])
                                    plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_HHL'])
                                    plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})


                        elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_H'])
                                plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'M_hat'])
                                    plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_HHL'])
                                    plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_H'])
                                plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'B_hat'])
                                    plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_HHL'])
                                    plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})

                        elif int(a[2]) == int(a[3]):
                            if 10 <= ff1 < 60:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_hat'])
                                plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                            elif 60 <= ff1 < 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_hat'])
                                plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                            elif ff1 >= 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_hat'])
                                plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})

                    elif a[0] == 'q1' and a[1] == 'p2':
                        if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_H'])
                                plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'S_bucket'])
                                    plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_LHH'])
                                    plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})


                        elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_H'])
                                plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'M_bucket'])
                                    plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_LHH'])
                                    plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})

                        elif int(a[2]) < int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_H'])
                                plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'B_bucket'])
                                    plt.text(x1,y1,'B_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_LHH'])
                                    plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_L'])
                                plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'S_bucket'])
                                    plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_LLH'])
                                    plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_L'])
                                plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'M_bucket'])
                                    plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_LLH'])
                                    plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and ff1 > 100:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_L'])
                                plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
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
                                    d_tobi.append([st1, et1, 'B_bucket'])
                                    plt.text(x1,y1,'B_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_LLH'])
                                    plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})



                        elif int(a[2]) == int(a[3]):
                            if 10 <= ff1 < 60:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_bucket'])
                                plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                print("bucket")
                            elif 60 <= ff1 < 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_bucket'])
                                plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                print("bucket")
                            elif ff1 >= 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_bucket'])
                                plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                print("bucket")


                elif l_a == 6:
                    if a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 10 <= ff1 < 60:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LHH'])
                            plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LLH'])
                            plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HLL'])
                            plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HHL'])
                            plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_L'])
                            plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HLH'])
                            plt.text(x1,y1,'M_HLH',{'color':'r','fontsize':7})


                    elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 60 <= ff1 < 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LHH'])
                            plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LLH'])
                            plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HLL'])
                            plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HHL'])
                            plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_L'])
                            plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HLH'])
                            plt.text(x1,y1,'M_HLH',{'color':'r','fontsize':7})


                    elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and ff1 >= 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LHH'])
                            plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LLH'])
                            plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HLL'])
                            plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HHL'])
                            plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_L'])
                            plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HLH'])
                            plt.text(x1,y1,'B_HLH',{'color':'r','fontsize':7})


                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 10 <= ff1 < 60:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HLL'])
                            plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HHL'])
                            plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LHH'])
                            plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LLH'])
                            plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_H'])
                            plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LHL'])
                            plt.text(x1,y1,'S_LHL',{'color':'r','fontsize':7})


                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 60 <= ff1 < 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HLL'])
                            plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HHL'])
                            plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LHH'])
                            plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LLH'])
                            plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_H'])
                            plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LHL'])
                            plt.text(x1,y1,'M_LHL',{'color':'r','fontsize':7})

                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and ff1 >= 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HLL'])
                            plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HHL'])
                            plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LHH'])
                            plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LLH'])
                            plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_H'])
                            plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LHL'])
                            plt.text(x1,y1,'B_LHL',{'color':'r','fontsize':7})


        print(d_tobi)
        ddd_tobi = pd.DataFrame(d_tobi)
        ddd_tobi.columns = ['Starttime', 'Endtime', 'Labels']
        print(ddd_tobi)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        out_pitch = file + ".pitch"
        out_pitch1=open('../speech_new/web/static/' + out_pitch, "w")
        for lines in str(ddd_tobi):
            out_pitch1.write(lines)
        out_pitch1.close()
        new_graph_name = "graph" +str(time.time()) +".jpg"
        for filename in os.listdir('../speech_new/web/static'):
            if filename.startswith('graph_'):
                os.remove('../speech_new/web/static/'+filename)
        plt.savefig('../speech_new/web/static/' + new_graph_name)
        plt.show()
        os.system("rm -f *.wav *.txt *.p *.phn p phn")


    return render_template('table.html', name=file, data=ddd_tobi, graph=new_graph_name, file1=out_pitch)


@app.route('/syllable',methods = ['GET','POST'])
def syllable_segmentation():
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        text_file.save(text_file.filename)
        wave_file.save(wave_file.filename)
        f1 =wave_file.filename.split('\\')
        file = f1[-1].split('.')[0]
        os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
        os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
        val=request.form.get('languages')
        language= str(val)

        if language.lower() == 'English':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <text.txt >temp.txt")
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            ####################PHONEME SEGMENTATION######################################
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            os.system("./scripts/ortho_to_phonetic1 data/temp.txt phoneset_smp >data/temp.lab")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            ########################SYLLABLE SEGMENTATION##################################
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_eng data/temp dep_list/con_c_v_cv_list_eng >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_eng data/temp dep_list/con_c_v_cv_list_eng >data/tempp")
            os.system("./scripts/syllable_from_char_eng data/tempp dep_list/v_list_eng dep_list/c_list_eng dep_list/CV_list_eng >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")

        elif language.lower() =='tamil':
            with open(text_file.filename, 'r') as f_t:
                text_content = f_t.read()

            with open('te.txt', 'w') as output_file:
                output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >t.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("rm ../speech_new/web/static/*.jhttp://10.23.13.12:5000pg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/sil/ /g' temp.txt")
            os.system("sed -i 's/sil/SIL/g' lists/out_word")
            os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
            os.system("./scripts/startsil p")
            os.system("mv temp.txt data")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_tam data/temp dep_list/con_c_v_cv_list_tam >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_tam data/temp dep_list/con_c_v_cv_list_tam >data/tempp")
            os.system("./scripts/syllable_from_char_tam data/tempp dep_list/v_list_tam dep_list/c_list_tam dep_list/CV_list_tam >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")
        else:
            # run_lang_relative(language,text_file,wave_file)
            run_lang_all(language,text_file,wave_file)


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
        floats1, floats1_e = [eval(x1) for x1 in starttimes1], [eval(x1) for x1 in endtimes1]
        syl_id = [x1 for x1 in syllable]
        seg1 = []
        for s1 in floats1:
            seg1.append(s1 / 10000000)
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
        print(ddd_syllable)
        out_syllable = file + ".syl"
        out_syllable1=open('../speech_new/web/static/' + out_syllable, "w")
        for lines in str(ddd_syllable):
            out_syllable1.write(lines)
        out_syllable1.close()
        ############################ MAIN #############################################
        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        ########################DISPLAY THE RESULTS####################################
        sns.set() # Use seaborn's default style to make attractive graphs
        plt.rcParams['figure.dpi'] = 320 # Show images nicely
        plt.figure(figsize=(10,5))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude",fontsize=8)
        # plt.xlabel("Samples/sec")
        plt.title("Time-Domain Signal",fontsize=10)
        plt.grid(False)

        plt.subplot(2, 1, 2)
        plt.plot(snd.xs(), snd.values.T)
        frame3 = plt.gca()
        for xlabel_i in frame3.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
        plt.title("Syllable Segmentation", fontsize=10)
        for s_l1 in seg1:
            plt.axvline(x=s_l1, color='black')

        new_graph_name = "graph" +str(time.time()) +".jpg"
        for filename in os.listdir('../speech_new/web/static'):
            if filename.startswith('graph_'):
                os.remove('../speech_new/web/static/'+filename)
        plt.savefig('../speech_new/web/static/' + new_graph_name)
        plt.show()
        os.system("rm -f *.wav *.txt *.p *.phn p phn")

    return render_template('table.html', name=file, data=ddd_syllable, graph=new_graph_name,file1=out_syllable)

@app.route('/speech_api',methods = ['GET','POST'])
def api_edit():
    auth_token = request.headers.get('Authorization')
    if auth_token != 'W1VfgQ9GeVxfpCj79mFiCX0tsKFWSpqM':
        return jsonify({'message': 'Unauthorized access'}), 401
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        if text_file.filename:
            text_file.save(text_file.filename)
        if wave_file.filename:
            wave_file.save(wave_file.filename)
        print(wave_file.filename)
        print(text_file.filename)
        if wave_file.filename:
            f1 =wave_file.filename.split('\\')
            file = f1[-1].split('.')[0]
            os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
            os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
            val=request.form.get('languages')
            language= str(val)
            print(language)
        if language.lower() == 'english':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_eng data/temp dep_list/con_c_v_cv_list_eng >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_eng data/temp dep_list/con_c_v_cv_list_eng >data/tempp")
            os.system("./scripts/syllable_from_char_eng data/tempp dep_list/v_list_eng dep_list/c_list_eng dep_list/CV_list_eng >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")
            os.system("./scripts/word_segment_modified data/temp.rec data/temp.txt")

        elif language.lower() =='tamil':
           with open(text_file.filename, 'r') as f_t:
              text_content = f_t.read()

           with open('te.txt', 'w') as output_file:
              output_file.write(text_content)
           os.system("tr -d '[:punct:]' <te.txt >t.txt")
           os.system("rm -r data")
           os.system("mkdir data")
           os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
           os.system("rm ../speech_new/web/static/*.jhttp://10.23.13.12:5000pg")
           os.system("rm ../speech_new/web/static/*.phn")
           os.system("rm ../speech_new/web/static/*.syl")
           os.system("rm ../speech_new/web/static/*.wrd")
           os.system("rm ../speech_new/web/static/*.intensity")
           os.system("rm ../speech_new/web/static/*.b")
           os.system("rm ../speech_new/web/static/*.pitch")
           os.system("sed -i 's/ /sil/g' t.txt")
           os.system("perl scripts/vuv.pl t.txt")
           os.system("cp -r lists/out_word temp.txt")
           os.system("sed -i 's/ /sil/g' t.txt")
           os.system("perl scripts/vuv.pl t.txt")
           os.system("cp -r lists/out_word temp.txt")
           os.system("sed -i 's/sil/ /g' temp.txt")
           os.system("sed -i 's/sil/SIL/g' lists/out_word")
           os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
           os.system("./scripts/startsil p")
           os.system("mv temp.txt data")
           os.system("ls -v data/*.wav >data/wavlist")
           os.system("ls -v data/*.lab >data/lablist")
           os.system("./scripts/create_maptable data/wavlist")
           os.system("HCopy -C config_files/speech_config_file -S data/maptable")
           os.system("cat data/*.lab | sort | uniq >data/wordlist")
           os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
           os.system("./scripts/mlf data/lablist >data/mlf")
           os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
           os.system("./scripts/phoneme data/temp.rec")
           os.system("tr -d '\n' <data/temp.lab >data/temp")
           os.system("./scripts/ortho_to_phonetic_cv_tam data/temp dep_list/con_c_v_cv_list_tam >data/t.led")
           os.system("cp data/temp.rec data/temp_syl.lab")
           os.system("HLEd data/t.led data/temp_syl.lab")
           os.system("./scripts/ortho_to_phonetic_syllable_tam data/temp dep_list/con_c_v_cv_list_tam >data/tempp")
           os.system("./scripts/syllable_from_char_tam data/tempp dep_list/v_list_tam dep_list/c_list_tam dep_list/CV_list_tam >data/temp.led")
           os.system("HLEd data/temp.led data/temp_syl.lab")
           os.system("./scripts/syllable data/temp_syl.lab")
           os.system("./scripts/word_segment_modified_tamil data/temp.rec data/temp.txt")

        else:
            run_lang_all(language,text_file,wave_file)


        fss=open('data/temp_word.lab',"r")
        lines2=fss.readlines()
        starttimes2=[]
        endtimes2=[]
        word=[]
        for x2 in lines2:
            starttimes2.append(x2.split()[0])
            endtimes2.append(x2.split()[1])
            word.append(x2.split()[2])
        fss.close()
        print(word)
        floats2,floats2_e = [eval(x1) for x1 in starttimes2],[eval(x1) for x1 in endtimes2]
        seg2=[]
        for s2 in floats2:
            seg2.append(s2/10000000)
        seg2_e=[]
        for s2 in floats2_e:
            seg2_e.append(s2/10000000)
        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        pitchEn = snd.to_pitch()
        pitch_values = pitchEn.selected_array['frequency']
        time_points = pitchEn.xs()
        f0 = pitch_values[pitch_values != 0]
        average_pitch = sum(f0) / len(f0)
        print(average_pitch)
        d_tobi_w =[]
        p_seg2 = []
        for s1 in floats2:
            p_seg2.append((s1 / 10000000))
        p_seg2_e = []
        for s1 in floats2_e:
            p_seg2_e.append((s1 / 10000000))

        dynamic_ranges1 = []
        dynamic_ranges2 = []
        k1 = 1
        for i, j, k in zip_longest(p_seg2, p_seg2_e, word, fillvalue=None):
            print(i, j, k)
            start_index = np.argmax(time_points >= i)
            end_index = np.argmax(time_points >= j)
            spliced_pitch_values = pitch_values[start_index:end_index]
            spliced_time_points = time_points[start_index:end_index]
            res = spliced_pitch_values[spliced_pitch_values != 0]
            snd1 = snd.extract_part(from_time=i, to_time=j)
            ff = []
            ff1 = []

            l = [x for x in res if ~np.isnan(x)]
            if l == []:
                dy_r1 = 0
            else:
                dy_r1 = max(l) - min(l)
                dynamic_ranges1.append(dy_r1)

            len_f = len(l)
            if len_f > 1:
                xin = np.arange(0, len_f)
                yin = l
            elif len_f == 1 or len_f < 1:
                yin = 0
                xin = 0
                pf = 0
                continue
            pf = np.polyfit(xin, yin, 3)
            yin_smooth = np.polyval(pf, xin)
            dy_r2 = max(yin_smooth) - min(yin_smooth)
            print(yin_smooth)
            dynamic_ranges2.append(dy_r2)
            if dy_r2 < 10:
                fig, axs = plt.subplots(3, 1)
                x1 = j
                y1 = pitchEn.ceiling / 4
                st1 = i
                et1 = j
                d_tobi_w.append([k, '(flat)'])

            else:
                for ss in range(0, len(yin_smooth) - 1):
                    ff.append(yin_smooth[ss + 1] - yin_smooth[ss])
                    ff1 = abs(max(yin_smooth) - min(yin_smooth))
                splitted = [list(g) for i, g in itertools.groupby(ff, lambda xs: xs < 0)]
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

                for idx, row in enumerate(splitted):

                    if all(n > 0 for n in row):
                        if idx == 0:
                            index.append('p1')
                            len_f.append(len(row))
                        elif idx == 1:
                            index.append("p2")
                            len_f.append(len(row))
                        elif idx == 2:
                            index.append("p3")
                            len_f.append(len(row))
                    else:
                        if idx == 0:
                            index.append("q1")
                            len_f.append(len(row))
                        elif idx == 1:
                            index.append("q2")
                            len_f.append(len(row))
                        elif idx == 2:
                            index.append("q3")
                            len_f.append(len(row))
                a = np.concatenate((index, len_f))
                print(a)
                l_a = len(a)
                fig, axs = plt.subplots(3, 1)
                if l_a == 2:

                    if a[0] == 'p1' and 10 <= ff1 < 60:
                        x1 = j / fs
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi_w.append([k, '(S_H)'])


                    elif a[0] == 'p1' and 60 <= ff1 < 100:
                        x1 = j / fs
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi_w.append([k, '(M_H)'])

                    elif a[0] == 'p1' and ff1 >= 100:
                        x1 = j / fs
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi_w.append([k, '(B_H)'])


                    elif a[0] == 'q1' and 10 <= ff1 < 60:
                        x1 = j / fs
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi_w.append([k, '(S_L)'])
                    elif a[0] == 'q1' and 60 <= ff1 < 100:
                        x1 = j / fs
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi_w.append([k, '(M_L)'])
                    elif a[0] == 'q1' and ff1 >= 100:
                        x1 = j / fs
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi_w.append([k, '(B_L)'])

                elif l_a == 4:
                    if a[0] == 'p1' and a[1] == 'q2':
                        if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(S_L)'])

                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_hat)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_HLL)'])
                        elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(M_L)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_hat)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_HLL)'])
                        elif int(a[2]) < int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(B_L)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_hat)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_HLL)'])

                        elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(S_H)'])
                            else:
                                thresh_p = abs(yin_smooth[f_p_i] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[f_n_i + 1]) - (yin_smooth[l_n_i + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_hat)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_HHL)'])

                        elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(M_H)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_hat)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_HHL)'])
                        elif int(a[2]) > int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(B_H)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_hat)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_HHL)'])
                        elif int(a[2]) == int(a[3]):
                            if 10 <= ff1 < 60:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(S_hat)'])
                            elif 60 <= ff1 < 100:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(M_hat)'])
                            elif ff1 >= 100:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(B_hat)'])
                    elif a[0] == 'q1' and a[1] == 'p2':
                        if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(S_H)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_bucket)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_LHH)'])

                        elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(M_H)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_bucket)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_LHH)'])
                        elif int(a[2]) < int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(B_H)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_bucket)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_LHH)'])

                        elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(S_L)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_bucket)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(S_LLH)'])
                        elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(M_L)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_bucket)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(M_LLH)'])
                        elif int(a[2]) > int(a[3]) and ff1 > 100:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(B_L)'])
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                print(thresh_p)
                                print(thresh_n)
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_bucket)'])
                                else:
                                    st1 = i
                                    et1 = j
                                    d_tobi_w.append([k, '(B_LLH)'])


                        elif int(a[2]) == int(a[3]):
                            if 10 <= ff1 < 60:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(S_bucket)'])
                            elif 60 <= ff1 < 100:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(M_bucket)'])
                            elif ff1 >= 100:
                                st1 = i
                                et1 = j
                                d_tobi_w.append([k, '(B_bucket)'])

                elif l_a == 6:
                    if a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 10 <= ff1 < 60:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_LHH)'])
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_LLH)'])
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_HLL)'])
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_HHL)'])
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_L)'])
                        else:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_HLH)'])

                    elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 60 <= ff1 < 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_LHH)'])
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_LLH)'])
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_HLL)'])
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_HHL)'])
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_L)'])
                        else:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_HLH)'])

                    elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and ff1 >= 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_LHH)'])
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_LLH)'])
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_HLL)'])
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_HHL)'])
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_L)'])
                        else:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_HLH)'])

                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 10 <= ff1 < 60:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_HLL)'])
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_HHL)'])
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_LHH)'])
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_LLH)'])
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_H)'])
                        else:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(S_LHL)'])

                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 60 <= ff1 < 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_HLL)'])
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_HHL)'])
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_LHH)'])
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_LLH)'])
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_H)'])
                        else:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(M_LHL)'])
                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and ff1 >= 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_HLL)'])
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_HHL)'])
                            # save_dir = '/home/speechlab/sooriya/complete_demo_Wave_modified_Final/labels/B_HHL'
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_LHH)'])
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_LLH)'])
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_H)'])
                        else:
                            st1 = i
                            et1 = j
                            d_tobi_w.append([k, '(B_LHL)'])
        combined_output = ' '.join([' '.join(pair) for pair in d_tobi_w])
        os.system("rm -f *.wav *.txt *.p *.phn p phn")

        return combined_output


@app.route('/word',methods = ['GET','POST'])
def word_segmentation():
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        text_file.save(text_file.filename)
        wave_file.save(wave_file.filename)
        f1 =wave_file.filename.split('\\')
        file = f1[-1].split('.')[0]
        os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
        os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
        val=request.form.get('languages')
        language= str(val)
        print(language)

        if language.lower() == 'english':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()

            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            os.system("./scripts/word_segment_modified data/temp.rec data/temp.txt")

        elif language.lower() =='tamil':
           with open(text_file.filename, 'r') as f_t:
              text_content = f_t.read()
           with open('te.txt', 'w') as output_file:
              output_file.write(text_content)
           os.system("tr -d '[:punct:]' <te.txt >t.txt")
           os.system("rm -r data")
           os.system("mkdir data")
           os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
           os.system("rm ../speech_new/web/static/*.jpg")
           os.system("rm ../speech_new/web/static/*.phn")
           os.system("rm ../speech_new/web/static/*.syl")
           os.system("rm ../speech_new/web/static/*.wrd")
           os.system("rm ../speech_new/web/static/*.intensity")
           os.system("rm ../speech_new/web/static/*.b")
           os.system("rm ../speech_new/web/static/*.pitch")
           os.system("sed -i 's/ /sil/g' t.txt")
           os.system("perl scripts/vuv.pl t.txt")
           os.system("cp -r lists/out_word temp.txt")
           os.system("sed -i 's/ /sil/g' t.txt")
           os.system("perl scripts/vuv.pl t.txt")
           os.system("cp -r lists/out_word temp.txt")
           os.system("sed -i 's/sil/ /g' temp.txt")
           os.system("sed -i 's/sil/SIL/g' lists/out_word")
           os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
           os.system("./scripts/startsil p")
           os.system("mv temp.txt data")
           os.system("ls -v data/*.wav >data/wavlist")
           os.system("ls -v data/*.lab >data/lablist")
           os.system("./scripts/create_maptable data/wavlist")
           os.system("HCopy -C config_files/speech_config_file -S data/maptable")
           os.system("cat data/*.lab | sort | uniq >data/wordlist")
           os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
           os.system("./scripts/mlf data/lablist >data/mlf")
           os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
           os.system("./scripts/phoneme data/temp.rec")
           os.system("./scripts/word_segment_modified_tamil data/temp.rec data/temp.txt")

        else:
            # run_word(language,text_file,wave_file)
            run_lang_all(language,text_file,wave_file)

        # run_word(language,text_file,wave_file)
        ##########################WORD SEGMENTATION####################################
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
        floats2, floats2_e = [eval(x1) for x1 in starttimes2], [eval(x1) for x1 in endtimes2]
        seg2 = []
        for s2 in floats2:
            seg2.append(s2 / 10000000)
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
        print(ddd_word)
        out_word = file + ".word"
        out_word1=open('../speech_new/web/static/' + out_word, "w")
        for lines in str(ddd_word):
            out_word1.write(lines)
        out_word1.close()
        ############################ MAIN #############################################
        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        ########################DISPLAY THE RESULTS####################################
        sns.set() # Use seaborn's default style to make attractive graphs
        plt.rcParams['figure.dpi'] = 320 # Show images nicely
        plt.figure(figsize=(10,5))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude",fontsize=8)
        # plt.xlabel("Samples/sec")
        plt.title("Time-Domain Signal",fontsize=10)
        plt.grid(False)
        plt.subplot(2, 1, 2)
        plt.plot(snd.xs(), snd.values.T)
        plt.ylabel("Amplitude", fontsize=8)
        plt.xlabel("sec --->")
        plt.title("Word Segmentation", fontsize=10)
        for s_l2 in seg2:
            plt.axvline(x=s_l2, color='black')

        new_graph_name = "graph" +str(time.time()) +".jpg"
        for filename in os.listdir('../speech_new/web/static'):
            if filename.startswith('graph_'):
                os.remove('../speech_new/web/static/'+filename)
        plt.savefig('../speech_new/web/static/' + new_graph_name)
        plt.show()
        os.system("rm -f *.wav *.txt *.p *.phn p phn")

    return render_template('table.html',name=file, data=ddd_word, graph=new_graph_name, file1=out_word)


@app.route('/phoneme', methods=['GET', 'POST'])
def phoneme_segmentation():
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        text_file.save(text_file.filename)
        wave_file.save(wave_file.filename)
        f1 =wave_file.filename.split('\\')
        file = f1[-1].split('.')[0]
        os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
        os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
        val=request.form.get('languages')
        language= str(val)
        print(language)

        if language == 'english':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")


        elif language =='Tamil':
            with open(text_file.filename, 'r') as f_t:
                text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
                output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >t.txt")
            #####################PHONEME SEGMENTATION######################################
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("rm ../speech_new/web/static/*.jhttp://10.23.13.12:5000pg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/ /sil/g' t.txt")
            os.system("perl scripts/vuv.pl t.txt")
            os.system("cp -r lists/out_word temp.txt")
            os.system("sed -i 's/sil/ /g' temp.txt")
            os.system("sed -i 's/sil/SIL/g' lists/out_word")
            os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
            os.system("./scripts/startsil p")
            os.system("mv temp.txt data")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
        # run_phoneme(language,text_file,wave_file)
        else:
            # run_phoneme(language,text_file,wave_file)
            run_lang_all(language,text_file,wave_file)

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
        floats, floats_e = [eval(x) for x in starttimes], [eval(x) for x in endtimes]
        seg = []
        for s in floats:
            seg.append(s / 10000000)
        seg_e = []
        for s in floats_e:
            seg_e.append(s / 10000000)
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
        print(ddd_phoneme)
        out_phoneme = file + ".phn"
        out_phoneme1=open('../speech_new/web/static/' + out_phoneme, "w")
        for lines in str(ddd_phoneme):
            out_phoneme1.write(lines)
        out_phoneme1.close()
        ############################ MAIN #############################################
        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
        ########################DISPLAY THE RESULTS####################################
        sns.set() # Use seaborn's default style to make attractive graphs
        plt.rcParams['figure.dpi'] = 320 # Show images nicely
        plt.figure(figsize=(10,5))
        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude",fontsize=8)
        plt.title("Time-Domain Signal",fontsize=10)
        plt.grid(False)
        plt.subplot(2, 1, 2)
        plt.plot(snd.xs(), snd.values.T)
        frame3 = plt.gca()
        for xlabel_i in frame3.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
        plt.title("Phoneme Segmentation", fontsize=10)
        for s_l in seg:
            plt.axvline(x=s_l, color='black')

        new_graph_name = "graph" +str(time.time()) +".jpg"
        for filename in os.listdir('../speech_new/web/static'):
            if filename.startswith('graph_'):
                os.remove('../speech_new/web/static/'+filename)
        plt.savefig('../speech_new/web/static/' + new_graph_name)
        plt.show()

    return render_template('table.html', name=file, data=ddd_phoneme, graph=new_graph_name, file1=out_phoneme)

@app.route('/upload_demo',methods = ['GET','POST'])
def full_code():
    if request.method == 'POST':
        text_file = request.files['text_file']
        wave_file = request.files['wave_file']
        text_file.save(text_file.filename)
        wave_file.save(wave_file.filename)
        f1 =wave_file.filename.split('\\')
        file = f1[-1].split('.')[0]
        os.system("ch_wave " + wave_file.filename + " -F 16000 -otype wav -o temp.wav")
        os.system("ch_wave temp.wav -c 1 -otype wav -o temp.wav")
        val=request.form.get('languages')
        language= str(val)

        ##########################Speech to Text#######################################
        if language.lower() == 'english':
            with open(text_file.filename, 'r') as f_t:
               text_content = f_t.read()
            with open('te.txt', 'w') as output_file:
               output_file.write(text_content)
            os.system("tr -d '[:punct:]' <te.txt >temp.txt")
            #####################PHONEME SEGMENTATION######################################
            os.system("rm -r data")
            os.system("mkdir data")
            os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
            os.system("mv temp.txt data")
            os.system("rm ../speech_new/web/static/*.jpg")
            os.system("rm ../speech_new/web/static/*.phn")
            os.system("rm ../speech_new/web/static/*.syl")
            os.system("rm ../speech_new/web/static/*.wrd")
            os.system("rm ../speech_new/web/static/*.intensity")
            os.system("rm ../speech_new/web/static/*.b")
            os.system("rm ../speech_new/web/static/*.pitch")
            with open("data/temp.txt") as f:
                texts = f.read()
            g2p = G2p()
            out = g2p(texts)
            outf = open("file.txt", "w")
            for lines in out:
                outf.write(lines)
                outf.write("\n")
            outf.close()
            os.system("./scripts/map_eng file.txt ./scripts/english_map")
            os.system("ls -v data/*.wav >data/wavlist")
            os.system("ls -v data/*.lab >data/lablist")
            os.system("./scripts/create_maptable data/wavlist")
            os.system("HCopy -C config_files/speech_config_file -S data/maptable")
            os.system("cat data/*.lab | sort | uniq >data/wordlist")
            os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
            os.system("./scripts/mlf data/lablist >data/mlf")
            os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmE dep_list/dictE data/wordlist data/temp.mfc")
            os.system("./scripts/phoneme data/temp.rec")
            ########################SYLLABLE SEGMENTATION##################################
            os.system("tr -d '\n' <data/temp.lab >data/temp")
            os.system("./scripts/ortho_to_phonetic_cv_eng data/temp dep_list/con_c_v_cv_list_eng >data/t.led")
            os.system("cp data/temp.rec data/temp_syl.lab")
            os.system("HLEd data/t.led data/temp_syl.lab")
            os.system("./scripts/ortho_to_phonetic_syllable_eng data/temp dep_list/con_c_v_cv_list_eng >data/tempp")
            os.system("./scripts/syllable_from_char_eng data/tempp dep_list/v_list_eng dep_list/c_list_eng dep_list/CV_list_eng >data/temp.led")
            os.system("HLEd data/temp.led data/temp_syl.lab")
            os.system("./scripts/syllable data/temp_syl.lab")
            ##########################WORD SEGMENTATION####################################
            os.system("./scripts/word_segment_modified data/temp.rec data/temp.txt")

        elif language.lower() =='tamil':
           with open(text_file.filename, 'r') as f_t:
              text_content = f_t.read()
           with open('te.txt', 'w') as output_file:
              output_file.write(text_content)
           os.system("tr -d '[:punct:]' <te.txt >t.txt")
           #####################PHONEME SEGMENTATION######################################
           os.system("rm -r data")
           os.system("mkdir data")
           os.system("ch_wave temp.wav -itype wav -otype nist -o data/temp.wav")
           os.system("rm ../speech_new/web/static/*.phn")
           os.system("rm ../speech_new/web/static/*.syl")
           os.system("rm ../speech_new/web/static/*.wrd")
           os.system("rm ../speech_new/web/static/*.intensity")
           os.system("rm ../speech_new/web/static/*.b")
           os.system("rm ../speech_new/web/static/*.pitch")
           os.system("sed -i 's/ /sil/g' t.txt")
           os.system("perl scripts/vuv.pl t.txt")
           os.system("cp -r lists/out_word temp.txt")
           os.system("sed -i 's/ /sil/g' t.txt")
           os.system("perl scripts/vuv.pl t.txt")
           os.system("cp -r lists/out_word temp.txt")
           os.system("sed -i 's/sil/ /g' temp.txt")
           os.system("sed -i 's/sil/SIL/g' lists/out_word")
           os.system("./scripts/ortho_to_phonetic1_phoneme_tam lists/out_word dep_list/phonelist_tamil >p")
           os.system("./scripts/startsil p")
           os.system("mv temp.txt data")
           os.system("ls -v data/*.wav >data/wavlist")
           os.system("ls -v data/*.lab >data/lablist")
           os.system("./scripts/create_maptable data/wavlist")
           os.system("HCopy -C config_files/speech_config_file -S data/maptable")
           os.system("cat data/*.lab | sort | uniq >data/wordlist")
           os.system("./scripts/mix_state_list data/wordlist >data/mix_state_list")
           os.system("./scripts/mlf data/lablist >data/mlf")
           os.system("HVite -a -o SW -I data/mlf -H dep_models/all_hmmT dep_list/dictT data/wordlist data/temp.mfc")
           os.system("./scripts/phoneme data/temp.rec")
           ########################SYLLABLE SEGMENTATION##################################
           os.system("tr -d '\n' <data/temp.lab >data/temp")
           # os.system("sed -i 's/SIL/ /g' data/temp")
           os.system("./scripts/ortho_to_phonetic_cv_tam data/temp dep_list/con_c_v_cv_list_tam >data/t.led")
           os.system("cp data/temp.rec data/temp_syl.lab")
           os.system("HLEd data/t.led data/temp_syl.lab")
           os.system("./scripts/ortho_to_phonetic_syllable_tam data/temp dep_list/con_c_v_cv_list_tam >data/tempp")
           os.system("./scripts/syllable_from_char_tam data/tempp dep_list/v_list_tam dep_list/c_list_tam dep_list/CV_list_tam >data/temp.led")
           os.system("HLEd data/temp.led data/temp_syl.lab")
           os.system("./scripts/syllable data/temp_syl.lab")
           ##########################WORD SEGMENTATION####################################
           os.system("./scripts/word_segment_modified_tamil data/temp.rec data/temp.txt")

        else:
            run_lang_all(language,text_file,wave_file)

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
        floats, floats_e = [eval(x) for x in starttimes], [eval(x) for x in endtimes]

        seg = []
        for s in floats:
            seg.append(s / 10000000)
        seg_e = []
        for s in floats_e:
            seg_e.append(s / 10000000)
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
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print(ddd_phoneme)
        out_phoneme = file + ".phn"
        print(out_phoneme)
        out_phoneme1=open('../speech_new/web/static/' + out_phoneme, "w")
        for lines in str(ddd_phoneme):
            out_phoneme1.write(lines)
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
        floats1, floats1_e = [eval(x1) for x1 in starttimes1], [eval(x1) for x1 in endtimes1]
        syl_id = [x1 for x1 in syllable]
        seg1 = []
        for s1 in floats1:
            seg1.append(s1 / 10000000)
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
        print(ddd_syllable)
        out_syllable = file + ".syl"
        out_syllable1=open('../speech_new/web/static/' + out_syllable, "w")
        for lines in str(ddd_syllable):
            out_syllable1.write(lines)
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
        floats2, floats2_e = [eval(x1) for x1 in starttimes2], [eval(x1) for x1 in endtimes2]
        seg2 = []
        for s2 in floats2:
            seg2.append(s2 / 10000000)
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
        print(ddd_word)
        out_word = file + ".wrd"
        out_word1=open('../speech_new/web/static/' + out_word, "w")
        for lines in str(ddd_word):
            out_word1.write(lines)
        out_word1.close()
        def stride_trick(a, stride_length, stride_step):
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
            rest_samples = np.abs(N - frames_overlap) % np.abs(frame_length - frames_overlap)
            pad_signal = np.append(snd, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))
            frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
            e = np.sum(np.abs(np.fft.rfft(a=frames, n=len(frames))) ** 2, axis=-1) / len(frames) ** 2
            log_energy = e / N  # find the Intensity
            energy = scipy.signal.medfilt(log_energy, 5)
            energy = np.repeat(energy, frame_length)
            return energy

        def vad_s(signal, fs):
            wlen = 320
            inc = 160
            eps = 1e-10

            N = len(signal)

            if N <= wlen:
                nf = 1
            else:
                nf = int(np.ceil((1.0 * N - wlen + inc) / inc))

            pad_length = int((nf - 1) * inc + wlen)
            zeros = np.zeros((pad_length - N,))
            pad_signal = np.concatenate((signal, zeros))

            indices = np.tile(np.arange(0, wlen), (nf, 1)) + \
                      np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T

            indices = indices.astype(np.int32)
            frames = pad_signal[indices]

            windown = np.hanning(wlen)
            time2 = np.arange(0, nf) * (inc / fs)

            ste = np.zeros(nf)
            for i in range(nf):
                frame = frames[i] * windown
                ste[i] = np.sum(frame ** 2)

            log_energy = 10 * np.log10((ste + eps) / (N + eps))
            p1 = np.exp((2 / N) * log_energy)  # geometric mean
            p2 = (2 / N) * (ste + eps)  # arithmetic mean
            sf = p1 / p2
            vad = sf < (25 * statistics.median(sf))

            return vad, time2

        fs, signal = wav.read("temp.wav")
        snd = parselmouth.Sound("temp.wav")
        N = len(snd)
        fs = snd.sampling_frequency
        signal = signal / max(abs(signal))
        t = np.arange(0, len(signal)) / fs
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
        plt.title("Time-Domain Signal", fontsize=10)
        plt.grid(False)
        plt.subplot(7, 1, 2)
        plt.plot(t1, er)
        frame2 = plt.gca()
        for xlabel_i in frame2.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
        plt.title("Relative Intensity Index", fontsize=10)
        d_intensity = []
        I_seg1 = []
        I_seg1_e = []
        for s1 in floats1:
            I_seg1.append((s1 / 10000000) * fs)
        for s2 in floats1_e:
            I_seg1_e.append((s2 / 10000000) * fs)
        floatss_I = [int(x) for x in I_seg1]
        floatss_E = [int(x) for x in I_seg1_e]
        for i1, j1, k1 in zip_longest(floatss_I, floatss_E, syllable, fillvalue=None):
            s1 = er[i1:j1]

            splitted = [list(j) for i, j in groupby(s1)]
            k = []
            for idx, row in enumerate(splitted):
                if all(n == n for n in row):
                    if row[0] >= 0 and row[0] < 0.1:
                        k.append(row[0])


                    elif row[0] >= 0.1 and row[0] < 0.4:

                        k.append(row[0])

                    elif row[0] >= 0.4 and row[0] < 0.7:
                        k.append(row[0])

                    elif row[0] >= 0.7:
                        k.append(row[0])

            st = i1 / fs
            et = j1 / fs
            min_energy = min(k)
            max_energy = max(k)
            Ave_energy = mean(k)
            if max_energy >= 0.9:
                x = i1 / fs
                y = max_energy
                plt.text(x, y, '5', {'color': 'r', 'fontsize': 15})
            elif max_energy < 0.9 and max_energy >= 0.55:
                x = i1 / fs
                y = max_energy
                plt.text(x, y, '4', {'color': 'r', 'fontsize': 15})
            elif max_energy < 0.55 and max_energy >= 0.4:
                x = i1 / fs
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
        floats_intensity, floats_e_intensity, mi_e, ma_e, av_e = [eval(x) for x in starttimes_intensity], [eval(x) for x in
                                                                                                           endtimes_intensity], [
                                                                     eval(x) for x in Min_Energy], [eval(x) for x in
                                                                                                    Max_Energy], [eval(x)
                                                                                                                  for x in
                                                                                                                  Average_Energy]
        d_rel_intensity = []
        for n5, v5, p5, q5, r5, t5 in zip(floats_intensity, floats_e_intensity, syllable, mi_e, ma_e, av_e):
            d_rel_intensity.append("{} {} {} {} {} {}".format(n5, v5, p5, q5, r5, t5))
        intensity_table = open("data/intensity_t.txt", "w")
        for lines in d_rel_intensity:
            intensity_table.write(lines)
            intensity_table.write("\n")
        intensity_table.close()
        filename_intensity = "data/intensity_t.txt"
        ddd_intensity = pd.read_csv(filename_intensity, header=None, sep="\s{1,}", engine='python')
        ddd_intensity.columns = ['Starttime', 'Endtime', 'Syllables', 'Min_Energy', 'Max_Energy', 'Average_Energy']
        print(ddd_intensity)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        out_intensity_index = file + ".intensity"
        out_intensity_index1=open('../speech_new/web/static/' + out_intensity_index, "w")
        for lines in str(ddd_intensity):
            out_intensity_index1.write(lines)
        out_intensity_index1.close()

        plt.grid(False)
        plt.subplot(7, 1, 3)
        plt.plot(snd.xs(), snd.values.T)
        frame3 = plt.gca()
        for xlabel_i in frame3.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("Amplitude", fontsize=8)
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
        plt.title("Break Indices", fontsize=9)
        splitted = [list(g) for i, g in itertools.groupby(vad, lambda x: x == True)]
        indices = [i for i, x in enumerate(vad) if not x]

        gg = [list(group) for group in mit.consecutive_groups(indices)]
        d_break = []
        for idx, row in enumerate(splitted):
            if all(n == False for n in row):
                if len(row) >= 10:
                    for idx1, row1 in enumerate(gg):
                        if len(row) == len(row1):
                            t = row1[0]
                            x = time2[t]
                            st = x
                            t1 = row1[len(row1) - 1]
                            et = time2[t1]
                            y = 0.25
                            plt.text(x, y, '3', {'color': 'r', 'fontsize': 15})
                            d_break.append([st, et, "LSIL"])
                elif len(row) >= 3:
                    for idx1, row1 in enumerate(gg):
                        if len(row) == len(row1):
                            t = row1[0]
                            x = time2[t]
                            st = x
                            t1 = row1[len(row1) - 1]
                            et = time2[t1]
                            y = 0.25
                            plt.text(x, y, '2', {'color': 'r', 'fontsize': 15})
                            d_break.append([st, et, "MSIL"])
                elif len(row) < 3:
                    for idx1, row1 in enumerate(gg):
                        if len(row) == len(row1):
                            t = row1[0]
                            x = time2[t]
                            st = x
                            t1 = row1[len(row1) - 1]
                            et = time2[t1]
                            y = 0.25
                            plt.text(x, y, '1', {'color': 'r', 'fontsize': 15})
                            d_break.append([st, et, "SSIL"])
        test_list = np.array(d_break)

        res_duplicate = list(set(map(lambda i: tuple(sorted(i)), test_list)))
        dd_break = sorted(res_duplicate, key=lambda x: x[1], reverse=False)
        dd_break1 = pd.DataFrame(dd_break)
        dd_break1.columns = ['Starttime', 'Endtime', 'Break_Indices']
        print(dd_break1)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        out_break = file + ".b"
        out_break1=open('../speech_new/web/static/' + out_break, "w")
        for lines in str(dd_break1):
            out_break1.write(lines)
        out_break1.close()
        plt.subplot(7, 1, 5)
        pitch_values = pitchEn.selected_array['frequency']
        plt.plot(pitchEn.xs(), pitch_values, '.', markersize=2.5)
        plt.ylim(0, pitchEn.ceiling)
        frame3 = plt.gca()
        for xlabel_i in frame3.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        plt.ylabel("freq[Hz]", fontsize=8)
        plt.title("Pitch Contour", fontsize=10)
        plt.grid(True)
        d_tobi = []
        d1_tobi=[]
        p_seg1 = []
        for s1 in floats1:
            p_seg1.append((s1 / 10000000))
        p_seg1_e = []
        for s1 in floats1_e:
            p_seg1_e.append((s1 / 10000000))


        dynamic_ranges1 = []
        dynamic_ranges2 = []
        k1 = 1
        for i, j, k in zip_longest(p_seg1, p_seg1_e, syllable, fillvalue=None):
            start_index = np.argmax(time_points >= i)
            end_index = np.argmax(time_points >= j)
            spliced_pitch_values = pitch_values[start_index:end_index]
            spliced_time_points = time_points[start_index:end_index]
            res = spliced_pitch_values[spliced_pitch_values != 0]
            snd1 = snd.extract_part(from_time=i, to_time=j)
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

            len_f = len(l)
            if len_f > 1:
                xin = np.arange(0, len_f)
                yin = l
            elif len_f == 1 or len_f < 1:
                yin = 0
                xin = 0
                pf = 0
                continue

            pf = np.polyfit(xin, yin, 3)
            yin_smooth = np.polyval(pf, xin)
            dy_r2 = max(yin_smooth) - min(yin_smooth)
            dynamic_ranges2.append(dy_r2)
            with open(file + 'dynamic2.txt', 'w') as doc2:
                for item in dynamic_ranges2:
                    doc2.write(str(item) + '\n')
            if dy_r2 < 10:
                x1 = j
                y1 = pitchEn.ceiling / 4
                st1 = i
                et1 = j
                d_tobi.append([st1, et1, 'flat'])
                plt.text(x1,y1,'flat',{'color':'r','fontsize':7})
                print("flat")

            else:
                for ss in range(0, len(yin_smooth) - 1):
                    ff.append(yin_smooth[ss + 1] - yin_smooth[ss])
                    ff1 = abs(max(yin_smooth) - min(yin_smooth))
                splitted = [list(g) for i, g in itertools.groupby(ff, lambda xs: xs < 0)]
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

                for idx, row in enumerate(splitted):

                    if all(n > 0 for n in row):
                        if idx == 0:
                            index.append('p1')
                            len_f.append(len(row))
                        elif idx == 1:
                            index.append("p2")
                            len_f.append(len(row))
                        elif idx == 2:
                            index.append("p3")
                            len_f.append(len(row))
                    else:
                        if idx == 0:
                            index.append("q1")
                            len_f.append(len(row))
                        elif idx == 1:
                            index.append("q2")
                            len_f.append(len(row))
                        elif idx == 2:
                            index.append("q3")
                            len_f.append(len(row))
                a = np.concatenate((index, len_f))
                l_a = len(a)
                if l_a == 2:

                    if a[0] == 'p1' and 10 <= ff1 < 60:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        t1 = np.arange(0, len(yin_smooth))
                        d_tobi.append([st1, et1, 'S_H'])
                        plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})


                    elif a[0] == 'p1' and 60 <= ff1 < 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        t1 = np.arange(0, len(yin_smooth))

                        d_tobi.append([st1, et1, 'M_H'])
                        plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                    elif a[0] == 'p1' and ff1 >= 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        t1 = np.arange(0, len(yin_smooth))

                        d_tobi.append([st1, et1, 'B_H'])
                        plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})

                    elif a[0] == 'q1' and 10 <= ff1 < 60:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j

                        d_tobi.append([st1, et1, 'S_L'])
                        plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                    elif a[0] == 'q1' and 60 <= ff1 < 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi.append([st1, et1, 'M_L'])
                        plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                    elif a[0] == 'q1' and ff1 >= 100:
                        x1 = j
                        y1 = pitchEn.ceiling / 4
                        st1 = i
                        et1 = j
                        d_tobi.append([st1, et1, 'B_L'])
                        plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})

                elif l_a == 4:
                    if a[0] == 'p1' and a[1] == 'q2':
                        if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_L'])
                                plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})

                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_hat'])
                                    plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_HLL'])
                                    plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})

                        elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_L'])
                                plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_hat'])
                                    plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_HLL'])
                                    plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})

                        elif int(a[2]) < int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_L'])
                                plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_hat'])
                                    plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_HLL'])
                                    plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})

                            k1 = k1 + 1

                        elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_H'])
                                plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_hat'])
                                    plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_HHL'])
                                    plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})


                        elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_H'])
                                plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_hat'])
                                    plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_HHL'])
                                    plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_H'])
                                plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_hat'])
                                    plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_HHL'])
                                    plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})

                        elif int(a[2]) == int(a[3]):
                            if 10 <= ff1 < 60:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_hat'])
                                plt.text(x1,y1,'S_hat',{'color':'r','fontsize':7})
                            elif 60 <= ff1 < 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_hat'])
                                plt.text(x1,y1,'M_hat',{'color':'r','fontsize':7})
                            elif ff1 >= 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_hat'])
                                plt.text(x1,y1,'B_hat',{'color':'r','fontsize':7})

                    elif a[0] == 'q1' and a[1] == 'p2':
                        if int(a[2]) < int(a[3]) and 10 <= ff1 < 60:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_H'])
                                plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_bucket'])
                                    plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_LHH'])
                                    plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})


                        elif int(a[2]) < int(a[3]) and 60 <= ff1 < 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_H'])
                                plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_bucket'])
                                    plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_LHH'])
                                    plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})

                        elif int(a[2]) < int(a[3]) and ff1 >= 100:
                            if int(a[2]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_H'])
                                plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_bucket'])
                                    plt.text(x1,y1,'B_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_LHH'])
                                    plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and 10 <= ff1 < 60:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_L'])
                                plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_bucket'])
                                    plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'S_LLH'])
                                    plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and 60 <= ff1 < 100:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_L'])
                                plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_bucket'])
                                    plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'M_LLH'])
                                    plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})

                        elif int(a[2]) > int(a[3]) and ff1 > 100:
                            if int(a[3]) <= 2:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_L'])
                                plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                            else:
                                thresh_p = abs(yin_smooth[(f_p_i)] - yin_smooth[f_n_i])
                                thresh_n = abs((yin_smooth[(f_n_i) + 1]) - (yin_smooth[(l_n_i) + 1]))
                                if thresh_p == thresh_n:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_bucket'])
                                    plt.text(x1,y1,'B_bucket',{'color':'r','fontsize':7})
                                else:
                                    st1 = i
                                    et1 = j
                                    x1 = j
                                    y1 = pitchEn.ceiling / 4
                                    d_tobi.append([st1, et1, 'B_LLH'])
                                    plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})



                        elif int(a[2]) == int(a[3]):
                            if 10 <= ff1 < 60:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'S_bucket'])
                                plt.text(x1,y1,'S_bucket',{'color':'r','fontsize':7})
                                print("bucket")
                            elif 60 <= ff1 < 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'M_bucket'])
                                plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                print("bucket")
                            elif ff1 >= 100:
                                st1 = i
                                et1 = j
                                x1 = j
                                y1 = pitchEn.ceiling / 4
                                d_tobi.append([st1, et1, 'B_bucket'])
                                plt.text(x1,y1,'M_bucket',{'color':'r','fontsize':7})
                                print("bucket")


                elif l_a == 6:
                    if a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 10 <= ff1 < 60:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LHH'])
                            plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LLH'])
                            plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HLL'])
                            plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HHL'])
                            plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_L'])
                            plt.text(x1,y1,'S_L',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HLH'])
                            plt.text(x1,y1,'M_HLH',{'color':'r','fontsize':7})


                    elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and 60 <= ff1 < 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LHH'])
                            plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LLH'])
                            plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HLL'])
                            plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HHL'])
                            plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_L'])
                            plt.text(x1,y1,'M_L',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HLH'])
                            plt.text(x1,y1,'M_HLH',{'color':'r','fontsize':7})


                    elif a[0] == 'p1' and a[1] == 'q2' and a[2] == 'p3' and ff1 >= 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LHH'])
                            plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LLH'])
                            plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HLL'])
                            plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HHL'])
                            plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_L'])
                            plt.text(x1,y1,'B_L',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HLH'])
                            plt.text(x1,y1,'B_HLH',{'color':'r','fontsize':7})


                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 10 <= ff1 < 60:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HLL'])
                            plt.text(x1,y1,'S_HLL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_HHL'])
                            plt.text(x1,y1,'S_HHL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LHH'])
                            plt.text(x1,y1,'S_LHH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LLH'])
                            plt.text(x1,y1,'S_LLH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_H'])
                            plt.text(x1,y1,'S_H',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'S_LHL'])
                            plt.text(x1,y1,'S_LHL',{'color':'r','fontsize':7})


                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and 60 <= ff1 < 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HLL'])
                            plt.text(x1,y1,'M_HLL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_HHL'])
                            plt.text(x1,y1,'M_HHL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LHH'])
                            plt.text(x1,y1,'M_LHH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LLH'])
                            plt.text(x1,y1,'M_LLH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_H'])
                            plt.text(x1,y1,'M_H',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'M_LHL'])
                            plt.text(x1,y1,'M_LHL',{'color':'r','fontsize':7})

                    elif a[0] == 'q1' and a[1] == 'p2' and a[2] == 'q3' and ff1 >= 100:
                        if int(a[3]) <= 2 and int(a[4]) < int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HLL'])
                            plt.text(x1,y1,'B_HLL',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[4]) > int(a[5]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_HHL'])
                            plt.text(x1,y1,'B_HHL',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) < int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LHH'])
                            plt.text(x1,y1,'B_LHH',{'color':'r','fontsize':7})
                        elif int(a[5]) <= 2 and int(a[3]) > int(a[4]):
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LLH'])
                            plt.text(x1,y1,'B_LLH',{'color':'r','fontsize':7})
                        elif int(a[3]) <= 2 and int(a[5]) <= 2:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_H'])
                            plt.text(x1,y1,'B_H',{'color':'r','fontsize':7})
                        else:
                            st1 = i
                            et1 = j
                            x1 = j
                            y1 = pitchEn.ceiling / 4
                            d_tobi.append([st1, et1, 'B_LHL'])
                            plt.text(x1,y1,'B_LHL',{'color':'r','fontsize':7})


        print(d_tobi)
        dd_tobi = pd.DataFrame(d_tobi)
        dd_tobi.columns = ['Starttime', 'Endtime', 'Labels']
        print(dd_tobi)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        out_pitch = file + ".pitch"
        out_pitch1=open('../speech_new/web/static/' + out_pitch, "w")
        for lines in str(dd_tobi):
            out_pitch1.write(lines)
        out_pitch1.close()

        new_graph_name = "graph" +str(time.time()) +".jpg"
        for filename in os.listdir('../speech_new/web/static'):
            if filename.startswith('graph_'):
                os.remove('../speech_new/web/static/'+filename)
        plt.savefig('../speech_new/web/static/' + new_graph_name)
        plt.show()
        os.system("rm -f *.wav *.txt *.p *.phn p phn")


    return render_template('output.html', name=file, graph=new_graph_name,data = out_phoneme, data1=out_syllable, data2=out_word, data3=out_intensity_index, data4=out_break, data5=out_pitch)


# region Run the Flask's built-in development server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=True)