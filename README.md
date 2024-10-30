# Prosody Modeling

This repository contains the source files of an automatic prosody annotation tool, that has been designed by the team at Speech Lab, Shiv Nadar University Chennai as a part of the Prosody Modeling module of the Bashini: NLTM Speech Technologies in Indian Languages Project, funded by the Ministry of Electronics and Information Technology, Govt. of India.. This tool has been designed to provide rich prosodic annotations for Indian languages. The annotator will take a speech signal and the corresponding orthographic transcription as inputs and provide the phoneme, syllable, and word boundaries, along with the pitch contour labels and the intensity index at the syllable level, and the break indices. While the phoneme boundaries are estimated using hidden Markov models (HMMs) trained on 5 hours of data (from a male and a female speaker) in each language, the rest are derived based on rules formulated after extensive analyses. The tool is currently designed for English, Tamil, and Hindi, but can be extended to other languages by including the appropriate letter-to-sound rules and training phoneme HMMs.

The data used to train the models can be accessed [here](https://www.iitm.ac.in/donlab/indictts)

The performance of the various modules of the tool is evaluated by comparing annotations derived from the tool for 50 audio files (per language) with the corresponding manual annotations. For most phoneme segments, the segmentation error is under 10 ms. The overall accuracy for break indices across three languages-Tamil, Hindi, and Indian English is 95%, while the pitch contour model achieves an accuracy of 99% relative to manual annotations.

## Tool Demo Screenshot

![image](https://github.com/speech-lab-snuchennai/Prosody_Modelling/assets/166628077/49e824d9-04de-4795-94e0-d29f8b617956)

---
## Installation

To use this project, you will need to clone the repository and install the required dependencies as follows:

### Clone the Repository

```bash
git clone https://github.com/speech-lab-snuchennai/Prosody_Modelling
cd project-name
```
### Installing Dependencies

```bash
pip install -r requirements.txt
```

## Usage

- Update the "filename" (with the path to the audio file to be annotated) and "language" fields in main.py. Provide the corresponding input text in the te.txt file. Then run the man.py file. This should display an image as shown above and generate label files in the directory containing your audio file.

```bash
python main.py
```

## Training a New Model for Segmentation

The HMM toolkit HTK has been employed to train the phoneme HMMs. In order to train models for new English, Tamil, or Hindi data, download and install HTK as described [here](https://speech.zone/forums/topic/how-to-compile-htk/).

After installing HTK, create a wav folder (containing the audio files in wav format) and a lab folder (containing a set of initial lab files, which could be generated using main.py). Then run the run.sh file to train the phoneme models.

```bash
./run.sh
```


