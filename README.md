# Prosody Modelling

The Automatic ToBI annotation tool has been designed to provide rich prosodic annotation for Indian languages. Our proposed automatic ToBI annotator will take speech and corresponding orthographic transcription alone. In addition to tones and break indices, we propose to output boundaries of phones, syllables, words, phrases, and intensity indices. The pitch contour is calculated using a short-time autocorrelation-based approach. 
The contour is then labelled based on rising and falling edges along the contour Currently, the tools work for English, Tamil, and Hindi.

## Tool Demo Screenshot

![image](https://github.com/speech-lab-snuchennai/Prosody_Modelling/assets/166628077/49e824d9-04de-4795-94e0-d29f8b617956)

---
## Installation

To use this project, you need to clone the repository and install the required dependencies.

### Clone the Repository

```bash
git clone [https://github.com/your-username/project-name.git](https://github.com/speech-lab-snuchennai/Prosody_Modelling)
cd project-name
```
### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py

```
