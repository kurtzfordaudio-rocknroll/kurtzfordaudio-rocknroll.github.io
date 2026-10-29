---
layout: default
title: Separating Instruments and Vocals from Your Music
permalink: /en/
lang: en
---

[한국어]({{ '/' | relative_url }}) | **English**

# Separating Instruments and Vocals from Your Own Music Files

**Convert an HTDemucs 6-Stem checkpoint into a Core ML FP16 model and use it with your own music files**

**Estimated time:** Approximately 10 minutes  
**Final result:** A Core ML package for music source separation, `HTDemucs_CoreML_FP16.mlpackage`

Can we separate the individual instruments from a recording that contains vocals, drums, bass, guitar, piano, and other sounds mixed together?

AI researcher **Alexandre Défossez and collaborators** developed an AI-based music source separation model called **Demucs** to address this challenge. They trained the model using **MUSDB HQ and additional music data** and released some of the code and pretrained model checkpoints so that others could build on their research. The later **Hybrid Transformer Demucs (HTDemucs)** combines waveform and spectrogram processing and can separate a mix into six stems: drums, bass, vocals, guitar, piano, and other sounds.

A **checkpoint** is a file containing the weights of a trained neural network. Using a checkpoint allows us to skip the lengthy process of collecting data and training a model from scratch. But can the model separate instruments from music it has never heard before?

In this tutorial, we will download a checkpoint, convert it to Core ML on a Mac, and finally separate a test recording into six stems and listen to the results.

## Table of Contents

- [What Will You Make?](#what-will-you-make)
- [Requirements](#requirements)
- [Step 1. Open Terminal and Create a Working Folder](#step-1-open-terminal-and-create-a-working-folder)
- [Step 2. Download the Checkpoint](#step-2-download-the-checkpoint)
- [Step 3. Prepare the Necessary Tools on Your Mac](#step-3-prepare-the-necessary-tools-on-your-mac)
- [Step 4. Download the Core ML Converter](#step-4-download-the-core-ml-converter)
- [Step 5. Install Python and Create a Dedicated Environment](#step-5-install-python-and-create-a-dedicated-environment)
- [Step 6. Place the Checkpoint in the PyTorch Cache](#step-6-place-the-checkpoint-in-the-pytorch-cache)
- [Step 7. Adapt the Converter for Six Stems](#step-7-adapt-the-converter-for-six-stems)
- [Step 8. Build the FP16 Core ML Model](#step-8-build-the-fp16-core-ml-model)
- [Step 9. Check the Model Inputs and Outputs](#step-9-check-the-model-inputs-and-outputs)
- [Step 10. Separate the Audio and Listen](#step-10-separate-the-audio-and-listen)
- [Troubleshooting FAQ](#troubleshooting-faq)
- [Wrapping Up](#wrapping-up)
- [References and Sources](#references-and-sources)

## What Will You Make?

We will create a Core ML model package named **`HTDemucs_CoreML_FP16.mlpackage`**. This model processes 10-second segments of 44.1 kHz stereo audio and separates them into **drums, bass, vocals, guitar, piano, and other sounds**. In the final exercise, we will feed it the supplied [`test_10s.wav`]({{ '/resources/test_10s.wav' | relative_url }}) and listen to six resulting **WAV files**.

## Requirements

- **A Mac running macOS** (Apple Silicon recommended)
- **Internet connection**
- **Terminal app** (included with macOS)
- **Web browser** (to download the checkpoint)
- **Test audio `test_10s.wav`** (provided with this tutorial: [download the WAV file]({{ '/resources/test_10s.wav' | relative_url }}))

---

## Step 1. Open Terminal and Create a Working Folder

### 1-1. What Is Terminal?

**Terminal** is an app that lets you send text commands to your Mac. It is the modern software equivalent of a physical computer terminal. Instead of using the mouse to create folders or navigate between them, you can type short commands to tell your computer what to do.

<img src="https://help.apple.com/assets/67DB80CC8ED20D93270BDBE5/67DB80CD888B30021203ACDC/en_US/4e8328e6927032645dfb715a76f5a527.png" width="100" alt="Apple Terminal app icon with a prompt symbol on a dark background">

To open Terminal:

1. Open Finder.
2. Navigate to **Applications** → **Utilities**.
3. Double-click **Terminal**, which looks like the icon above.

Alternatively, press `⌘ Command + Space`, search for `Terminal`, and press Return.

When Terminal opens, you will see a **prompt** similar to this one. You can type commands after the prompt.

```text
username@Mac-name ~ %
```

### 1-2. Create a Working Folder

Copy the following command, paste it into Terminal, and press Return. You can also type it manually.

```
mkdir ~/htdemucs
```

`mkdir` means **make a new directory**. The `~` symbol represents your home folder, so this command creates a folder called `htdemucs` inside your home folder. It is normal for the command to finish without displaying a message. If the folder already exists, you might see `File exists`; in that case, simply use the existing folder.

Move into the folder you just created.

```
cd ~/htdemucs
```

`cd` means **change directory**. Check your current location:

```
pwd
```

If the displayed path ends with `/htdemucs`, you are in the correct place. Now list the contents of the folder:

```
ls
```

`ls` **lists files in the current folder**. Because you just created the folder, the output might be empty.

---

## Step 2. Download the Checkpoint

The checkpoint file is named:

```text
5c90dfd2-34c22ccb.th
```

Download it from one of these locations:

- **Official Meta server:** [Download 5c90dfd2-34c22ccb.th](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th)
- **Hugging Face - lainlives/audio-separator-models:** [File page](https://huggingface.co/lainlives/audio-separator-models/blob/main/5c90dfd2-34c22ccb.th)
- **Hugging Face - lainlives/audio-separator-models-old:** [File page](https://huggingface.co/lainlives/audio-separator-models-old/blob/a78be39e563e8337bc7f2e5241d8063256d86167/5c90dfd2-34c22ccb.th)

Open the `Downloads` folder in Finder and make sure `5c90dfd2-34c22ccb.th` is there. The following steps assume that the file is located at `~/Downloads/5c90dfd2-34c22ccb.th`.

---

## Step 3. Prepare the Necessary Tools on Your Mac

We need a few programs. Skip any installation steps for tools already available on your Mac.

### 3-1. Install Apple Command Line Tools

**Apple Command Line Tools** provide basic utilities used for development on a Mac. Run the following command in Terminal:

```
xcode-select -p
```

If it prints `/Library/Developer/CommandLineTools` or another installation path, the tools are already installed.

If you see an error instead, run:

```
xcode-select --install
```

When the installation dialog appears, click **Install** and follow the instructions. When it finishes, run `xcode-select -p` again to confirm that a path is displayed.

### 3-2. Install Homebrew

**Homebrew** is a tool that makes it easy to install programs from Terminal.

1. Open the [official Homebrew website](https://brew.sh/) in a browser.
2. Find the **Install Homebrew** section near the top of the page.
3. **Copy the installation command from the official website**, paste it into Terminal, and press Return.
4. Follow the prompts if you are asked for a password or confirmation. It is normal for password characters not to appear as you type.

The installation command on the official website looks like this. **This is only an example; copy the current command from the Homebrew website instead.**

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

When installation finishes, Homebrew displays **Next steps**. They might look like this:

```text
==> Next steps:
- Run these two commands in your terminal to add Homebrew to your PATH:
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/username/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
```

Copy the two commands shown **in your own Terminal output**, starting with `(echo;` and `eval`, respectively. Run them in that order. Do not copy the example above because the username and path may differ.

Then check that Homebrew works:

```
brew --version
```

If you see something like `Homebrew 4.4.x`, the installation and configuration are complete.

### 3-3. Install Git

**Git** is a tool that lets us download files from public code repositories. First, check whether it is already installed:

```
git --version
```

If a version number appears, you do not need to install it. Otherwise run:

```
brew install git
```

Run `git --version` again to confirm.

---

## Step 4. Download the Core ML Converter

We will use the [dexxdean/htdemucs-coreml](https://github.com/dexxdean/htdemucs-coreml) project to convert HTDemucs to Core ML. This converter was originally configured for four stems. We will make a few changes so it can handle six.

Run these four commands in order. You can also paste all four lines at once. They move to `~/htdemucs`, clone the repository, enter the downloaded folder, and check that the necessary files are present.

```
cd ~/htdemucs
git clone https://github.com/dexxdean/htdemucs-coreml.git
cd htdemucs-coreml
ls convert.py requirements.txt
```

If both file names appear, this step is complete.

---

## Step 5. Install Python and Create a Dedicated Environment

**Python** is the programming language used to run the converter. It might already be installed on your Mac, so we will check first.

### 5-1. Check Whether Python Is Installed

We will use **Python 3.11**. Check whether it is installed:

```
python3.11 --version
```

If you see `Python 3.11.x`, proceed to Step 5-2. If you see `command not found` or a similar error, install it:

```
brew install python@3.11
```

Then check again:

```
python3.11 --version
```

### 5-2. Create a Working Environment

This tutorial requires several Python packages. We will create a separate **virtual environment**, which keeps these packages from interfering with other Python projects.

Enter the project folder, create the environment, and activate it:

```
cd ~/htdemucs/htdemucs-coreml
python3.11 -m venv .venv
source .venv/bin/activate
```

If `(.venv)` appears at the beginning of the Terminal prompt, you have **successfully activated the environment for this project**.

```text
(.venv) username@Mac-name ~/htdemucs/htdemucs-coreml %
```

### 5-3. Install the Required Python Packages

Now install the packages required by the converter. Run these **one line at a time**:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install scipy
```

The first command upgrades `pip`, Python's package installer. The second installs dependencies required by the converter. The third installs `scipy`, which we will use in the final audio separation exercise. Installation might take some time.

When finished, check the environment:

```
python -m pip check
python -c 'import torch, demucs, coremltools, numpy, einops; print("Python packages: OK")'
```

If you see `Python packages: OK`, the required packages can be imported. If you encounter an error, see the **Troubleshooting FAQ** below.

**What if you close Terminal and start again later?** You do not need to reinstall anything. Open Terminal and run these two commands before continuing:

```
cd ~/htdemucs/htdemucs-coreml
source .venv/bin/activate
```

---

## Step 6. Place the Checkpoint in the PyTorch Cache

Paste the entire block below into Terminal and press Return. This copies the downloaded checkpoint (`5c90dfd2-34c22ccb.th`) from your Downloads folder to the PyTorch cache.

```
python - <<'PY'
from pathlib import Path
import shutil
import torch

source = Path.home() / 'Downloads' / '5c90dfd2-34c22ccb.th'
target_dir = Path(torch.hub.get_dir()) / 'checkpoints'
if not source.is_file():
    raise SystemExit(f'ERROR: checkpoint not found: {source}')
target_dir.mkdir(parents=True, exist_ok=True)
target = target_dir / source.name
shutil.copy2(source, target)
print('Copied checkpoint to:', target)
PY
```

If the last line of output begins with `Copied checkpoint to:` followed by a path, the copy succeeded.

---

## Step 7. Adapt the Converter for Six Stems

This is the most important step in the tutorial. The converter we downloaded uses four-stem settings by default. We need to change them as follows:

| Setting | Original value | New value |
| --- | --- | --- |
| Model name | `htdemucs` | `htdemucs_6s` |
| Number of stems | `4` | `6` |
| Output reordering | `[3, 0, 1, 2]` | `[0, 1, 3, 5, 4, 2]` |
| Output names | vocals, drums, bass, other | drums, bass, vocals, guitar, piano, other |
| Output shape | `(1, 4, 2, …)` | `(1, 6, 2, …)` |

The original Demucs six-source model produces outputs in this order: **drums, bass, other, vocals, piano, guitar**. We will reorder the outputs to **drums, bass, vocals, guitar, piano, other**.

### 7-1. Download the Helper Script

Download the [`prepare_six_stem.py`]({{ '/resources/prepare_six_stem.py' | relative_url }}) file supplied with this tutorial. We assume your browser saves it to the `Downloads` folder.

Run these commands one line at a time:

```
cd ~/htdemucs/htdemucs-coreml
cp ~/Downloads/prepare_six_stem.py .
ls convert.py prepare_six_stem.py
```

The first command is a precaution in case you moved to another directory. You can skip it if you are already in the correct folder.

The `cp` command copies the downloaded file to the current folder, and `ls` verifies that the copy worked. If both files are listed, run the helper script:

```
python prepare_six_stem.py
```

The script backs up the original `convert.py` as `convert.py.original-4stem` and modifies several parts of `convert.py`. It is designed to print details if an error occurs.

> If the helper script cannot find the required sections of the original code, it stops rather than making incomplete changes. See the FAQ below if this happens.

**Expected output:**

```text
SUCCESS: convert.py adapted for six stems.
Backup: convert.py.original-4stem
Model: htdemucs_6s
Output: drums, bass, vocals, guitar, piano, other
```

### 7-2. Verify the Changes

Copy the following block into Terminal and press Return:

```
python -m py_compile convert.py
python - <<'PY'
from pathlib import Path

s = Path('convert.py').read_text()
for line in s.splitlines():
    if line.startswith(('MODEL_NAME =', 'NUM_SOURCES =', 'SOURCE_REORDER =', 'SOURCE_NAMES =')):
        print(line)
PY
```

The output should look like this:

```python
MODEL_NAME = "htdemucs_6s"
NUM_SOURCES = 6
SOURCE_REORDER = [0, 1, 3, 5, 4, 2]
SOURCE_NAMES = ["drums", "bass", "vocals", "guitar", "piano", "other"]
```

### Why Modify the Converter?

Some of the operations in the original HTDemucs architecture are difficult for `coremltools` to convert directly. The publicly available converter represents complex-number operations using real-valued computations and implements attention and overlap-add operations in a form compatible with Core ML conversion. **We are not retraining the weights**. The goal is to convert the existing neural network into Core ML format.

---

## Step 8. Build the FP16 Core ML Model

Move to the `htdemucs-coreml` folder and run `convert.py`. The Terminal prompt should begin with `(.venv)` to indicate that the virtual environment is active.

```
cd ~/htdemucs/htdemucs-coreml
python convert.py --fp16 --compute-units cpu_and_gpu
```

During conversion, you should see steps similar to these:

```text
[1/5] Loading Demucs 'htdemucs_6s' ...
[2/5] Building real-valued wrapper ...
[3/5] PyTorch forward pass ...
[4/5] torch.jit.trace ...
[5/5] Core ML conversion ...
```

Converting a large neural network can temporarily use considerable memory. **Make sure the conversion and validation complete without errors.**

Verify that the model package was created:

```
ls -ld HTDemucs_CoreML_FP16.mlpackage
```

---

## Step 9. Check the Model Inputs and Outputs

Use `coremltools` to inspect the model package you just created:

```
python - <<'PY'
import coremltools as ct

path = 'HTDemucs_CoreML_FP16.mlpackage'
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
print('Inputs:')
for item in model.get_spec().description.input:
    print(' ', item.name, list(item.type.multiArrayType.shape))
print('Outputs:')
for item in model.get_spec().description.output:
    print(' ', item.name, list(item.type.multiArrayType.shape))
PY
```

The expected values are:

```text
Inputs:
  audio [1, 2, 441000]
Outputs:
  sources [1, 6, 2, 441000]
```

**Producing six output arrays** is not the same as **guaranteeing accurate audio separation**. The converter attempts to compare PyTorch and Core ML results after conversion, so be sure to check its validation logs as well.

Follow Step 10 to evaluate the separation quality and source order by listening to the actual outputs. For example, make sure `drums.wav` actually contains drums, and listen for unwanted vocals in `guitar.wav`. Note that Meta describes the six-source model's piano separation performance as limited.

---

## Step 10. Separate the Audio and Listen

Now we will test how well the Core ML model separates instruments in a real recording.

Download the 10-second sample [`test_10s.wav`]({{ '/resources/test_10s.wav' | relative_url }}).

1. In this GitHub repository, select **`test_10s.wav`** and click the **Download raw file** icon.
2. Confirm that the file is saved in your Mac's `Downloads` folder.
3. Run the following commands one line at a time to enter the converter directory and copy the sample audio file:

```
cd ~/htdemucs/htdemucs-coreml
source .venv/bin/activate
cp ~/Downloads/test_10s.wav .
ls -lh test_10s.wav HTDemucs_CoreML_FP16.mlpackage
```

Copy the entire code block below into Terminal and press Return. The model will separate the input audio into six WAV files.

```
python - <<'PY'
from pathlib import Path
import numpy as np
import coremltools as ct
from scipy.io import wavfile

rate, data = wavfile.read('test_10s.wav')
if rate != 44100 or data.ndim != 2 or data.shape[1] != 2:
    raise ValueError('A 44.1 kHz stereo WAV file is required.')

data = data.astype(np.float32)
segment_samples = 441000
if len(data) < segment_samples:
    data = np.pad(data, ((0, segment_samples - len(data)), (0, 0)))
else:
    data = data[:segment_samples]

audio = np.transpose(data, (1, 0))[None, :, :]
model = ct.models.MLModel(
    'HTDemucs_CoreML_FP16.mlpackage',
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
)
sources = np.asarray(model.predict({'audio': audio})['sources'])[0]

names = ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other']
if sources.shape != (6, 2, segment_samples):
    raise ValueError(f'Unexpected output shape: {sources.shape}')

output_dir = Path('separated_stems')
output_dir.mkdir(exist_ok=True)
for name, stem in zip(names, sources):
    file_path = output_dir / f'{name}.wav'
    wavfile.write(file_path, rate, stem.T.astype(np.float32))
    print('Created:', file_path)
PY
```

When processing finishes, open the output folder:

```
open separated_stems
```

You should find `drums.wav`, `bass.wav`, `vocals.wav`, `guitar.wav`, `piano.wav`, and `other.wav`. Double-click each file to listen and compare it with the original `test_10s.wav`.

---

## Troubleshooting FAQ

**Q. Terminal says `command not found: brew`.**  
A. Confirm that [Homebrew](https://brew.sh/) is installed. Apply the shell PATH instructions shown at the end of installation, then try again in a new Terminal window.

**Q. Terminal cannot find `python3.11`.**  
A. Check that you ran `brew install python@3.11`. Use `brew info python@3.11` to see where its executable is installed.

**Q. I get Python errors such as `No module named demucs`.**  
A. Activate the virtual environment again with `source .venv/bin/activate`, and check that `python -m pip install -r requirements.txt` completed successfully.

**Q. Only four stems are created.**  
A. Verify that `MODEL_NAME`, `NUM_SOURCES`, `SOURCE_REORDER`, and `SOURCE_NAMES` match the values shown in Step 7. Also make sure you are not accidentally using an older `.mlpackage`.

**Q. I created the Core ML package, but loading or prediction fails.**  
A. Check your macOS and `coremltools` versions, available disk space, memory, and the `CPU_AND_GPU` setting. Changing compute devices without testing is not recommended.

**Q. How do I exit the virtual environment?**  
A. Run `deactivate` in Terminal. To reactivate it later, go to the project folder and run `source .venv/bin/activate`.

---

## Wrapping Up

You have now used an HTDemucs checkpoint to create your own Core ML model and separate a real recording into six stems. Along the way, you converted an already-trained AI model into a Core ML package and applied it to new audio to inspect the results.

When you listen to the six WAV files, you might notice that the quality varies across instruments or that a little sound from other sources leaks into a stem. Still, extracting individual instruments from a single mixed music file is an interesting technology to explore.

Finally, **keep the `HTDemucs_CoreML_FP16.mlpackage` file that you created!** You can reuse it later without going through the conversion process again.

---

## References and Sources

- [Meta Demucs original repository](https://github.com/facebookresearch/demucs)
- [HTDemucs Core ML converter - dexxdean/htdemucs-coreml](https://github.com/dexxdean/htdemucs-coreml)
- [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)
- [Official Homebrew website](https://brew.sh/)
