---
layout: default
title: Music Stem Separation
lang: en
description: "Convert an HTDemucs six-stem checkpoint to a Core ML FP16 model on macOS. Step-by-step setup, conversion, verification, and audio stem separation."
author: K. Ford
---

**English** | [한국어]({{ "/kr.html" | relative_url }}) | [日本語]({{ "/jp.html" | relative_url }})

<p class="article-byline">K. Ford · Last updated: November 2025</p>

# Extract Individual Instruments and Vocals from Your Music Files

**Convert an HTDemucs 6-Stem checkpoint into a Core ML FP16 model and try it on your own music**

**Estimated time:** About 30 minutes  
**Final result:** A Core ML package for audio source separation, `HTDemucs_CoreML_FP16.mlpackage`, targeting macOS 13 or later

Can we separate the individual instruments and vocals from a song that mixes drums, bass, guitar, piano, vocals, and other sounds together?

AI researcher **Alexandre Défossez and colleagues** developed **Demucs**, a model for separating audio sources, to tackle this problem. They trained the model using **MUSDB HQ and other music datasets** and released code and pretrained model checkpoints for further research. The later **Hybrid Transformer Demucs (HTDemucs)** combines waveform and spectrogram processing to separate audio into six stems: drums, bass, vocals, guitar, piano, and other sounds.

In this tutorial, you will download a checkpoint yourself, convert it to Core ML on a Mac, then separate a test audio file into six stems and listen to the results.

<figure style="margin: 24px 0; text-align: center;">
  <img
    src="./images/manlistening.jpg"
    alt="A person concentrating while listening to music"
    width="500"
    height="500"
    style="width: 500px; height: 500px; object-fit: cover; object-position: center bottom; border-radius: 8px;"
    loading="lazy"
  />
  <figcaption style="margin-top: 8px; font-size: 14px; color: #666;">
    Photo credit —
    <a href="https://www.pexels.com/@wahlex-1869682520/" target="_blank" rel="noopener noreferrer">Wahlex</a>,
    pexels.com
  </figcaption>
</figure>

## Contents

- [What Will You Create?](#what-will-you-create)
- [Requirements](#requirements)
- [Step 1. Open Terminal and Create a Working Folder](#step-1-open-terminal-and-create-a-working-folder)
- [Step 2. Download the Checkpoint](#step-2-download-the-checkpoint)
- [Step 3. Install Apple Command Line Tools](#step-3-install-apple-command-line-tools)
- [Step 4. Download the Core ML Converter](#step-4-download-the-core-ml-converter)
- [Step 5. Install Python and Create a Dedicated Environment](#step-5-install-python-and-create-a-dedicated-environment)
- [Step 6. Copy the Checkpoint into the PyTorch Cache](#step-6-copy-the-checkpoint-into-the-pytorch-cache)
- [Step 7. Make a Few Changes to the Conversion Code](#step-7-make-a-few-changes-to-the-conversion-code)
- [Step 8. Create the FP16 Core ML Model](#step-8-create-the-fp16-core-ml-model)
- [Step 9. Check the Model Inputs and Outputs](#step-9-check-the-model-inputs-and-outputs)
- [Step 10. Separate and Listen to the Stems](#step-10-separate-and-listen-to-the-stems)
- [Troubleshooting FAQ](#troubleshooting-faq)
- [Conclusion](#conclusion)
- [References](#references)

## What Will You Create?

You will create a Core ML model package named **HTDemucs_CoreML_FP16.mlpackage**. It accepts 44.1 kHz stereo audio in 10-second segments and separates it into drums, bass, vocals, guitar, piano, and other sounds. In the final exercise, you will use the supplied test_10s.wav file to generate six WAV files and listen to each one. With additional code to process successive segments and join their outputs, the model can also separate music files longer than 10 seconds into continuous stem tracks.

## Requirements

- **A Mac running macOS 13 or later** (both Apple Silicon and Intel are supported)
- **An internet connection**
- **The Terminal app** (included with macOS)
- **A web browser** (for downloading the checkpoint and other files)
- **The sample audio `test_10s.wav`** (provided with this tutorial — [`test_10s.wav`](resources/test_10s.wav))

---

## Step 1. Open Terminal and Create a Working Folder

### 1-1. What Is Terminal?

**Terminal** lets you give your Mac instructions by typing commands. Instead of creating and navigating folders with the mouse, you can use short commands to tell the computer what to do.

<img src="https://help.apple.com/assets/67DB80CC8ED20D93270BDBE5/67DB80CD888B30021203ACDC/en_US/4e8328e6927032645dfb715a76f5a527.png" width="100" alt="Apple Terminal icon — white prompt symbol on a black background">

To open Terminal:

1. Open Finder.
2. Go to **Applications** → **Utilities**.
3. Double-click **Terminal**, which looks like the icon above.

Alternatively, press `⌘ Command + Space`, search for `Terminal`, and press Return.

After opening Terminal, you will see a **prompt** similar to the one below. The `~` indicates your Mac user's home folder. The prompt is showing that your **current working directory is your home folder**.

```text
username@Mac-name ~ %
```

You can type commands at the end of the prompt.

### 1-2. Create a Working Folder

Copy the following command into Terminal and press Return (Enter). You can also type it yourself.

```
mkdir ~/htdemucs
```

`mkdir` means **make a directory**. This command creates an `htdemucs` folder inside your home folder. Normally, no message appears after the command completes. If the folder already exists, you may see `File exists`; simply use the existing folder.

Move into the new folder:

```
cd ~/htdemucs
```

`cd` means **change directory**. Next, check your current location:

```
pwd
```

If the path ends in `/htdemucs`, the command worked. Finally, inspect the files in this folder:

```
ls
```

`ls` lists the files in the current folder. A newly created folder may be empty, so nothing may appear.

---

## Step 2. Download the Checkpoint

A **checkpoint** is a file containing the weights a neural network learned during training. You can think of these weights as a central part of the trained model: the checkpoint lets you use an existing model without gathering a large dataset and training it from scratch.

The file to download is `5c90dfd2-34c22ccb.th`. Open one of the links below to download it:

- **Meta server:** [Download 5c90dfd2-34c22ccb.th](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th)
- **Hugging Face — lainlives/audio-separator-models:** [File page](https://huggingface.co/lainlives/audio-separator-models/blob/main/5c90dfd2-34c22ccb.th)
- **Hugging Face — lainlives/audio-separator-models-old:** [File page](https://huggingface.co/lainlives/audio-separator-models-old/blob/a78be39e563e8337bc7f2e5241d8063256d86167/5c90dfd2-34c22ccb.th)

In Finder, open your `Downloads` folder and confirm that `5c90dfd2-34c22ccb.th` is there. The following steps assume it is saved as `~/Downloads/5c90dfd2-34c22ccb.th`.

---

## Step 3. Install Apple Command Line Tools

**Apple Command Line Tools** provide basic development commands for your Mac.

First, check whether they are already installed by running this command in Terminal:

```
xcode-select -p
```

If you see `/Library/Developer/CommandLineTools` or another installation path, they are already installed.

If the command instead displays an error, run:

```
xcode-select --install
```

When the installation window appears, click **Install** and follow the instructions. After installation, run `xcode-select -p` again to confirm the path.

Also check that **Git**, included with Apple Command Line Tools, is available:

```
git --version
```

If Git prints a version number, you are ready to continue.

---

## Step 4. Download the Core ML Converter

We will use the [dexxdean/htdemucs-coreml](https://github.com/dexxdean/htdemucs-coreml) project to convert HTDemucs into Core ML. This converter is originally configured for four stems. We will make a few changes to support six stems.

Run the following commands in order. You can paste all four lines into Terminal at once. They move into `~/htdemucs`, clone the Git repository, enter the downloaded project folder, and confirm that two required files are present.

```
cd ~/htdemucs
git clone https://github.com/dexxdean/htdemucs-coreml.git
cd htdemucs-coreml
ls convert.py requirements.txt
```

A successful result should look similar to this:

```
Cloning into 'htdemucs-coreml'...
remote: Enumerating objects: 12, done.
remote: Counting objects: 100% (12/12), done.
remote: Compressing objects: 100% (9/9), done.
remote: Total 12 (delta 1), reused 12 (delta 1), pack-reused 0 (from 0)
Receiving objects: 100% (12/12), 18.83 KiB | 9.42 MiB/s, done.
Resolving deltas: 100% (1/1), done.
convert.py       requirements.txt
```

---

## Step 5. Install Python and Create a Dedicated Environment

**Python** is the programming language used to run the converter. We will use Python 3.9.

### 5-1. Check Whether Python Is Installed

Run the following command in Terminal:

```
python3.9 --version
```

If you see `Python 3.9.x`, continue to Step 5-2. If you see `command not found` or a similar message, download and install the macOS installer for Python 3.9 from the [official Python website](https://www.python.org/downloads/macos/). Choose an installer compatible with your Mac's processor (Intel or Apple Silicon).

After installation, run the same command again to confirm the version.

### 5-2. Create a Working Environment

This tutorial requires several Python packages. We will create a separate **virtual environment** so that these packages do not interfere with packages used by your other projects.

Move to `~/htdemucs/htdemucs-coreml`, create a virtual environment, and activate it:

```
cd ~/htdemucs/htdemucs-coreml
python3.9 -m venv .venv
source .venv/bin/activate
```

If `(.venv)` appears at the start of the Terminal prompt, the **virtual environment for this tutorial is active**.

```text
(.venv) username@Mac-name ~/htdemucs/htdemucs-coreml %
```

It is also fine if `(base)` appears next to `(.venv)`.

### 5-3. Install the Required Python Packages

Install the packages needed by the converter. Copy the entire block into Terminal and execute it:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "numpy==2.0.2" scipy
python -m pip check
```

The first command updates `pip`, Python's package installer. The second installs the converter's dependencies. The third installs `NumPy 2.0.2` and `scipy`, which is needed for the final audio separation exercise. The last command checks whether the installed packages have conflicting dependency requirements. Installation may take some time.

`requirements.txt` already includes `NumPy`, but we install it separately here to pin the version to 2.0.2 for this tutorial.

When installation is complete, check that the required packages can be imported:

```
python -c 'import torch, demucs, coremltools, numpy, einops; print("Python packages: OK")'
```

If you see `Python packages: OK`, the imports succeeded. If an error occurs, consult the **Troubleshooting FAQ** below.

**What if you close Terminal and open a new window later?** You do not need to reinstall the packages. Run these two commands before continuing:

```
cd ~/htdemucs/htdemucs-coreml
source .venv/bin/activate
```

---

## Step 6. Copy the Checkpoint into the PyTorch Cache

Paste the following complete block into Terminal and press Return. It automatically copies the checkpoint file (`5c90dfd2-34c22ccb.th`) downloaded in Step 2 into the PyTorch cache directory.

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

If the last line begins with `Copied checkpoint to:` followed by a path, the copy succeeded.

---

## Step 7. Make a Few Changes to the Conversion Code

This is the most important part of the guide. The converter we downloaded uses settings for a four-stem model. We need to change them as shown below. We will also set the Core ML minimum deployment target to macOS 13 so that the converted model can be used on relatively older versions of macOS.

| Setting                  | Original                   | Modified                                  |
| ------------------------ | -------------------------- | ----------------------------------------- |
| Model name               | `htdemucs`                 | `htdemucs_6s`                             |
| Number of stems          | `4`                        | `6`                                       |
| Source reorder           | `[3, 0, 1, 2]`             | `[0, 1, 3, 4, 5, 2]`                      |
| Output names             | vocals, drums, bass, other | drums, bass, vocals, guitar, piano, other |
| Output shape description | `(1, 4, 2, …)`             | `(1, 6, 2, …)`                            |
| Minimum macOS version    | `macOS14`                  | `macOS13`                                 |

The original Demucs six-source model outputs stems in this order: **drums, bass, other, vocals, guitar, piano**. We will change the order to **drums, bass, vocals, guitar, piano, other**.

### 7-1. Download the Helper Script

Download [`prepare_six_stem.py`](resources/prepare_six_stem.py), provided with this tutorial. We assume it is saved in `~/Downloads`.

Run the following commands one line at a time:

```
cd ~/htdemucs/htdemucs-coreml
cp ~/Downloads/prepare_six_stem.py .
ls convert.py prepare_six_stem.py
```

The first command is a precaution in case you have moved to a different folder; you can skip it if you are already in the correct location.
The `cp` command copies the helper script into the current folder, and `ls` confirms that both files are present. If they are, run the helper:

```
python prepare_six_stem.py
```

The script backs up `convert.py` as `convert.py.original-4stem`, then modifies several sections of `convert.py`. If it encounters an error, it is designed to provide a detailed message.

> If the helper cannot find the expected sections of the original converter, it stops rather than applying incomplete changes. Refer to the FAQ below.

**Expected output:**

```text
SUCCESS: convert.py adapted for six stems.
Backup: convert.py.original-4stem
Model: htdemucs_6s
Output: drums, bass, vocals, guitar, piano, other
```

### 7-2. Verify the Changes

Copy the entire block into Terminal and press Return:

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
SOURCE_REORDER = [0, 1, 3, 4, 5, 2]
SOURCE_NAMES = ["drums", "bass", "vocals", "guitar", "piano", "other"]
```

### Why Do We Need to Modify the Converter?

Some operations in the original HTDemucs architecture cannot be directly converted by `coremltools`. The public converter implements complex-number operations using real-valued representations and provides conversion-compatible attention and overlap-add operations. **We are not retraining the weights**; the goal is to convert the trained network into Core ML format.

---

## Step 8. Create the FP16 Core ML Model

This is the final step in creating the **Core ML package**.

Move into the `htdemucs-coreml` folder, then run `convert.py`. Make sure `(.venv)` appears at the beginning of the Terminal prompt.

```
cd ~/htdemucs/htdemucs-coreml
python convert.py --fp16 --compute-units cpu_and_gpu
```

During conversion, you will see stages similar to the following:

```text
[1/5] Loading Demucs 'htdemucs_6s' ...
[2/5] Building real-valued wrapper ...
[3/5] PyTorch forward pass ...
[4/5] torch.jit.trace ...
[5/5] Core ML conversion ...
```

Converting this complex neural network can increase memory use. Check that conversion and validation finish **without errors**. You should see a validation result like this:

```
[Val] Validating Core ML vs. PyTorch reference ...

      Max diff:  0.005583
      Mean diff: 0.000232
      Validation OK.
```

The maximum output difference (`Max diff`) and average output difference (`Mean diff`) are better when smaller. The message `Validation OK.` indicates that the converter's validation passed.

You can ignore the `Next step:` message at the end; it is intended for Xcode developers. We will test the converted Core ML model on actual audio in the next steps.

Check that the model package `HTDemucs_CoreML_FP16.mlpackage` was created:

```
ls -ld HTDemucs_CoreML_FP16.mlpackage
```

---

## Step 9. Check the Model Inputs and Outputs

Use `coremltools` to inspect the input and output specifications of the model you just created.

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

This check may take a little time. You should see:

```text
Inputs:
  audio [1, 2, 441000]
Outputs:
  sources [1, 6, 2, 441000]
```

The shape `[1, 6, 2, 441000]` means that one audio input produces six stems, each with two channels (left and right) containing 441,000 samples (10 seconds at 44.1 kHz).

In Step 10, be sure to listen to the stems to verify both separation quality and source order. For example, `drums.wav` should contain drums, while `guitar.wav` should primarily contain guitar rather than an obviously different instrument.

---

## Troubleshooting FAQ

**Q. The `python3.9` command cannot be found.**  
A. Download Python 3.9.x from the [official Python website](https://www.python.org/downloads/macos/) and install it. Then run `python3.9 --version` in Terminal.

**Q. I get `No module named demucs` or another Python import error.**  
A. Run `source .venv/bin/activate` again and check `python -m pip install -r requirements.txt`.

**Q. Only four stems are produced.**  
A. Check that `MODEL_NAME`, `NUM_SOURCES`, `SOURCE_REORDER`, and `SOURCE_NAMES` match the values in Step 7, and make sure you have not selected an older `.mlpackage`.

**Q. I created the Core ML package, but loading or prediction fails.**  
A. Check your macOS version, `coremltools` version, available disk space, memory, and the `CPU_AND_GPU` setting. Do not change compute units without validating the result.

**Q. How do I leave the virtual environment?**  
A. Run `deactivate` in Terminal. When you return, go to the working folder and run `source .venv/bin/activate`.

---

## Step 10. Separate and Listen to the Stems

Now let's see how well the Core ML model separates a real audio recording into individual instruments and vocals.

Download the 10-second [`test_10s.wav`](resources/test_10s.wav) file.

1. In this GitHub repository, click **`test_10s.wav`**, then click the **Download raw file** icon.
2. Confirm that the file was saved in your Mac's `Downloads` folder.
3. Run the following commands one line at a time to move into the converter's working folder and copy the sample file:

```
cd ~/htdemucs/htdemucs-coreml
source .venv/bin/activate
cp ~/Downloads/test_10s.wav .
ls -lh test_10s.wav HTDemucs_CoreML_FP16.mlpackage
```

Copy the following entire code block into Terminal and press Return. It will separate the sample audio into six WAV files.

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

When the process finishes, open the results folder:

```
open separated_stems
```

You will find `drums.wav`, `bass.wav`, `vocals.wav`, `guitar.wav`, `piano.wav`, and `other.wav`. Double-click each file to listen, and compare it with the original `test_10s.wav`.

---

## Conclusion

You have now used an HTDemucs checkpoint to create your own Core ML model and separate six stems from real music. Rather than training a neural network from scratch, you converted a pretrained model to Core ML and applied it to new audio.

Listening to the six WAV files, you may notice that separation quality varies by instrument and that some sounds from other instruments remain audible. Even so, extracting separate instrument tracks from a single mixed recording is a fascinating capability.

**Keep your `HTDemucs_CoreML_FP16.mlpackage` file in a safe place!** You can reuse it later without repeating the conversion process.

---

## References

- [Original Meta Demucs repository](https://github.com/facebookresearch/demucs)
- [HTDemucs Core ML converter — dexxdean/htdemucs-coreml](https://github.com/dexxdean/htdemucs-coreml)
- [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)
- [Official Python downloads](https://www.python.org/downloads/macos/)
