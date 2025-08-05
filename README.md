# Robust Spiking Neural Network (SNN) Framework

This repository contains the official code for our **Robust Semantic Communication enabled by Spiking Neural Networks**.

## Environment

The code has been tested with the following environment:

- Python: `3.10.13`  
- SpikingJelly: `0.0.0.0.14`  
- NumPy: `1.24.3`  
- SciPy: `1.15.1`  
- h5py: `3.5.0`  

## Dataset: DVS Gesture

> ⚠️ **Note:** The DVS Gesture dataset does not support automatic download.

To use it, please follow these steps:

1. Create a folder named `download` inside the `data/` directory:
   ```bash
   mkdir -p ./data/download
   ```
2. Manually download the dataset from the following link:  
   [https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)
3. Place the downloaded file(s) into the `./data/download/` directory.

## Current Status

The current version includes the implementation of adversarial attacks on the DVS Gesture dataset.
The full codebase will be uploaded shortly.

---

## License

This project is licensed under the MIT License.

---

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in  
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.
