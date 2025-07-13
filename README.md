# 🎨 Neural Style Transfer App

## 📸 Demo

<div align="center">

  <img src="demo_pictures/Screenshot 2025-07-13 122424.png" width="700" alt="Demo 1"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 123739.png" width="700" alt="Demo 2"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 123906.png" width="700" alt="Demo 3"/>

  <br><br>

  <em>🎨 Neural Style Transfer in action - Transform any photo into artistic masterpiece!</em>

</div>

---

## � About

A simple Neural Style Transfer application for fun! This project combines the content of one image with the artistic style of another using deep learning. Built with PyTorch and Gradio for an easy-to-use web interface.

### Features

- Web-based interface with Gradio
- Real-time style transfer
- Adjustable parameters (style weight, content weight, epochs, learning rate)
- GPU support for faster processing
- Multiple initialization options

---

## �️ Tech Stack

- **PyTorch** - Deep learning framework
- **VGG19** - Pre-trained model for feature extraction
- **Gradio** - Web interface
- **PIL** - Image processing

---

## 🏗️ Model Architecture

The neural style transfer uses a **VGG19** pre-trained network as feature extractor:

### Content Representation
- **Layer**: `conv4_2` (layer index 21)
- **Purpose**: Captures high-level content structure while preserving spatial information
- **Loss**: MSE between content and generated image features

### Style Representation
- **Layers**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` (indices 0, 5, 10, 19, 28)
- **Purpose**: Captures texture and artistic patterns across multiple scales
- **Method**: Gram matrices of feature maps to represent style correlations
- **Loss**: MSE between style and generated image Gram matrices

```python
# Layer configuration
content_layer_n = 21                    # VGG19 conv4_2
style_layers = [0, 5, 10, 19, 28]      # Multiple conv layers for style
```

---

## 🔧 Techniques Implemented

This project showcases several deep learning optimization techniques:

- **Adam Optimizer** - Adaptive learning rate optimization
- **Learning Rate Scheduling** - StepLR with decay for stable convergence
- **Gradient Clipping** - Prevents gradient explosion during training
- **Mixed Precision Training** - Memory efficient training with AMP
- **Gram Matrix** - Style representation using normalized feature correlations

```python
# Example: Gradient clipping implementation
torch.nn.utils.clip_grad_norm_([generated_image], max_norm=1.0)

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
```

---

## 🚀 How to Run

1. **Install dependencies**
```bash
pip install torch torchvision gradio pillow
```

2. **Run the app**
```bash
python app.py
```

3. **Open the web interface** and start creating art!

---

## ⚙️ Parameters

| Parameter | What it does | Range |
|-----------|-------------|-------|
| Style Weight | How much style to apply | 10-1000 |
| Content Weight | How much original content to keep | 1-100 |
| Epochs | Number of optimization steps | 100-10000 |
| Learning Rate | Optimization step size | 0.01-0.5 |

---

## 🎯 What I Learned

This project helped me practice:
- Neural network optimization techniques
- PyTorch implementation
- Loss function design (content + style loss)
- GPU programming and memory management
- Building user-friendly ML applications

---

## �️ More Demo Results

<div align="center">

  <img src="demo_pictures/Screenshot 2025-07-13 124202.png" width="600" alt="Demo 4"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 124454.png" width="600" alt="Demo 5"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 124625.png" width="600" alt="Demo 6"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 125205.png" width="600" alt="Demo 7"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 125407.png" width="600" alt="Demo 8"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 125910.png" width="600" alt="Demo 9"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 130829.png" width="600" alt="Demo 10"/>

  <br><br>

  <img src="demo_pictures/Screenshot 2025-07-13 123156.png" width="600" alt="Demo 10"/>



  <em>✨ Explore different artistic styles and see the magic happen!</em>

</div>

---

## �📄 License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.