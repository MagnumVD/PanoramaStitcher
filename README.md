# Panorama Stitcher

A lightweight desktop tool for manually and automatically stitching images into panoramas.

## Features

* Drag-and-drop style image placement
* Blender-based Hotkey transform tools
* Auto-alignment using feature matching (uses current selected image as base)
* Layer ordering
* Canvas resizing and background image support
* Highres export (PNG/JPEG)

## Installation

Install dependencies:

```
pip install PyQt6 opencv-python numpy pillow
```

## Usage

```
python main.py
```

### Basic Workflow

1. **Add images** using the left panel
2. **Select an image** to be used as the starting point of the stitch, this should be a wide angle shot or one with a lot of overlapping neighbours if possible.
3. Click **Auto Align**
4. Adjust layering if needed
5. Click on a random place on the canvas to select the camera. Set your resolution in the **top left** and use G,R,S to position it to the frame you want to export
5. Click **Stitch and Export**

## Controls

* **Scroll** - Zoom
* **Ctrl + Scroll** - Fine zoom (implemented for touchpad controls)
* **Middle Mouse Drag** - Pan
* **Leftclick on image** - Select image
* **Leftclick on canvas** - Select camera

* **G** - Move
* **R** - Rotate
* **S** - Scale
* **Enter** or **Leftclick** - Confirm transform
* **Esc** or **Rightclick** - Cancel transform

## Future Improvements

* Compiled releases
* 4-point transform controls
* Blending (exposure compensation, auto-masks & mask painting, laplacian blending)
* Better auto-alignment robustness
* Video support (tracking, cleanplate-generation)