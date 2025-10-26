
# PurePython (PyGL): A Hyper-Optimized Software Rasterizer

![Project Status: Performance Pass 5 - Dynamic Resolution](https://img.shields.io/badge/Status-Performance%20Pass%205-blue)
![Language: Python](https://img.shields.io/badge/Language-Python-green)
![Dependencies: NumPy, Tkinter, Pillow](https://img.shields.io/badge/Dependencies-NumPy%2C%20Tkinter%2C%20Pillow-brightgreen)

## 🚀 Overview

**PurePython** (or PyGL) is a custom, educational 3D software rasterizer built from scratch using pure Python, leveraging the power of **NumPy** for extreme performance through vectorization.

This project is not a wrapper for an existing graphics library; it implements the entire graphics pipeline—from ModelView-Projection matrices and vertex transformation to perspective divide, clipping, backface culling, and full-screen triangle rasterization with depth testing—all on the CPU.

### Performance Pass 5: Dynamic Resolution

The current version integrates an advanced performance feature: **Dynamic Resolution Scaling**. This system continuously monitors the frame rate and automatically adjusts the internal rendering resolution to maintain a target FPS, ensuring a smooth experience even on slower machines or during complex scenes.

---

## ✨ Features

*   **Pure Python / NumPy Core:** Highly optimized matrix operations and vectorized triangle rasterization.
*   **OpenGL-like API:** Simple, familiar syntax for setting up scenes (`glMatrixMode`, `glBegin`/`glEnd`, `gluPerspective`, etc.).
*   **Full 3D Pipeline:** Supports vertex transformation, W-clipping, perspective divide, and viewport mapping.
*   **Essential Graphics Primitives:** Includes **Depth Testing** (`GL_DEPTH_TEST`) and **Backface Culling** (`GL_CULL_FACE`).
*   **FPS Camera Controls:** Built-in `Camera` class with mouse-look and WASD movement.
*   **Dynamic Resolution:** Renders at a lower resolution when needed and scales up to the fixed window size using `Image.Resampling.NEAREST` (via Pillow) for fast, pixelated display.
*   **Fullscreen Toggle:** Press `Esc` to toggle between windowed and fullscreen mode.

## 🛠️ Installation

### Prerequisites

You must have Python 3 installed. The following dependencies are required:

```bash
pip install numpy pillow
```

### Running the Demo

1. Save the provided code as `main.py`.
2. Execute the file from your terminal:

```bash
python PythonGL.py
```

## 💻 Usage Example

The `main.py` file includes a full working example that renders a rotating 3D colored cube, demonstrating the core functionality and performance features.

```python
# Snippet from PythonGL.py
def example_solid_cube():
    gl = PyGL()
    gl.init_display(800, 600)
    # ... setup projection, clear color, enable depth/cull ...
    
    # ... defines vertices and faces for a cube ...

    def render(delta_time):
        gl.clear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        gl._multiply_matrix(gl.camera.get_view_matrix())
        gl.glTranslate(0.0, 0.0, -5.0) 
        gl.glRotate(rotation_angle, 0.5, 1.0, 0.2)
        
        gl.glBegin(gl.GL_QUADS)
        # ... draw cube vertices ...
        gl.glEnd()
        
        # Display real-time performance metrics
        gl.draw_text(f"Render Scale: {gl.current_render_scale:.2f}", 10, 10)
        gl.draw_text(f"Render Res: {gl.render_width}x{gl.render_height}", 10, 30)

        gl.swap_buffers()

    gl.main_loop(render)
```

## 🎮 Controls

| Action | Key / Input | Notes |
| :--- | :--- | :--- |
| **Move Forward** | `W` | |
| **Move Backward** | `S` | |
| **Strafe Left** | `A` | |
| **Strafe Right** | `D` | |
| **Look Around** | Mouse Drag (LMB held) | Camera pitch and yaw control. |
| **Toggle Fullscreen** | `Escape` | Toggles between the fixed window size and the maximum screen resolution. |

## ⚙️ Dynamic Resolution Logic

The `PyGL` class contains logic to ensure a minimum framerate:

*   **Target FPS:** `30` (A smooth baseline for the renderer).
*   **Minimum Acceptable FPS:** `15` (The threshold for dropping resolution).
*   **Scale Range:** The render scale can drop as low as `0.25` (25% of the base resolution) to maintain performance.

When the FPS drops below 15, the `current_render_scale` is aggressively lowered by `0.2`. When performance is excellent (FPS > 45), the scale is slowly ramped up by `0.1` to recover image quality.

