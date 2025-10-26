import numpy as np
import tkinter as tk
from tkinter import Canvas
import math
import time
import subprocess

try:
    # Ensure Pillow is imported as it is required for this fix
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow library not found. Please install it with 'pip install pillow'")
    exit()

class Camera:
    def __init__(self, position=(0.0, 1.0, 3.0), up=(0.0, 1.0, 0.0), yaw=-90.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.right = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = self.up
        self.yaw = yaw
        self.pitch = pitch
        self.movement_speed = 0.05
        self.mouse_sensitivity = 0.1
        self.update_camera_vectors()

    def update_camera_vectors(self):
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float32)
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)

    def get_view_matrix(self):
        return self.look_at(self.position, self.position + self.front, self.up)

    def look_at(self, eye, center, up):
        f = (center - eye) / np.linalg.norm(center - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)

        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0:3] = s
        matrix[1, 0:3] = u
        matrix[2, 0:3] = -f
        matrix[0:3, 3] = -np.array([np.dot(s, eye), np.dot(u, eye), np.dot(-f, eye)])
        return matrix

    def process_keyboard(self, direction):
        if direction == "FORWARD":
            self.position += self.front * self.movement_speed
        elif direction == "BACKWARD":
            self.position -= self.front * self.movement_speed
        elif direction == "LEFT":
            self.position -= self.right * self.movement_speed
        elif direction == "RIGHT":
            self.position += self.right * self.movement_speed

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= -self.mouse_sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self.update_camera_vectors()

class PyGL:
    """Custom software renderer with PyOpenGL-like syntax - Performance Pass 5 (Dynamic Resolution)"""

    GL_TRIANGLES, GL_QUADS = 0x0004, 0x0007
    GL_MODELVIEW, GL_PROJECTION = 0x1700, 0x1701
    GL_DEPTH_TEST, GL_CULL_FACE = 0x0B71, 0x0B44
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x00004000, 0x00000100

    def __init__(self):
        # Fixed Window Size (Tkinter size)
        self.width, self.height = 800, 600
        self.original_width, self.original_height = 800, 600 # Windowed Mode Size

        # Dynamic Resolution State (Internal Rendering Size)
        self.base_width, self.base_height = self.width, self.height # Full res target
        self.current_render_scale = 1.0 # 1.0 = full resolution
        self.min_render_scale = 0.25    # Drastically lowered minimum scale
        self.target_fps = 30
        self.min_acceptable_fps = 15
        
        self.render_width, self.render_height = self.base_width, self.base_height
        self.frame_buffer = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        self.depth_buffer = np.full((self.render_height, self.render_width), float('inf'), dtype=np.float32)
        
        self.matrix_mode = self.GL_MODELVIEW
        self.modelview_stack = [np.eye(4, dtype=np.float32)]
        self.projection_stack = [np.eye(4, dtype=np.float32)]
        self.current_color = [1.0, 1.0, 1.0, 1.0]
        self.vertex_buffer, self.color_buffer = [], []
        self.primitive_type = self.GL_TRIANGLES
        self.depth_test_enabled = False
        self.cull_face_enabled = False
        self.viewport_x, self.viewport_y = 0, 0
        self.viewport_width, self.viewport_height = self.render_width, self.render_height
        self.root, self.canvas, self.photo_image = None, None, None
        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_down = False
        self.keys_pressed = set()
        self.camera = Camera(position=(0.0, 1.0, -3.0)) 
        self.last_frame_time = time.time() # For delta time calculation
        self.last_scale_adjustment_time = time.time()

    def resize_frame_buffers(self):
        # 1. Update internal render dimensions based on base dimensions and scale
        self.render_width = int(self.base_width * self.current_render_scale)
        self.render_height = int(self.base_height * self.current_render_scale)
        
        # 2. Recreate internal buffers
        self.frame_buffer = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        self.depth_buffer = np.full((self.render_height, self.render_width), float('inf'), dtype=np.float32)
        
        # FIX: Recreate PIL image to match the new render dimensions
        self.pil_image = Image.fromarray(self.frame_buffer)

        # 3. Update GL Viewport to match new render dimensions
        self.glViewport(0, 0, self.render_width, self.render_height)
        
        # 4. Re-calculate projection matrix 
        if self.matrix_mode == self.GL_PROJECTION:
            self.glLoadIdentity()
            self.gluPerspective(45, self.base_width/self.base_height, 0.1, 100.0)
    
    def check_performance_lag(self, delta_time):
        current_fps = 1.0 / delta_time if delta_time > 0 else float('inf')
        
        if time.time() - self.last_scale_adjustment_time < 0.5:
            return

        scale_changed = False
        
        # Lagging: Drop resolution (More drastic step: -0.2)
        if current_fps < self.min_acceptable_fps and self.current_render_scale > self.min_render_scale:
            self.current_render_scale = max(self.min_render_scale, self.current_render_scale - 0.2)
            scale_changed = True
        
        # Performing well: Increase resolution (More drastic step: +0.1)
        elif current_fps > self.target_fps * 1.5 and self.current_render_scale < 1.0:
            self.current_render_scale = min(1.0, self.current_render_scale + 0.1)
            scale_changed = True
                
        if scale_changed:
            self.resize_frame_buffers()
            self.last_scale_adjustment_time = time.time()


    def init_display(self, width=800, height=600, title="Hyper-Optimized 3D Renderer"):
        self.width, self.height = width, height
        self.original_width, self.original_height = width, height
        self.base_width, self.base_height = width, height
        self.render_width, self.render_height = width, height
        
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}+100+100") # Fixed window size
        self.root.overrideredirect(False) 
        
        self.canvas = Canvas(self.root, width=width, height=height)
        self.canvas.pack()
        
        # Initialize PIL image for display
        self.pil_image = Image.fromarray(self.frame_buffer)
        self.photo_image = ImageTk.PhotoImage(image=self.pil_image.resize((self.width, self.height), Image.Resampling.NEAREST))
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
        
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

    def clear_color(self, r, g, b, a=1.0): self.clear_color_value = (int(r*255), int(g*255), int(b*255))
    def clear(self, mask):
        if mask & self.GL_COLOR_BUFFER_BIT: self.frame_buffer[:] = getattr(self, 'clear_color_value', 0)
        if mask & self.GL_DEPTH_BUFFER_BIT: self.depth_buffer[:] = float('inf')
    def enable(self, cap):
        if cap == self.GL_DEPTH_TEST: self.depth_test_enabled = True
        elif cap == self.GL_CULL_FACE: self.cull_face_enabled = True
    def disable(self, cap):
        if cap == self.GL_DEPTH_TEST: self.depth_test_enabled = False
        elif cap == self.GL_CULL_FACE: self.cull_face_enabled = False
    def glMatrixMode(self, mode): self.matrix_mode = mode
    def glLoadIdentity(self):
        stack = self.modelview_stack if self.matrix_mode == self.GL_MODELVIEW else self.projection_stack
        stack[-1] = np.eye(4, dtype=np.float32)
    def glPushMatrix(self):
        stack = self.modelview_stack if self.matrix_mode == self.GL_MODELVIEW else self.projection_stack
        stack.append(stack[-1].copy())
    def glPopMatrix(self):
        stack = self.modelview_stack if self.matrix_mode == self.GL_MODELVIEW else self.projection_stack
        if len(stack) > 1: stack.pop()
    def _multiply_matrix(self, matrix):
        stack = self.modelview_stack if self.matrix_mode == self.GL_MODELVIEW else self.projection_stack
        stack[-1] = stack[-1] @ matrix
    def glTranslate(self, x, y, z): self._multiply_matrix(np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]], dtype=np.float32))
    def glRotate(self, angle, x, y, z):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        l=math.sqrt(x*x+y*y+z*z); x/=l; y/=l; z/=l; nc=1-c
        self._multiply_matrix(np.array([[c+x*x*nc,x*y*nc-z*s,x*z*nc+y*s,0],[y*x*nc+z*s,c+y*y*nc,y*z*nc-x*s,0],[z*x*nc-y*s,z*y*nc+x*s,c+z*z*nc,0],[0,0,0,1]], dtype=np.float32))
    def gluPerspective(self, fovy, aspect, near, far):
        f=1.0/math.tan(math.radians(fovy)/2.0)
        proj = np.array([[f/aspect,0,0,0],[0,f,0,0],[0,0,(far+near)/(near-far),(2*far*near)/(near-far)],[0,0,-1,0]], dtype=np.float32)
        if self.matrix_mode == self.GL_PROJECTION: self.projection_stack[-1] = proj
    def glScale(self, x, y, z):
        self._multiply_matrix(np.array([[x,0,0,0],[0,y,0,0],[0,0,z,0],[0,0,0,1]], dtype=np.float32))
    def glViewport(self, x, y, width, height):
        self.viewport_x, self.viewport_y, self.viewport_width, self.viewport_height = x, y, width, height
    def glColor3f(self, r, g, b): self.current_color = [r, g, b, 1.0]
    def glVertex3f(self, x, y, z):
        self.vertex_buffer.append([x, y, z, 1.0])
        self.color_buffer.append(self.current_color)
    def glBegin(self, primitive):
        self.primitive_type, self.vertex_buffer, self.color_buffer = primitive, [], []
    
    def glEnd(self):
        if not self.vertex_buffer: return
        
        verts_np = np.array(self.vertex_buffer, dtype=np.float32)
        colors_np = np.array(self.color_buffer, dtype=np.float32)

        modelview = self.modelview_stack[-1]
        projection = self.projection_stack[-1]
        mvp = projection @ modelview
        
        verts_camera = verts_np @ modelview.T
        verts_clip = verts_np @ mvp.T

        if self.primitive_type == self.GL_QUADS:
            num_quads = len(verts_np) // 4
            base_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
            quad_offsets = np.arange(0, num_quads * 4, 4, dtype=np.int32)
            indices = (base_indices[np.newaxis, :] + quad_offsets[:, np.newaxis]).flatten()

        elif self.primitive_type == self.GL_TRIANGLES:
            indices = np.arange(len(verts_np))
        else: return

        tri_verts = verts_clip[indices].reshape(-1, 3, 4) # Vertices in clip space
        tri_colors = colors_np[indices].reshape(-1, 3, 4)
        tri_camera_verts = verts_camera[indices].reshape(-1, 3, 4)

        # --- New Clipping (before perspective divide) ---
        clipped_data = self._clip_to_near_plane(tri_verts, tri_colors, tri_camera_verts)
        if clipped_data is None: return
        tri_verts, tri_colors, tri_camera_verts = clipped_data
        
        if tri_verts.shape[0] == 0: return # No valid triangles left after clipping

        # --- Face Culling (Backface Culling) ---
        if self.cull_face_enabled:
            v0, v1, v2 = tri_camera_verts[:, 0, :3], tri_camera_verts[:, 1, :3], tri_camera_verts[:, 2, :3]
            normals = np.cross(v1 - v0, v2 - v0)
            cull_mask = np.einsum('ij,ij->i', normals, v0) < 0 
            
            tri_verts = tri_verts[cull_mask]
            tri_colors = tri_colors[cull_mask]
            if tri_verts.shape[0] == 0: return

        # --- Perspective Divide (to NDC) ---
        w = tri_verts[:, :, 3:] 
        one_over_w = 1.0 / w
        interpolated_colors_over_w = tri_colors / w
        verts_ndc = tri_verts / w

        # --- Viewport Transform (to Screen Coordinates) ---
        screen_coords = np.empty_like(verts_ndc[:, :, :3])
        screen_coords[:, :, 0] = (verts_ndc[:, :, 0] + 1.0) * 0.5 * self.render_width + self.viewport_x 
        screen_coords[:, :, 1] = (1.0 - verts_ndc[:, :, 1]) * 0.5 * self.render_height + self.viewport_y 
        screen_coords[:, :, 2] = verts_ndc[:, :, 2]
        
        for i in range(screen_coords.shape[0]):
            self._draw_triangle_vectorized(screen_coords[i], interpolated_colors_over_w[i], one_over_w[i])

    def _clip_to_near_plane(self, tri_verts, tri_colors, tri_camera_verts):
        w_values = tri_verts[:, :, 3] 
        w_clip_mask = (w_values > 1e-6).all(axis=1) 
        
        return tri_verts[w_clip_mask], tri_colors[w_clip_mask], tri_camera_verts[w_clip_mask]

    def _draw_triangle_vectorized(self, tri_points_screen, tri_colors_over_w, tri_one_over_w):
        p1, p2, p3 = tri_points_screen[0], tri_points_screen[1], tri_points_screen[2]
        c1_ow, c2_ow, c3_ow = tri_colors_over_w[0], tri_colors_over_w[1], tri_colors_over_w[2]
        ow1, ow2, ow3 = tri_one_over_w[0], tri_one_over_w[1], tri_one_over_w[2]
        
        min_x = max(0, int(np.min(tri_points_screen[:, 0])))
        max_x = min(self.render_width, int(np.max(tri_points_screen[:, 0])) + 1)
        min_y = max(0, int(np.min(tri_points_screen[:, 1])))
        max_y = min(self.render_height, int(np.max(tri_points_screen[:, 1])) + 1)

        if min_x >= max_x or min_y >= max_y: return

        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)
        xx, yy = np.meshgrid(x_range, y_range)

        den = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
        if abs(den) < 1e-6: return
        
        alpha = ((p2[1] - p3[1])*(xx - p3[0]) + (p3[0] - p2[0])*(yy - p3[1])) / den
        beta  = ((p3[1] - p1[1])*(xx - p3[0]) + (p1[0] - p3[0])*(yy - p3[1])) / den
        gamma = 1.0 - alpha - beta

        mask = (alpha >= -1e-6) & (beta >= -1e-6) & (gamma >= -1e-6) 
        if not mask.any(): return

        alpha_m, beta_m, gamma_m = alpha[mask], beta[mask], gamma[mask]
        
        one_over_w_interpolated = alpha_m * ow1 + beta_m * ow2 + gamma_m * ow3
        color_over_w_interpolated = (np.outer(alpha_m, c1_ow) + np.outer(beta_m, c2_ow) + np.outer(gamma_m, c3_ow))
        z_interpolated = alpha_m * p1[2] + beta_m * p2[2] + gamma_m * p3[2]

        pixels_to_draw_mask_1D = np.ones_like(alpha_m, dtype=bool) 

        if self.depth_test_enabled:
            current_depth_buffer_values = self.depth_buffer[yy[mask], xx[mask]]
            depth_test_passed_1D = z_interpolated < current_depth_buffer_values
            pixels_to_draw_mask_1D = depth_test_passed_1D
            self.depth_buffer[yy[mask][depth_test_passed_1D], xx[mask][depth_test_passed_1D]] = z_interpolated[depth_test_passed_1D]
        
        if not pixels_to_draw_mask_1D.any(): return

        final_one_over_w = one_over_w_interpolated[pixels_to_draw_mask_1D].reshape(-1, 1)
        final_color_over_w = color_over_w_interpolated[pixels_to_draw_mask_1D]

        color = (final_color_over_w / final_one_over_w) * 255
        
        framebuffer_update_mask_2D = np.zeros_like(mask, dtype=bool)
        framebuffer_update_mask_2D[mask] = pixels_to_draw_mask_1D

        self.frame_buffer[min_y:max_y, min_x:max_x][framebuffer_update_mask_2D] = color[:, :3].astype(np.uint8)

    def swap_buffers(self):
        if self.root:
            # 1. Update PIL Image content from the (scaled) frame buffer
            self.pil_image.paste(Image.fromarray(self.frame_buffer))
            
            # 2. Resize the low-res image UP to the fixed window size for display
            # self.width/self.height is the fixed Tkinter window size
            resized_image = self.pil_image.resize((self.width, self.height), Image.Resampling.NEAREST)
            
            # 3. Update the PhotoImage on the Tkinter canvas
            self.photo_image.paste(resized_image)

            # Recreate image item and draw text
            self.canvas.delete("all") 
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)

    def draw_text(self, text, x, y, color="white", font_size=16, anchor="nw"):
        if self.canvas:
            self.canvas.create_text(x, y, text=text, fill=color, font=("Arial", font_size), anchor=anchor)

    def _on_mouse_motion(self, event):
        if self.mouse_down:
            dx = event.x - self.mouse_x
            dy = event.y - self.mouse_y
            self.camera.process_mouse_movement(dx, dy)
        self.mouse_x, self.mouse_y = event.x, event.y

    def _on_mouse_down(self, event):
        self.mouse_down = True

    def _on_mouse_up(self, event):
        self.mouse_down = False

    def _on_key_press(self, event):
        self.keys_pressed.add(event.keysym)
        if event.keysym == 'Escape':
            is_fullscreen = self.root.overrideredirect()
            
            if is_fullscreen: # Transition to Windowed
                self.root.overrideredirect(False) 
                self.root.geometry(f"{self.original_width}x{self.original_height}") 
                self.width = self.original_width
                self.height = self.original_height
            else: # Transition to Fullscreen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}+0+0")
                self.root.overrideredirect(True)
                self.width = screen_width
                self.height = screen_height
            
            # Update the fixed base dimensions and re-initialize buffers at the current scale
            self.base_width = self.width
            self.base_height = self.height
            
            # Canvas configuration to new fixed size
            self.canvas.config(width=self.width, height=self.height)
            self.canvas.pack()
            self.resize_frame_buffers() # This handles buffers, PIL image, and projection
            
            # Recreate PhotoImage object for display on canvas
            self.photo_image = ImageTk.PhotoImage(image=self.pil_image.resize((self.width, self.height), Image.Resampling.NEAREST))
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)

    def _on_key_release(self, event):
        self.keys_pressed.discard(event.keysym)

    def update_camera(self):
        if "w" in self.keys_pressed: self.camera.process_keyboard("FORWARD")
        if "s" in self.keys_pressed: self.camera.process_keyboard("BACKWARD")
        if "a" in self.keys_pressed: self.camera.process_keyboard("LEFT")
        if "d" in self.keys_pressed: self.camera.process_keyboard("RIGHT")

    def main_loop(self, game_loop_func):
        self.running = True
        def on_close():
            self.running = False
            self.root.destroy() 
        self.root.protocol("WM_DELETE_WINDOW", on_close)

        while self.running:
            try:
                current_time = time.time()
                delta_time = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                self.check_performance_lag(delta_time) 

                self.update_camera()
                game_loop_func(delta_time) 
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self.running = False
        
def example_solid_cube():
    gl = PyGL()
    gl.init_display(800, 600)
    gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity()
    gl.gluPerspective(45, 800/600, 0.1, 100.0) 
    gl.glViewport(0, 0, 800, 600)
    gl.clear_color(0.1, 0.1, 0.2, 1.0)
    gl.enable(gl.GL_DEPTH_TEST); gl.enable(gl.GL_CULL_FACE)
    
    rotation_angle = 0
    vertices = [[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]]
    vertex_colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
        [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0] 
    ]
    faces = [[0,1,2,3],[4,7,6,5],[4,0,3,7],[5,6,2,1],[5,1,0,4],[6,7,3,2]]

    def update(delta_time):
        nonlocal rotation_angle
        rotation_angle += 1

    def render(delta_time):
        gl.clear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # Apply camera view transformation
        gl._multiply_matrix(gl.camera.get_view_matrix())

        # Set cube position at Z=-5.0 (Camera starts at Z=-3.0, distance is 2.0 units)
        gl.glTranslate(0.0, 0.0, -5.0) 
        gl.glRotate(rotation_angle, 0.5, 1.0, 0.2)
        
        gl.glBegin(gl.GL_QUADS)
        for i, face in enumerate(faces):
            for vertex_index in face:
                gl.glColor3f(*vertex_colors[vertex_index])
                gl.glVertex3f(*vertices[vertex_index])
        gl.glEnd()
        
        # Draw current resolution scale for debugging
        gl.draw_text(f"Render Scale: {gl.current_render_scale:.2f}", 10, 10)
        gl.draw_text(f"Render Res: {gl.render_width}x{gl.render_height}", 10, 30)

        gl.swap_buffers()

    gl.main_loop(render)

if __name__ == "__main__":
    example_solid_cube()