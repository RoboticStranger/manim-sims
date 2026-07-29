from manim import *
import numpy as np

class SurfaceExample(ThreeDScene):
    def construct(self):
        # 1. Setup Axes
        axes = ThreeDAxes()
        
        # 2. Define the Surface Function
        # u, v are our parameters (like x and y)
        def ripple_func(u, v):
            x = u
            y = v
            z = np.sin(u**2 + v**2) # The "height" of the ripple
            return np.array([x, y, z])

        # 3. Create the Surface object
        surface = Surface(
            ripple_func,
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(32, 32), # Density of the mesh
            checkerboard_colors=[BLUE_D, BLUE_E],
            fill_opacity=0.7
        )

        # 4. Camera Orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, zoom=0.3)
        
        # 5. Render
        self.add(axes)
        self.play(Create(surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.1) # Slowly rotate
        self.wait(3)

        self.move_camera(
            phi=60 * DEGREES,   # Vertical tilt
            theta=-120 * DEGREES, # Horizontal rotation
            zoom=0.3,               # Zoom (default is 1)
            # frame_center=[2, 0, 1], # Move the "look at" point
            run_time=3
        )