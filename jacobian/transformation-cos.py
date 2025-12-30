from manim import *
import numpy as np

# Cosine:  z = k cos(xy/sigma)
#
#
class JacobianTransformation(ThreeDScene):
    def construct(self):
        # 1. Trackers & Constants
        u_tracker = ValueTracker(0.0)
        v_tracker = ValueTracker(0.0)
        ds = 1.0 
        
        # INDEPENDENT SETTINGS
        # Change these values freely; they won't break the other frame.
        uv_scale = 0.18
        uv_pos = LEFT * 1.5  # + UP * 0.5

        # 2. LEFT FRAME: UV SPACE
        axes_2d = NumberPlane(
            x_range=[-2, 2, 1], 
            y_range=[-2, 2, 1], 
            x_length=6, 
            y_length=6,
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).scale(uv_scale).move_to(uv_pos)
        
        u_lab = MathTex("u", font_size=10).next_to(
            axes_2d.x_axis.get_end(), # Get the tip of the axis
            DOWN,                     # Place it below
            buff=0.1                  # Small gap
        )
        # Position 'v' to the left of the top tip of the y-axis
        v_lab = MathTex("v", font_size=10).next_to(
            axes_2d.y_axis.get_end(), # Get the tip of the axis
            LEFT,                     # Place it to the left
            buff=0.1                  # Small gap
        )
        uv_frame = VGroup(axes_2d, u_lab, v_lab)

        # 3. RIGHT FRAME: XYZ SPACE
        xyz_scale = 0.18
        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-1, 3, 1], 
            x_length=10, y_length=10, z_length=10
        ).scale(xyz_scale)

        # Centering on the screen:
        # This ensures the physical center of the axis (at z=1) is in the middle of the frame.
        axes_3d.move_to(ORIGIN)

        # 4. MAPPING FUNCTIONS (Locked to their respective axes)
        def get_uv_pt(u, v):
            return axes_2d.c2p(u, v)

        k = 2.0
        sigma = 3.0
        def get_xyz_pt(u, v):
            z = k * np.cos(u*v/sigma)
            return axes_3d.c2p(u, v, z)

        # 5. OBJECTS
        surface = Surface(
            lambda u, v: get_xyz_pt(u, v),
            u_range=[-2,2], v_range=[-2, 2],
            resolution=(30, 30), fill_opacity=0.3,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )

        # 2D Unit Square & Basis
        patch_2d = always_redraw(lambda: Polygon(
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_uv_pt(u_tracker.get_value() + ds, v_tracker.get_value()),
            get_uv_pt(u_tracker.get_value() + ds, v_tracker.get_value() + ds),
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value() + ds),
            fill_opacity=0.5, fill_color=GREEN, stroke_color=WHITE, stroke_width=2
        ))

        vec_u_2d = always_redraw(lambda: Arrow(
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_uv_pt(u_tracker.get_value() + 1, v_tracker.get_value()),
            color=YELLOW, buff=0, stroke_width=4
        ))
        
        vec_v_2d = always_redraw(lambda: Arrow(
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value() + 1),
            color=PINK, buff=0, stroke_width=4
        ))

        # 3D Patch & Tangent Vectors
        patch_3d = always_redraw(lambda: Polygon(
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()),
            # Point 2: Start + Vector U
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()) + 
                (get_xyz_pt(u_tracker.get_value() + ds, v_tracker.get_value()) - 
                get_xyz_pt(u_tracker.get_value(), v_tracker.get_value())),
            # Point 3: Start + Vector U + Vector V (The true parallelogram corner)
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()) + 
                (get_xyz_pt(u_tracker.get_value() + ds, v_tracker.get_value()) - 
                get_xyz_pt(u_tracker.get_value(), v_tracker.get_value())) +
                (get_xyz_pt(u_tracker.get_value(), v_tracker.get_value() + ds) - 
                get_xyz_pt(u_tracker.get_value(), v_tracker.get_value())),
            # Point 4: Start + Vector V
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()) + 
                (get_xyz_pt(u_tracker.get_value(), v_tracker.get_value() + ds) - 
                get_xyz_pt(u_tracker.get_value(), v_tracker.get_value())),
            fill_opacity=0.8, fill_color=GREEN, stroke_color=WHITE, stroke_width=1
        ))

        vec_u_3d = always_redraw(lambda: Arrow(
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_xyz_pt(u_tracker.get_value() + 1, v_tracker.get_value()),
            color=YELLOW, buff=0, stroke_width=4
        ))

        vec_v_3d = always_redraw(lambda: Arrow(
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value() + 1),
            color=PINK, buff=0, stroke_width=4
        ))

        # 1. Group all 3D elements
        three_d_elements = Group(axes_3d, surface, patch_3d, vec_u_3d, vec_v_3d)

        # 6. UI & CAMERA
        self.set_camera_orientation(phi=80 * DEGREES, theta=60 * DEGREES)
        
        jac_label = MathTex(
            r"\|\mathbf{r}_u \times \mathbf{r}_v\| = ", 
            font_size=8
        )
        #.next_to(axes_3d, RIGHT, buff=0.1)

        # Define the math formulas
        surf_eq = MathTex(r"\mathbf{r}(u, v) = \langle u, v, k\cos(uv/\sigma) \rangle", font_size=8)
        partial_u = MathTex(r"\mathbf{r}_u = \langle 1, 0, -\frac{kv}{\sigma}\sin(uv/\sigma) \rangle", font_size=8)
        partial_v = MathTex(r"\mathbf{r}_v = \langle 0, 1, -\frac{ku}{\sigma}\sin(uv/\sigma) \rangle", font_size=8)

        # Arrange them in a column
        formula_stack = VGroup(surf_eq, partial_u, partial_v, jac_label).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # Position the stack above your existing jac_label
        formula_stack.next_to(axes_3d, RIGHT, buff=0.1)
            #jac_label, UP, buff=0.1)

        # Calculate the numerical value of the Jacobian magnitude
        def get_jac_mag():
            u = u_tracker.get_value()
            v = v_tracker.get_value()
            
            # Common term to simplify the calculation
            sin_term = np.sin(u * v / sigma)
            
            # f_u = -(k * v / sigma) * sin_term
            # f_v = -(k * u / sigma) * sin_term
            # Magnitude = sqrt(f_u^2 + f_v^2 + 1)
            
            return np.sqrt(
                ((k * v / sigma) * sin_term)**2 + 
                ((k * u / sigma) * sin_term)**2 + 
                1
            )

        jac_num_value = always_redraw(lambda: DecimalNumber(
            get_jac_mag(),
            num_decimal_places=3,
            font_size=8
        ).next_to(jac_label, RIGHT, buff=0.1))

        # 7. ADD TO SCENE
        # 2D elements go in fixed_in_frame
        self.add_fixed_in_frame_mobjects(uv_frame, patch_2d, vec_u_2d, vec_v_2d, jac_label, jac_num_value, formula_stack)
        
        # 3D elements go in world space
        # self.add(axes_3d, surface, patch_3d, vec_u_3d, vec_v_3d)
        self.add(three_d_elements)
        
        # 8. ANIMATE
        rt = 10
        self.play(u_tracker.animate.set_value(1.0), v_tracker.animate.set_value(1.0), run_time=rt)
        self.play(v_tracker.animate.set_value(0.0), run_time=rt)
        self.wait(5)