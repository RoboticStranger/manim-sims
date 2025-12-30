from manim import *
import numpy as np

# Exponential z = k exp(-(x^2+y^2)/sigma)
#
#
class JacobianTransformation(ThreeDScene):
    def construct(self):
        # 1. Trackers & Constants
        u_tracker = ValueTracker(0.0)
        v_tracker = ValueTracker(0.0)
        ds = 0.5
        
        # INDEPENDENT SETTINGS
        # Change these values freely; they won't break the other frame.
        uv_scale = 0.18
        uv_pos = LEFT * 1.5  # + UP * 0.5

        # 2. LEFT FRAME: UV SPACE
        axes_2d = NumberPlane(
            x_range=[-2, 2, 1], 
            y_range=[-2, 2, 1], 
            x_length=6.66,      # (4 units * 10/6)
            y_length=6.66,
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
        sigma = 2.0
        def get_xyz_pt(u, v):
            z = k * np.exp(-(u**2 + v**2)/sigma)
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
            get_uv_pt(u_tracker.get_value() + ds, v_tracker.get_value()),
            color=YELLOW, buff=0, stroke_width=4
        ))
        
        vec_v_2d = always_redraw(lambda: Arrow(
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_uv_pt(u_tracker.get_value(), v_tracker.get_value() + ds),
            color=PINK, buff=0, stroke_width=4
        ))

        # 1. Helper to compute the tangent plane points (Linearization)
        def get_tangent_data():
            u = u_tracker.get_value()
            v = v_tracker.get_value()
            z = k * np.exp(-(u**2 + v**2) / sigma)
            
            # Partial derivatives (slopes at the current point)
            f_u = z * (-2 * u / sigma)
            f_v = z * (-2 * v / sigma)
            
            # Point 1: The start point on the surface
            p_start = axes_3d.c2p(u, v, z)
            
            # Point 2: Move ds along the tangent slope in U
            p_u_end = axes_3d.c2p(u + ds, v, z + f_u * ds)
            
            # Point 3: Move ds along the tangent slope in V
            p_v_end = axes_3d.c2p(u, v + ds, z + f_v * ds)
            
            # Point 4: The far corner of the parallelogram
            p_corner = axes_3d.c2p(u + ds, v + ds, z + f_u * ds + f_v * ds)
            
            return p_start, p_u_end, p_v_end, p_corner

        # 2. Updated Arrows (using the linearized end-points)
        vec_u_3d = always_redraw(lambda: Arrow(
            get_tangent_data()[0], get_tangent_data()[1], 
            color=YELLOW, buff=0, stroke_width=4
        ))

        vec_v_3d = always_redraw(lambda: Arrow(
            get_tangent_data()[0], get_tangent_data()[2], 
            color=PINK, buff=0, stroke_width=4
        ))

        # 3. Updated Patch (order: start -> u_end -> corner -> v_end)
        patch_3d = always_redraw(lambda: Polygon(
            get_tangent_data()[0], 
            get_tangent_data()[1], 
            get_tangent_data()[3], 
            get_tangent_data()[2],
            fill_opacity=0.8, fill_color=GREEN, stroke_color=WHITE, stroke_width=1
        ))

        # Define the full vector first
        def get_normal_vector():
            u = u_tracker.get_value()
            v = v_tracker.get_value()
            # These are the components: grad z = <-z_u, -z_v, 1> * ds
            # For z = k * exp(-(u^2+v^2)/sigma)
            comp_x = (2 * k * u / sigma) * np.exp(-(u**2 + v**2) / sigma) * ds
            comp_y = (2 * k * v / sigma) * np.exp(-(u**2 + v**2) / sigma) * ds
            return np.array([comp_x, comp_y, ds])
        
        # Use it for the Arrow
        vec_n_3d = always_redraw(lambda: Arrow(
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()),
            get_xyz_pt(u_tracker.get_value(), v_tracker.get_value()) + (
                axes_3d.c2p(*get_normal_vector()) - axes_3d.c2p(0, 0, 0)
            ),
            color=WHITE, buff=0, stroke_width=2
        ))

        # Group all 3D elements
        three_d_elements = Group(axes_3d, surface, patch_3d, vec_u_3d, vec_v_3d, vec_n_3d)

        # 6. UI & CAMERA
        self.set_camera_orientation(phi=80 * DEGREES, theta=60 * DEGREES)
        
        jac_label = MathTex(
            r"\|\mathbf{r}_u \times \mathbf{r}_v\|\; dS = ", 
            font_size=8
        )

        # Define the math formulas
        surf_eq = MathTex(r"\mathbf{r}(u, v) = \langle u, v, k e^{-(u^2 + v^2)/\sigma} \rangle", font_size=8)
        partial_u = MathTex(r"\mathbf{r}_u = \langle 1, 0, -\frac{2ku}{\sigma} e^{-(u^2 + v^2)/\sigma} \rangle", font_size=8)
        partial_v = MathTex(r"\mathbf{r}_v = \langle 0, 1, -\frac{2kv}{\sigma} e^{-(u^2 + v^2)/\sigma} \rangle", font_size=8)
    
        # Arrange them in a column
        formula_stack = VGroup(surf_eq, partial_u, partial_v, jac_label).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # Position the stack above your existing jac_label
        formula_stack.next_to(axes_3d, RIGHT, buff=0.1)

        jac_num_value = always_redraw(lambda: DecimalNumber(
            np.linalg.norm(get_normal_vector()), # Re-uses the vector to find length
            num_decimal_places=3,
            font_size=8
        ).next_to(jac_label, RIGHT, buff=0.1))

        # 7. ADD TO SCENE
        # 2D elements go in fixed_in_frame
        self.add_fixed_in_frame_mobjects(uv_frame, patch_2d, vec_u_2d, vec_v_2d, jac_label, jac_num_value, formula_stack)
        
        # 3D elements go in world space
        self.add(three_d_elements)
        
        # 8. ANIMATE
        rt = 10
        self.play(u_tracker.animate.set_value(1.0), v_tracker.animate.set_value(1.0), run_time=rt)
        self.play(u_tracker.animate.set_value(-1.0), run_time=rt)
        self.wait(5)