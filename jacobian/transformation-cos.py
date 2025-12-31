from manim import *
import numpy as np

###############################################################
SIM_TIME = 5
#
# Exponential z = k exp(-(x^2+y^2)/sigma)
#
k = 2.0
sigma = 3.0
def f_vals(u, v):
    # function value
    z = k * np.cos(u*v/sigma)
    # Partial Derivatives
    f_u = - k * v / sigma * np.sin(u*v/sigma)
    f_v = - k * u / sigma * np.sin(u*v/sigma)
    return z, f_u, f_v
#

# the u-v range to plot
U_RANGE = [-2,2]
V_RANGE = [-2,2]

# the 3-d ranges for the 3d-axes
# NOTE:  x,y,z must have same range/length ratio for the normal vector to look orthogonal
X_RANGE=[-3, 3, 1]
Y_RANGE=[-3, 3, 1]
Z_RANGE=[-2, 4, 1]   

# u-v area element size
ds = 1.0

# Define the math formulas
const_eq = MathTex(r"dS = "+str(ds)+ ", k = "+str(k)+ ", \sigma = "+str(sigma), font_size=8)
surf_eq = MathTex(r"\mathbf{r}(u, v) = \langle u, v, k\cos(uv/\sigma) \rangle", font_size=8)
partial_u = MathTex(r"\mathbf{r}_u = \langle 1, 0, -\frac{kv}{\sigma}\sin(uv/\sigma) \rangle", font_size=8)
partial_v = MathTex(r"\mathbf{r}_v = \langle 0, 1, -\frac{ku}{\sigma}\sin(uv/\sigma) \rangle", font_size=8)
#
##############################################################


class JacobianTransformation(ThreeDScene):
    def construct(self):
        # Trackers & Constants
        u_tracker = ValueTracker(0.0)
        v_tracker = ValueTracker(0.0)
        
        # INDEPENDENT SETTINGS
        uv_scale = 0.15
        uv_pos = LEFT * 1.5  # + UP * 0.5

        # LEFT FRAME: UV SPACE
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
        
        # u,v axis labels
        u_lab = MathTex("u", font_size=10).next_to(
            axes_2d.x_axis.get_end(), # Get the tip of the axis
            DOWN,                     # Place it below
            buff=0.1                  # Small gap
        )
        v_lab = MathTex("v", font_size=10).next_to(
            axes_2d.y_axis.get_end(), # Get the tip of the axis
            LEFT,                     # Place it to the left
            buff=0.1                  # Small gap
        )
        uv_frame = VGroup(axes_2d, u_lab, v_lab)

        def get_uv_pt(u, v):
            return axes_2d.c2p(u, v)

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

        # RIGHT FRAME: XYZ SPACE
        xyz_scale = 0.2

        axes_3d = ThreeDAxes(
            x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE,   
            # NOTE:  x,y,z must have same range/length ratio for the normal vector to look orthogonal
            x_length=10, y_length=10, z_length=10
        ).scale(xyz_scale)

        # Centering on the screen:
        # This ensures the physical center of the axis (at z=1) is in the middle of the frame.
        axes_3d.move_to(ORIGIN)

        # SURFACE
        def get_xyz_pt(u, v):
            z = f_vals(u,v)[0]
            return axes_3d.c2p(u, v, z)

        surface = Surface(
            lambda u, v: get_xyz_pt(u, v),
            u_range=U_RANGE, v_range=V_RANGE,
            resolution=(30, 30), fill_opacity=0.3,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )

        # THE UNIFIED BRAIN
        def get_surface_geometry():
            u = u_tracker.get_value()
            v = v_tracker.get_value()
            
            func_vals = f_vals(u,v)
            # Surface Height
            z = func_vals[0]
            # Slopes (Partial Derivatives)
            f_u = func_vals[1]
            f_v = func_vals[2]
            
            # Points in World Space
            p_start = axes_3d.c2p(u, v, z)
            p_u_end = axes_3d.c2p(u + ds, v, z + f_u * ds)
            p_v_end = axes_3d.c2p(u, v + ds, z + f_v * ds)
            p_corner = axes_3d.c2p(u + ds, v + ds, z + f_u * ds + f_v * ds)
            
            # Normal Vector: < -f_u, -f_v, 1 >
            # We must scale the direction properly relative to the axis units
            raw_n = np.array([-f_u, -f_v, 1]) * ds
            # The displacement vector in Manim world-space
            n_dir = axes_3d.c2p(*raw_n) - axes_3d.c2p(0, 0, 0)
            p_n_end = p_start + n_dir
            
            return p_start, p_u_end, p_v_end, p_corner, p_n_end, raw_n

        # SYNCHRONIZED OBJECTS
        vec_u_3d = always_redraw(lambda: Arrow(
            get_surface_geometry()[0], get_surface_geometry()[1],
            color=YELLOW, buff=0, stroke_width=4
        ))

        vec_v_3d = always_redraw(lambda: Arrow(
            get_surface_geometry()[0], get_surface_geometry()[2],
            color=PINK, buff=0, stroke_width=4
        ))

        vec_n_3d = always_redraw(lambda: Arrow(
            get_surface_geometry()[0], get_surface_geometry()[4],
            color=WHITE, buff=0, stroke_width=4
        ))

        patch_3d = always_redraw(lambda: Polygon(
            get_surface_geometry()[0], 
            get_surface_geometry()[1], 
            get_surface_geometry()[3], 
            get_surface_geometry()[2],
            fill_opacity=0.8, fill_color=GREEN, stroke_color=WHITE, stroke_width=1
        ))

        # Group all 3D elements
        three_d_elements = Group(axes_3d, surface, patch_3d, vec_u_3d, vec_v_3d, vec_n_3d)

        # UI & CAMERA
        self.set_camera_orientation(phi=70 * DEGREES, theta= 60 * DEGREES)

        # the Jacobian 
        jac_label = MathTex(r"\|\mathbf{r}_u \times \mathbf{r}_v\|\; dS = ", font_size=8)
        jac_num_value = always_redraw(lambda: DecimalNumber(
            np.linalg.norm(get_surface_geometry()[5]), # Re-uses the vector to find length
            num_decimal_places=3,
            font_size=8
        ).next_to(jac_label, RIGHT, buff=0.1))

        # Arrange formulas in a column to the right of the 3D plot
        formula_stack = VGroup(const_eq, surf_eq, partial_u, partial_v, jac_label).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        formula_stack.next_to(axes_3d, RIGHT, buff=0.1)

        # ADD TO SCENE
        # 2D elements go in fixed_in_frame
        self.add_fixed_in_frame_mobjects(uv_frame, patch_2d, vec_u_2d, vec_v_2d, jac_label, jac_num_value, formula_stack)
        
        # 3D elements go in world space
        self.add(three_d_elements)
        
        # ANIMATE
        rt = SIM_TIME
        self.play(u_tracker.animate.set_value(1.0), v_tracker.animate.set_value(1.0), run_time=rt)
        self.play(u_tracker.animate.set_value(-1.0), run_time=rt)
        self.play(u_tracker.animate.set_value(0.0), v_tracker.animate.set_value(0.0), run_time=rt)
        self.wait(5)