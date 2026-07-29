from manim import *
import numpy as np

###############################################################
# Example 5 from the lecture notes: z = sqrt(4 - x^2 - y^2), the upper
# hemisphere of radius R_MAX. Level curves f(x,y) = k are exact circles
# of radius sqrt(R_MAX^2 - k^2), so the horizontal trace and its
# projection can be computed analytically (no marching-squares needed).
R_MAX = 2.0

def radius_at_level(k):
    return np.sqrt(np.clip(R_MAX**2 - k**2, 0, None))
###############################################################


class LevelCurvesHemisphere(ThreeDScene):
    def construct(self):
        z_level = ValueTracker(0.02)

        # NOTE: x,y,z must have the same range/length ratio (here 1 unit-per-
        # value on every axis), or the surface renders visually stretched.
        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-1, 3, 1],
            x_length=6, y_length=6, z_length=4,
        )

        # Spherical (phi, theta) parametrization, not (r, theta): equal steps
        # in the polar angle phi naturally cluster more sample points near
        # the equator (phi=pi/2), exactly where the surface is steepest.
        # Linear radius sampling under-resolved that region and made the
        # dome look like a cylinder capped with a dome.
        hemisphere = Surface(
            lambda phi, theta: axes.c2p(
                R_MAX * np.sin(phi) * np.cos(theta),
                R_MAX * np.sin(phi) * np.sin(theta),
                R_MAX * np.cos(phi),
            ),
            u_range=[0, PI / 2], v_range=[0, TAU],
            resolution=(18, 36), fill_opacity=0.6,
            checkerboard_colors=False, fill_color=BLUE_D,
            stroke_width=0.3, stroke_opacity=0.2,
            should_make_jagged=True,
        )

        # The sweeping plane z = k: a flat rectangle with a circular hole
        # (Cutout) matching the dome's cross-section at height k. Cairo's 3D
        # renderer sorts whole faces by depth (no real z-buffer), which
        # produces streaking artifacts wherever two meshes spatially
        # intersect -- so the hole keeps the plane from ever passing through
        # the dome's interior; it only ever touches the dome along their
        # shared boundary circle. Built flat (z=0) in axes-local coordinates,
        # then mapped into world space via c2p (an affine transform, so it
        # carries the Bezier control points along correctly).
        # A square happens to look like a symmetric diamond (all 4 apparent
        # edges equal length) at this camera angle, which doesn't read as
        # "a rectangle in perspective". A non-square rectangle breaks that
        # symmetry regardless of viewing angle.
        plane_width, plane_height = 8.0, 5.0
        def make_plane():
            k = z_level.get_value()
            r_inner = radius_at_level(k)
            ring = Cutout(
                Rectangle(width=plane_width, height=plane_height),
                Circle(radius=r_inner, num_components=48),
                fill_color=GREY_B, fill_opacity=0.25,
                stroke_color=GREY_B, stroke_width=2, stroke_opacity=0.6,
            )
            ring.apply_function(lambda p: axes.c2p(p[0], p[1], k))
            return ring
        plane = always_redraw(make_plane)

        # Horizontal trace: where the plane cuts the hemisphere
        def make_trace_3d():
            k = z_level.get_value()
            r = radius_at_level(k)
            curve = ParametricFunction(
                lambda t: axes.c2p(r * np.cos(t), r * np.sin(t), k),
                t_range=[0, TAU], color=YELLOW, stroke_width=2,
            )
            return DashedVMobject(curve, num_dashes=36)
        trace_3d = always_redraw(make_trace_3d)

        # Its projection onto the xy-plane: the level curve f(x,y) = k
        def make_level_curve():
            k = z_level.get_value()
            r = radius_at_level(k)
            return ParametricFunction(
                lambda t: axes.c2p(r * np.cos(t), r * np.sin(t), 0),
                t_range=[0, TAU], color=RED, stroke_width=2,
            )
        level_curve = always_redraw(make_level_curve)

        # Dashed connectors linking the trace to its projection
        def make_connectors():
            k = z_level.get_value()
            r = radius_at_level(k)
            lines = VGroup()
            for angle in [0, PI / 2, PI, 3 * PI / 2]:
                top = axes.c2p(r * np.cos(angle), r * np.sin(angle), k)
                bottom = axes.c2p(r * np.cos(angle), r * np.sin(angle), 0)
                lines.add(DashedLine(top, bottom, color=GREY_B, stroke_width=2))
            return lines
        connectors = always_redraw(make_connectors)

        # UI & CAMERA
        self.set_camera_orientation(phi=70 * DEGREES, theta=-65 * DEGREES, zoom=0.35)

        # NOTE: fixed-in-frame mobjects placed too far from the origin (e.g.
        # via to_corner()) silently vanish -- an apparent Cairo 3D-camera
        # clipping quirk, unrelated to `zoom`. Empirically safe up to
        # roughly 2 units from origin, so this stays close in rather than
        # pinned to the true frame corner.
        z_label = MathTex("z = k = ", font_size=20).shift(DOWN * 1.0)
        z_value = always_redraw(lambda: DecimalNumber(
            z_level.get_value(), num_decimal_places=2, font_size=20
        ).next_to(z_label, RIGHT, buff=0.15))

        self.add(axes, hemisphere, plane, trace_3d, level_curve, connectors)
        self.add_fixed_in_frame_mobjects(z_label, z_value)

        # ANIMATE: sweep the plane up, then back down
        rt = 5 * 1.33
        self.play(z_level.animate.set_value(R_MAX - 0.02), run_time=rt)
        self.play(z_level.animate.set_value(0.02), run_time=rt)
        self.wait()
