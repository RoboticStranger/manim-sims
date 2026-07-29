from manim import *
import numpy as np

class GeneralizedLineIntegral(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=22)
        
        # 1. Coordinate Systems
        field_axes = Axes(x_range=[-4, 4], y_range=[-2, 3], x_length=12, y_length=8).shift(UP * 2)
        # Graph x_range now matches alpha tracker (0 to 1)
        graph_axes = Axes(x_range=[0, 1], y_range=[-20, 20], 
                          x_length=12, y_length=4,
                          x_axis_config={"include_ticks": True},
                          y_axis_config={"include_ticks": False}).shift(DOWN * 3)

        # Position X-label at the far right of the X-axis
        x_label = MathTex("t").next_to(graph_axes.x_axis.get_end(), RIGHT, buff=0.2)

        # Position Y-label at the very top of the Y-axis
        y_label = MathTex(r"\mathbf{F}\cdot \mathbf{r}'(t)").scale(0.8).next_to(graph_axes.y_axis.get_top(), UP, buff=0.2)

        self.add(graph_axes, x_label, y_label)

        # 2. DEFINITIONS
        def path_func(t):
            # t is 0 to 1
            tau_t = t * TAU
            return field_axes.c2p(2 * np.cos(tau_t), 1 * np.sin(2 * tau_t), 0)

        def vector_field_func(pos):
            p = field_axes.p2c(pos)
            x, y = p[0], p[1]
            
            # Calculate the magnitude (denominator)
            # Adding 1e-6 prevents division by zero error at the origin
            mag = np.sqrt(x**2 + y**2) + 1e-6
            
            return np.array([x / mag, y / mag, 0])

        def get_velocity(t, dt=0.001):
            p1 = path_func(t)
            p2 = path_func(t + dt if t + dt <= 1 else t - dt)
            v1_coords, v2_coords = field_axes.p2c(p1), field_axes.p2c(p2)
            v1 = np.array([v1_coords[0], v1_coords[1], 0])
            v2 = np.array([v2_coords[0], v2_coords[1], 0])
            return (v2 - v1) / dt if t + dt <= 1 else (v1 - v2) / dt

        def get_dot_product(t):
            return np.dot(vector_field_func(path_func(t))[:2], get_velocity(t)[:2])

        # 3. ValueTracker & Mobjects
        alpha = ValueTracker(0)

        formula = MathTex(r"\int_C \mathbf{F} \cdot d\mathbf{r}").scale(1.0)
        formula.move_to(self.camera.frame.get_corner(UL), aligned_edge=UL).shift(DOWN*0.5 + RIGHT*0.5)

        # Instantaneous Label: F · dr = [Value]
        inst_label = MathTex(r"\mathbf{F} \cdot \mathbf{r}'(t) = ").scale(1.1)
        inst_label.next_to(formula, DOWN, aligned_edge=LEFT).shift(RIGHT*0.5)
        
        inst_val = always_redraw(lambda: DecimalNumber(
            get_dot_product(alpha.get_value()),
            num_decimal_places=2,
            include_sign=True,
            color=PINK
        ).next_to(inst_label, RIGHT))

    
        def calculate_cumulative(current_t):
            if current_t <= 0: return 0
            steps = 100 # Increase for more precision
            t_values = np.linspace(0, current_t, steps)
            dt = current_t / steps
            # Simple Riemann sum
            return sum(get_dot_product(t) * dt for t in t_values)
    
        '''
        def calculate_cumulative(current_t):
            if current_t < 0.01: return 0
            t_samples = np.linspace(0, current_t, 100)
            # Vectorized calculation of dot products
            dots = [get_dot_product(t) for t in t_samples]
            return np.sum(dots) * (current_t / 100)
        '''

        # Cumulative Label: Total Work = [Value]
        # Using \sum or W to represent the accumulation
        cum_label = MathTex(r"\int_C \mathbf{F} \cdot \mathbf{r}'(t)\; dt = ").scale(0.9)
        cum_label.next_to(inst_label, DOWN, aligned_edge=LEFT)
        
        cum_val = always_redraw(lambda: DecimalNumber(
            calculate_cumulative(alpha.get_value()),
            num_decimal_places=2,
            include_sign=True,
            color=YELLOW
        ).next_to(cum_label, RIGHT))
        
        # vector_field = ArrowVectorField(vector_field_func, x_range=[-6, 6, 2], y_range=[-3, 6, 2])
        vector_field = ArrowVectorField(
            vector_field_func, 
            color=RED,
            # Step size of 2.0 makes it sparser (default is usually smaller)
            x_range=[-3, 3, 0.5], 
            y_range=[-1, 4, 0.5],
            # scale_factor makes the arrows longer/larger
            # Use length_func to scale the size of the arrows
            # This multiplier (0.5) adjusts the overall visual length
            length_func=lambda norm: 0.5 * norm,
            vector_config={
                "stroke_width": 6,        # Makes the body of the arrow thicker
                "max_tip_length_to_length_ratio": 0.25, # Ensures tips are visible
            }
        )    
        path = ParametricFunction(path_func, t_range=[0, 1], color=GREEN)
 
        dot_product_scale = 1.0
        # We create the FULL graph but don't add it to the scene yet
        full_graph = graph_axes.plot(
            lambda t: dot_product_scale * np.dot(vector_field_func(path_func(t))[:2], get_velocity(t)[:2]),
            x_range=[0, 1], color=PINK
        )

        # 4. Sync Animation Logic
        # We still use ValueTracker for the Dot and Vector
        alpha = ValueTracker(0)
        dot = always_redraw(lambda: Dot(color=PINK).move_to(path_func(alpha.get_value())))
        
        dr_vector = always_redraw(lambda: Arrow(
            start=dot.get_center(),
            end=dot.get_center() + get_velocity(alpha.get_value()) * 0.1, 
            buff=0, color=BLUE
        ))

        '''
        # --- Legend Construction ---
        # 1. Force Legend (Using TEAL to match your vector field)
        force_arrow = Arrow(LEFT, RIGHT, color=RED, buff=0).scale(0.5)
        force_text = MathTex(r"= \text{Force } (\mathbf{F})", font_size=34).next_to(force_arrow, RIGHT)
        force_legend = VGroup(force_arrow, force_text)
        # 2. Velocity Legend (Green dr vector)
        vel_arrow = Arrow(LEFT, RIGHT, color=GREEN, buff=0).scale(0.5)
        vel_text = MathTex(r"= \text{Velocity } (d\mathbf{r})", font_size=34).next_to(vel_arrow, RIGHT)
        vel_legend = VGroup(vel_arrow, vel_text)
        # 3. Combine and Position
        legend = VGroup(force_legend, vel_legend).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        # Position in the bottom left corner
        legend.to_corner(DL, buff=1.0).shift(LEFT * 8)
        '''

        '''
        # Coordinate Label Logic
        # We create a label that maps the math values to a string
        coord_label = always_redraw(lambda: 
            VGroup(
                # Background bubble for readability
                RoundedRectangle(height=0.8, width=2.5, corner_radius=0.1)
                    .set_fill(BLACK, opacity=0.8).set_stroke(PINK, 1),
                # The actual math text
                MathTex(
                    f"({alpha.get_value():.2f}, {calculate_cumulative(alpha.get_value()):.2f})",
                    font_size=30, color=WHITE
                )
            )
            # Position it above the "current" point on the graph
            .next_to(graph_axes.c2p(alpha.get_value(), calculate_cumulative(alpha.get_value())), UP)
        )
        '''

        # 5. The Render
        self.add(field_axes, vector_field, path, inst_label, inst_val, cum_label, cum_val)
        # self.play(FadeIn(legend)) # Introduce the legend first
        self.add(dot, dr_vector)
        # self.add(coord_label)
        
        # SYNC: We play the Tracker and the Create animation at the same time
        self.play(
            alpha.animate.set_value(1),
            Create(full_graph),
            run_time=60, 
            rate_func=linear # CRITICAL: Both must be linear to stay in sync
        )
        self.wait(10)