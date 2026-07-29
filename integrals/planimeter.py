from manim import *
import numpy as np

class Planimeter(MovingCameraScene):
    def construct(self):
        # 1. Setup Axes - Slightly larger range to show context
        axes = Axes(
            x_range=[-3, 5, 1], y_range=[-3, 5, 1],
            x_length=8, y_length=8,
            axis_config={"include_tip": True}
        ).shift(RIGHT * 2)

        center_x = 1
        center_y = 1
        curve = ParametricFunction(
            lambda t: axes.c2p(
                center_x + 3*(np.cos(t))**5, 
                center_y + 3*(np.sin(t))**3, 
                0
            ),
            t_range=[0, TAU], color=WHITE
        )

        # 3. ValueTracker for animation
        t_tracker = ValueTracker(0)

        # 4. Tracer & Sweep Arm (The x component)
        tracer = Dot(color=YELLOW)
        tracer.add_updater(lambda d: d.move_to(curve.point_from_proportion(t_tracker.get_value() / TAU)))
        
        sweep_arm = always_redraw(lambda: Line(
            start=axes.c2p(0, axes.p2c(tracer.get_center())[1]), 
            end=tracer.get_center(),
            color=BLUE, stroke_width=4
        ))

        # 5. Live Integral Display
        self.accumulated_area = 0
        self.last_y = center_y
        
        area_display = VGroup(
            MathTex(r"\iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dA  = \oint_{\partial D} x \, dy = ", color=WHITE),
            DecimalNumber(0, num_decimal_places=3, color=YELLOW)
        ).arrange(RIGHT).shift(UP * 2).shift(LEFT * 5)

        def update_area(mob):
            curr_x, curr_y = axes.p2c(tracer.get_center())[:2]
            dy = curr_y - self.last_y
            self.accumulated_area += curr_x * dy
            self.last_y = curr_y
            mob[1].set_value(self.accumulated_area)

        area_display.add_updater(update_area)

        # 6. Optimized Swept Area (The "Planimeter Footprint")
        self.curve_points = []
        self.axis_points = []
        swept_shape = VMobject().set_fill(BLUE, opacity=0.5).set_stroke(width=0)

        def update_swept_area(mob):
            curr_p = tracer.get_center()
            y_axis_p = axes.c2p(0, axes.p2c(curr_p)[1])
            
            # Only start drawing after we have moved away from the start
            if t_tracker.get_value() > 0.01: 
                self.curve_points.append(curr_p)
                self.axis_points.append(y_axis_p)
                
                all_points = [*self.curve_points, *reversed(self.axis_points)]
                mob.set_points_as_corners(all_points)

        swept_shape.add_updater(update_swept_area)

        # 7. CAMERA ZOOM: Setting width to 22 to see everything
        self.camera.frame.set(width=22)

        # Animation Sequence
        self.add(axes)
        self.play(Create(curve), run_time=2)
        self.add(swept_shape, curve, sweep_arm, tracer, area_display)
       
        rt = 30
        self.play(
            t_tracker.animate.set_value(TAU),
            run_time=rt,
            rate_func=linear
        )

        final_value = self.accumulated_area / PI
        result_label = MathTex(
            r"\text{Total Area } \approx ", 
            f"{final_value:.2f}", 
            r"\pi", 
            color=YELLOW
        ).next_to(area_display, DOWN, buff=0.5)

        self.play(Write(result_label))
        self.play(Indicate(result_label), Indicate(area_display[1]))
        
        self.wait(5)