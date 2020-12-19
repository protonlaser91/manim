import math
import os
from pathlib import Path

from scipy.integrate import odeint
from scipy.spatial.transform import Rotation as R

from manim import *

FLAGS = "-pp --media_dir ../manimStuff/media"

class LongJohn(Scene):

    bruh = {
         "axis_config": {
            "stroke_color": LIGHT_GREY,
            "stroke_width": 2,
            "include_ticks": True,
            "include_tip": False,
        },

        "x_axis_config": {"x_min": -0.5, "x_max":2,
                          "unit_size":4,
                          "numbers_to_show":np.arange(0+0.5,2.5,0.5),
                          "tick_frequency": 0.5,
                          'decimal_number_config':{
                              "num_decimal_places": 1,}
                          },
        "y_axis_config": {"label_direction": LEFT,

                          "x_min": -0.5, "x_max":2,
                          "unit_size":4,
                          "numbers_to_show": np.arange(0 + 0.5, 2.5, 0.5),
                          "tick_frequency": 0.5,
                          'decimal_number_config': {
                              "num_decimal_places": 1,
                          },
                          },
        "center_point": 2*DOWN + 4*LEFT,
    }

    def arrowToDot(self, t):
        t.put_start_and_end_on(self.n.c2p(*ORIGIN),self.arcCurve.evaluate(self.degreeMeasure.get_value()))

    def placeDotTangent(self,t):
        point = self.arcCurve.evaluate(self.degreeMeasure.get_value())
    #    vd = normalize(np.array([deriv(x,2) for x in ]))
        t.next_to(point, buff=0, direction=deriv(self.arcCurve.function,t=self.degreeMeasure.get_value(),n=2))

    def get_points(self,n=632):
        def f(s, t):
            s0 = s[0]
            s1 = s[1]
            s2 = 9.8 * np.cos(s0)
            return [s1, s2]

        init = 0, 0
        t = np.linspace(0, 1, n)
        return [self.n.c2p(*a) for a in np.column_stack((t, odeint(f, init, t)[:, 0], np.zeros(len(t))))]


    def construct(self):
        self.n = Axes(**self.bruh)
        self.play(Write(self.n),Write(self.n.get_coordinate_labels()))
        self.degreeMeasure = ValueTracker(0) #

        self.arcCurve = self.n.get_parametric_curve(lambda t: [1-np.cos(t),1-np.sin(t),0],t_min=0,t_max=np.pi/2).set_color(BLUE_C)

        self.diffyQsoln = ParametricFunction(lambda t: [0,0,0]).set_points_smoothly(self.get_points(3))
        self.d = Dot(self.n.c2p(*self.arcCurve.evaluate(0))).set_color(ORANGE).add_updater(lambda t: self.placeDotTangent(t))


        self.play(ShowCreation(self.arcCurve),ShowCreation(self.diffyQsoln))

           #       GrowArrow(
          #            Arrow(self.arcCurve.evaluate(0),deriv(self.arcCurve.function,t=self.degreeMeasure.get_value(),n=2),).set_color(RED_C).add_updater(
           #               lambda t: t.put_start_and_end_on(self.d.get_center(),self.d.get_center() +
                            #                               self.arcCurve.vector_derivative(t=self.degreeMeasure.get_value())) #ok....howdowedothis
          #            )
         #         ))


        self.wait()


        self.play(GrowFromCenter(self.d),run_time=2)

        self.wait()



if __name__ == "__main__":
    script_name = f"{Path(__file__).resolve()}"
    os.system(f"manim {script_name} {FLAGS}")

def sigmoid(x):
  return [1 / (1 + math.exp(-a)) for a in x]

def normalize(x):
    return x / np.sqrt((x ** 2).sum())

def deriv(func,t,dt=0.01,n=1):
  derivSum = np.array([0,0,0],dtype=float)
  for k in range(0,n+1):
    derivSum += np.array([(-1)**(k+n) * math.comb(n,k) * a for a in func(t+k*dt)])

  return 1/dt**n * derivSum