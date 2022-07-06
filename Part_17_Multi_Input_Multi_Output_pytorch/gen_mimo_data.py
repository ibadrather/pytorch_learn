"""
    Author: Ibad Rather, ibad.rather.ir@gmail.com

    I will generate data based on the relations
        Inputs: 13p,  24q, 9r, 2s, 34t, 19u 
        Outputs: p/1.5, q/1.5, r/1.5, (s+t+u)/5

    This is done so that some relationship exists.

    Also you can see one input is associated with only one output.
    This is what I want here. 
    And one output is associated with 3 inputs
"""
import numpy as np
from numpy.random import rand
import pandas as pd

n = 100000
def gen_mimo_data(n=10000):
    # Multi-variate-Inputs
    p, q, r, s, t, u = 13*rand(n), 24*rand(n), 9*rand(n), \
                    2*rand(n), 34*rand(n), 19*rand(n)

    # Multi-variate Outputs
    a, b, c, d = p/1.5, q/1.5, r/1.5, (s+t+u) / 5
    data = pd.DataFrame(
        dict(
            i_p = p,
            i_q = q,
            i_r = r,
            i_s = s,
            i_t = t,
            i_u = u,
            o_a = a,
            o_b = b,
            o_c = c,
            o_d = d
        )
    )
    data.to_csv("mimo_data.csv", index=False)

if __name__ =="__main__":
    gen_mimo_data(n=1000000)