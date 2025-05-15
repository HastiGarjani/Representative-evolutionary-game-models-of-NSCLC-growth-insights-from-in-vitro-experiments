from .CancerModel import CancerModel
import numpy as np

class Logistic6(CancerModel):

    def f_dot(self, xs, ps):
        ro1, ro2, K1, K2, aSR, aRS, Lambda_m = ps
        S, R, c_drug = xs
        out1 = ro1*(1-(S+aSR*R)/K1)*S - Lambda_m*c_drug*S
        out2 = ro2*(1-(aRS*S+R)/K2)*R
        return [out1, out2]