from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

@dataclass
class CancerModel:
    
    def f_rk(self, xs, ps):
        
        try:
            name = ps.keys()
            param_rk = []
            for key in name:
                param_rk.append(ps[key].value)
        except:
            param_rk = ps

        h = 0.05
        S, R, c_drug = xs
        [k1S, k1R] = np.array(self.f_dot(xs, param_rk))
        [k2S, k2R] = np.array(self.f_dot(np.hstack(([S,R]+np.multiply(h/2,[k1S, k1R]), c_drug)), param_rk))
        [k3S, k3R] = np.array(self.f_dot(np.hstack(([S,R]+np.multiply(h/2,[k2S, k2R]), c_drug)), param_rk))
        [k4S, k4R] = np.array(self.f_dot(np.hstack(([S,R]+np.multiply(h,[k3S, k3R]), c_drug)), param_rk))
        return [k1S+k2S+k2S+k3S+k3S+k4S, k1R+k2R+k2R+k3R+k3R+k4R]

    @abstractmethod
    def f_dot(self, c):
        pass

    def eval_fit(self, t_seq, x0, cDrug, ps):
        x_seq = []
        x_seq.append(x0)
        h = 0.05
        for tt in range(len(t_seq)-1):
            x = x_seq[-1] + np.multiply(h/6, self.f_rk(np.hstack((x_seq[-1] , cDrug[tt])), ps))
            x_seq.append(x)
        return x_seq
    
    def residual(self, ps, ts, data, cDrug):
        x0 = data[0]
        model1 = np.array(self.eval_fit(ts, x0, cDrug, ps))
        return np.array(([(model1[:,0] - data[:,0])/np.max(data[:,0]),\
            (model1[:,1] - data[:,1])/np.max(data[:,1])])).ravel()