import numpy as np

class ODESolver() :
    def __init__(self, f) :
        self.f = f

    def set_initial_conditions(self, U0) :
        if isinstance(U0, (int, float)) :
            self.number_equations = 1
            U0 = float(U0)
        else :
            U0 = np.asarray(U0)
            self.number_equations = U0.size
        self.U0 = U0

    def solve(self, time) :
        self.t = np.asarray(time)
        n = self.t.size

        self.u = np.zeros((n, self.number_equations))
        self.u[0, :] = self.U0

        for k in range(n-1) :
            self.k = k
            self.u[k+1] = self.advance()
        return self.u, self.t

    def advance(self) :
        raise NotImplementedError

class ForwardEuler(ODESolver) :
    def advance(self) :
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k + 1] - t[k]
        return u[k, :] + dt * f(u[k, :], t[k])
