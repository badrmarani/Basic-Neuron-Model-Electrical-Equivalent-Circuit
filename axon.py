import numpy as np
from odesolver import ForwardEuler
from scipy.signal import square

import matplotlib.pyplot as plt

class Axon() :
    def __init__(
        self,
        R1 = 1e8,
        R2 = 1e6,
        C = 1e-10,
        V_s = .1,
        N = 1
    ) :
        self.R1 = R1
        self.R2 = R2
        self.C = C

        self.initial_conditions = np.zeros(N)

        if isinstance(V_s, (int, float)) :
            self.V_s = lambda t : V_s
            self.initial_conditions[0] = V_s
        elif callable(V_s) :
            self.V_s = V_s
            self.initial_conditions[0] = V_s(0)

    def dV(self, u_left, u_right) :
        I2 = (u_left - u_right) / self.R2
        I1 = (u_right) / self.R1

        dV = (I2-I1) / self.C
        return dV

    def __call__(self, u, t) :
        new_u = np.zeros_like(u)

        new_u[0] = self.dV( self.V_s(t), u[0] )
        for i in range(len(u)-1) :
            new_u[i+1] += self.dV( u[i], u[i+1] )

        return new_u

if __name__ == '__main__' :
    dt = 1e-6
    T = .05

    number_time_steps = int(T/dt)
    time_steps = np.linspace(0, T, number_time_steps)

    V_s = lambda t : .1 * np.heaviside( square(t * 2 * np.pi * 200), 1) # * np.heaviside( .01 - t, 1)

    model = Axon(
        V_s = V_s,
        N = 100
    )

    solver = ForwardEuler(model)
    solver.set_initial_conditions(model.initial_conditions)

    u, t = solver.solve(time_steps)

    # for i in range(0, number_time_steps, 1000) :
    #     plt.plot(u[i, :], label = f"time step = {i}")

    # plt.plot(t, u[-1, :], label = 'last time step', color = 'blue')

    plt.plot(t, V_s(t), label = 'Input signal')
    plt.plot(t, u[:, -1], label = 'Output signal')

    plt.legend()
    plt.show()
