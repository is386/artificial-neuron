import matplotlib.pyplot as plt


def plot_3D(x, y, z, x_lab="", y_lab="", z_lab=""):
    ax = plt.axes(projection='3d')
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_zlabel(z_lab)
    ax.plot3D(x, y, z)
    plt.show()


def gradient_descent(lr, tc):
    t1 = 0
    t2 = 0
    t1_vals = []
    t2_vals = []
    obj_vals = []

    for i in range(tc):
        t1_vals.append(t1)
        t2_vals.append(t1)

        # Calculates the objective value.
        J = obj_func(t1, t2)
        obj_vals.append(J)

        # Calculates the gradient.
        d_t1, d_t2 = calc_gradients(t1, t2)

        # Calculates the theta values.
        t1 = t1 - (lr * d_t1)
        t2 = t2 - (lr * d_t2)

    return t1_vals, t2_vals, obj_vals


def obj_func(t1, t2):
    return (t1 + t2 - 2) ** 2


def calc_gradients(t1, t2):
    d1 = 2 * (t1 + t2 - 2)
    d2 = 2 * (t1 + t2 - 2)
    return d1, d2


lr = 0.05
tc = 100
t1, t2, J = gradient_descent(lr, tc)
plot_3D(t1, t2, J)
