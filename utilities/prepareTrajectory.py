import numpy as np

def prepare_trajectory(csv_path):
    """
    inputs:
        csv_path: string
            path to a csv file containing a trajectory in the
            below specified format

    The csv file should have 4 columns with values for
    [time, position, velocity, torque] respectively.
    The values shopuld be separated with a comma.
    Each row in the file is one timestep. The number of rows can vary.
    The values are assumed to be in SI units, i.e. time in s, position in rad,
    velocity in rad/s, torque in Nm.
    The first line in the csv file is reserved for comments
    and will be skipped during read out.

    Example:

        # time, position, velocity, torque
        0.00, 0.00, 0.00, 0.10
        0.01, 0.01, 0.01, -0.20
        0.02, ....

    """
    # load trajectories from csv file
    trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
    des_time_list = trajectory.T[0].T                       # desired time in s
    des_pos_list = trajectory.T[1].T               # desired position in radian
    des_vel_list = trajectory.T[2].T             # desired velocity in radian/s
    des_tau_list = trajectory.T[3].T                     # desired torque in Nm

    n = len(des_time_list)
    t = des_time_list[n - 1]
    dt = round((des_time_list[n - 1] - des_time_list[0]) / n, 3)

    # create 4 empty numpy array, where measured data can be stored
    meas_time_list = np.zeros(n)
    meas_pos_list = np.zeros(n)
    meas_vel_list = np.zeros(n)
    meas_tau_list = np.zeros(n)
    vel_filt_list = np.zeros(n)

    data_dict = {"des_time_list": des_time_list,
                 "des_pos_list": des_pos_list,
                 "des_vel_list": des_vel_list,
                 "des_tau_list": des_tau_list,
                 "meas_time_list": meas_time_list,
                 "meas_pos_list": meas_pos_list,
                 "meas_vel_list": meas_vel_list,
                 "meas_tau_list": meas_tau_list,
                 "vel_filt_list": vel_filt_list,
                 "n": n,
                 "dt": dt,
                 "t": t}
    return data_dict