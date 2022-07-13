import os
import numpy as np

def saveFunnel(rho, S_t, time, N, max_dt):

    S = S_t.value(time[0]).flatten()
    for i in range(1,len(time)):
        S = np.vstack((S,S_t.value(time[i]).flatten()))

    log_dir = "log_data/sos"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_data = np.vstack((rho, np.array(S).T))

    csv_path = os.path.join(log_dir, f"funnel{max_dt}-{N}.csv")
    np.savetxt(csv_path, csv_data, delimiter=',',
            header="rho,S_t", comments="")
    return csv_path

def saveFunnel_probabilistic(rho, S_t, time, N, max_dt):

    S = S_t.value(time[0]).flatten()
    for i in range(1,len(time)):
        S = np.vstack((S,S_t.value(time[i]).flatten()))

    log_dir = "log_data/probabilistic"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_data = np.vstack((rho, np.array(S).T))

    csv_path = os.path.join(log_dir, f"funnel{max_dt}-{N}.csv")
    np.savetxt(csv_path, csv_data, delimiter=',',
            header="rho,S_t", comments="")
    return csv_path

def getEllipseFromCsv(csv_path, index):

    data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

    rho = data[0].T[index]

    S_t = data[1:len(data)].T[index]
    state_dim = int(np.sqrt(len(data)-1))
    S_t = np.reshape(S_t,(state_dim,state_dim))

    return rho, S_t