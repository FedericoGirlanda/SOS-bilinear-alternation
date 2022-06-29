import numpy as np
from abc import ABC, abstractmethod
import scipy.linalg

from pydrake.all import FiniteHorizonLinearQuadraticRegulatorOptions, \
                        FiniteHorizonLinearQuadraticRegulator, \
                        PiecewisePolynomial, \
                        Linearize, \
                        LinearQuadraticRegulator
from pydrake.examples.pendulum import  PendulumPlant

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    ref: Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the lqr gain
    K = np.array(scipy.linalg.inv(R).dot(B.T.dot(X)))
    eigVals, eigVecs = scipy.linalg.eig(A-B.dot(K))
    return K, X, eigVals

class AbstractController(ABC):
    """
    Abstract controller class. All controller should inherit from
    this abstract class.
    """
    @abstractmethod
    def get_control_output(self, meas_pos, meas_vel, meas_tau, meas_time):
        """
        The function to compute the control input for the pendulum actuator.
        Supposed to be overwritten by actual controllers. The API of this
        method should be adapted. Unused inputs/outputs can be set to None.

        **Parameters**

        ``meas_pos``: ``float``
            The position of the pendulum [rad]
        ``meas_vel``: ``float``
            The velocity of the pendulum [rad/s]
        ``meas_tau``: ``float``
            The meastured torque of the pendulum [Nm]
        ``meas_time``: ``float``
            The collapsed time [s]

        Returns
        -------
        ``des_pos``: ``float``
            The desired position of the pendulum [rad]
        ``des_vel``: ``float``
            The desired velocity of the pendulum [rad/s]
        ``des_tau``: ``float``
            The torque supposed to be applied by the actuator [Nm]
        """

        des_pos = None
        des_vel = None
        des_tau = None
        return des_pos, des_vel, des_tau

    def init(self, x0):
        """
        Initialize the controller. May not be necessary.

        Parameters
        ----------
        ``x0``: ``array like``
            The start state of the pendulum
        """
        self.x0 = x0

    def set_goal(self, x):
        """
        Set the desired state for the controller. May not be necessary.

        Parameters
        ----------
        ``x``: ``array like``
            The desired goal state of the controller
        """

        self.goal = x

class LQRController(AbstractController):
    """
    Controller which stabilizes the pendulum at its instable fixpoint.
    """
    def __init__(self, mass=1.0, length=0.5, damping=0.1,
                 gravity=9.81, torque_limit=np.inf):
        """
        Controller which stabilizes the pendulum at its instable fixpoint.

        Parameters
        ----------
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        """
        self.m = mass
        self.len = length
        self.b = damping
        self.g = gravity
        self.torque_limit = torque_limit

        self.A = np.array([[0, 1],
                           [self.g/self.len, -self.b/(self.m*self.len**2.0)]])
        self.B = np.array([[0, 1./(self.m*self.len**2.0)]]).T
        self.Q = np.diag((10, 1))
        self.R = np.array([[1]])

        self.K, self.S, _ = lqr(self.A, self.B, self.Q, self.R)

        self.clip_out = False

    def set_goal(self, x):
        pass

    def set_clip(self):
        self.clip_out = True

    def get_control_output(self, meas_pos, meas_vel,
                           meas_tau=0, meas_time=0):
        """
        The function to compute the control input for the pendulum actuator

        Parameters
        ----------
        meas_pos : float
            the position of the pendulum [rad]
        meas_vel : float
            the velocity of the pendulum [rad/s]
        meas_tau : float, default=0
            the meastured torque of the pendulum [Nm]
            (not used)
        meas_time : float, default=0
            the collapsed time [s]
            (not used)

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
            (not used, returns None)
        des_vel : float
            the desired velocity of the pendulum [rad/s]
            (not used, returns None)
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))

        th = pos + np.pi
        th = (th + np.pi) % (2*np.pi) - np.pi

        y = np.asarray([th, vel])

        u = np.asarray(-self.K.dot(y))[0]

        if not self.clip_out:
            if np.abs(u) > self.torque_limit:
                u = None

        else:
            u = np.clip(u, -self.torque_limit, self.torque_limit)

        

        # since this is a pure torque controller,
        # set des_pos and des_pos to None
        des_pos = None
        des_vel = None

        return des_pos, des_vel, u


class TVLQRController(AbstractController):
    """
    Controller acts on a predefined trajectory.
    """
    def __init__(self,
                 data_dict,
                 mass=1.0,
                 length=0.5,
                 damping=0.1,
                 gravity=9.81,
                 torque_limit=np.inf):
        """
        Controller acts on a predefined trajectory.

        Parameters
        ----------
        data_dict : dictionary
            a dictionary containing the trajectory to follow
            should have the entries:
            data_dict["des_time_list"] : desired timesteps
            data_dict["des_pos_list"] : desired positions
            data_dict["des_vel_list"] : desired velocities
            data_dict["des_tau_list"] : desired torques
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        """

        self.Qf = np.array([None]) # Useful only for RoA purposes

        # load the trajectory
        self.traj_time = data_dict["des_time_list"]
        self.traj_pos = data_dict["des_pos_list"]
        self.traj_vel = data_dict["des_vel_list"]
        self.traj_tau = data_dict["des_tau_list"]

        self.max_time = self.traj_time[-1]

        self.traj_time = np.reshape(self.traj_time,
                                    (self.traj_time.shape[0], -1))
        self.traj_tau = np.reshape(self.traj_tau,
                                   (self.traj_tau.shape[0], -1)).T

        x0_desc = np.vstack((self.traj_pos, self.traj_vel))
        # u0_desc = self.traj_tau

        u0 = PiecewisePolynomial.FirstOrderHold(self.traj_time, self.traj_tau)
        x0 = PiecewisePolynomial.CubicShapePreserving(
                                              self.traj_time,
                                              x0_desc,
                                              zero_end_point_derivatives=True)

        # create drake pendulum plant
        self.plant = PendulumPlant()
        self.context = self.plant.CreateDefaultContext()
        params = self.plant.get_mutable_parameters(self.context)
        params[0] = mass
        params[1] = length
        params[2] = damping
        params[3] = gravity

        self.torque_limit = torque_limit

        # create lqr context
        self.tilqr_context = self.plant.CreateDefaultContext()
        self.plant.get_input_port(0).FixValue(self.tilqr_context, [0])
        self.Q_tilqr = np.diag((50., 1.))
        self.R_tilqr = [1]

        # Setup Options and Create TVLQR
        self.options = FiniteHorizonLinearQuadraticRegulatorOptions()
        self.Q = np.diag([200., 0.1])
        self.options.x0 = x0
        self.options.u0 = u0

        self.counter = 0
        self.last_pos = 0.0
        self.last_vel = 0.0

    def init(self, x0):
        self.counter = 0
        self.last_pos = 0.0
        self.last_vel = 0.0

    def set_goal(self, x):
        pos = x[0] + np.pi
        pos = (pos + np.pi) % (2*np.pi) - np.pi
        self.tilqr_context.SetContinuousState([pos, x[1]])
        linearized_pendulum = Linearize(self.plant, self.tilqr_context)
        (K, S) = LinearQuadraticRegulator(linearized_pendulum.A(),
                                          linearized_pendulum.B(),
                                          self.Q_tilqr,
                                          self.R_tilqr)

        if (not self.Qf.all() == None):
            self.options.Qf = self.Qf # Useful only for RoA purposes
        else:
            self.options.Qf = S  
        
        self.tvlqr = FiniteHorizonLinearQuadraticRegulator(
                        self.plant,
                        self.context,
                        t0=self.options.u0.start_time(),
                        tf=self.options.u0.end_time(),
                        Q=self.Q,
                        R=np.eye(1)*2,
                        options=self.options)

    def get_control_output(self, meas_pos=None, meas_vel=None, meas_tau=None,
                           meas_time=None):
        """
        The function to read and send the entries of the loaded trajectory
        as control input to the simulator/real pendulum.

        Parameters
        ----------
        meas_pos : float, deault=None
            the position of the pendulum [rad]
        meas_vel : float, deault=None
            the velocity of the pendulum [rad/s]
        meas_tau : float, deault=None
            the meastured torque of the pendulum [Nm]
        meas_time : float, deault=None
            the collapsed time [s]

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
        des_vel : float
            the desired velocity of the pendulum [rad/s]
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))
        x = np.array([[pos], [vel]])

        des_pos = self.last_pos
        des_vel = self.last_vel
        # des_tau = 0

        if self.counter < len(self.traj_pos):
            des_pos = self.traj_pos[self.counter]
            des_vel = self.traj_vel[self.counter]
            # des_tau = self.traj_tau[self.counter]
            self.last_pos = des_pos
            self.last_vel = des_vel

        self.counter += 1

        time = min(meas_time, self.max_time)

        uu = self.tvlqr.u0.value(time)
        xx = self.tvlqr.x0.value(time)
        KK = self.tvlqr.K.value(time)
        kk = self.tvlqr.k0.value(time)

        xdiff = x - xx
        pos_diff = (xdiff[0] + np.pi) % (2*np.pi) - np.pi
        xdiff[0] = pos_diff

        des_tau = (uu - KK.dot(xdiff) - kk)[0][0]
        des_tau = np.clip(des_tau, -self.torque_limit, self.torque_limit)

        return des_pos, des_vel, des_tau

    def set_Qf(self, Qf):
        """
        This function is useful only for RoA purposes. Used to set the
        final S-matrix of the tvlqr controller.

        Parameters
        ----------
        Qf : matrix
            the S-matrix from time-invariant RoA estimation around the 
            up-right position.
        """
        self.Qf = Qf