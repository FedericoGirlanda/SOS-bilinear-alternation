import numpy as np
from pydrake.all import Variables, MonomialBasis, Solve, MathematicalProgram, Jacobian, PiecewisePolynomial

def TVrhoSearch(pendulum, controller, x0_traj, knot, time, Q, rho_t):

    # failing checker
    fail = False

    # Sampled constraints
    t_iplus1 = time[knot]
    t_i = time[knot-1]
    dt = t_iplus1 - t_i

    # K and S matrices from TVLQR control
    K_t = controller.tvlqr.K
    S_t = controller.tvlqr.S
    K_i = K_t.value(t_i)
    S_i = S_t.value(t_i)
    S_iplus1 = S_t.value(t_iplus1)
    Sdot_i = (S_iplus1-S_i)/dt

    # Pendulum parameters
    m = pendulum.m
    l = pendulum.l
    g = pendulum.g
    b = pendulum.b
    torque_limit = pendulum.torque_limit

    # Opt. problem definition
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "x") # shifted system state

    rho_i = prog.NewContinuousVariables(1)[0]
    rho_dot_i = 0
    prog.AddCost(-rho_i)
    prog.AddConstraint(rho_i >= 0)

    # Dynamics definition
    xg = np.array(x0_traj).T[knot-1] 
    x = xbar + xg
    ubar = -K_i.dot(xbar)[0]
    Tsin = -(x[0]-xg[0]) + (x[0]-xg[0])**3/6  - (x[0]-xg[0])**5/120 
    fn = [x[1], (ubar-b*x[1]-Tsin*m*g*l)/(m*l*l)]

    # Lyapunov function and its derivative
    V_i = (xbar).dot(S_i.dot(xbar))
    Vdot_i_x = V_i.Jacobian(xbar).dot(fn)
    Vdot_i_t = xbar.dot(Sdot_i.dot(xbar))
    Vdot_i = Vdot_i_x + Vdot_i_t

    # Boundaries due to the saturation 
    u_minus = - torque_limit
    u_plus = torque_limit
    fn_minus = [x[1], (u_minus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_minus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_minus)
    fn_plus = [x[1], (u_plus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_plus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_plus)

    # Multipliers definition
    lambda_1 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_4 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()

    # Retriving the mu result from the Q matrix
    MB = MonomialBasis(Variables(xbar), 2)
    h = 0
    for i, mi in enumerate(MB):
        for j, mj in enumerate(MB):
            h += mi.ToExpression() * Q[i, j] * mj.ToExpression()

    # Optimization constraints 
    constr_minus = - (Vdot_minus) +rho_dot_i + h*(V_i - rho_i) + lambda_1*(-u_minus+ubar) 
    constr = - (Vdot_i) + rho_dot_i + + h*(V_i - rho_i) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar) 
    constr_plus = - (Vdot_plus) +rho_dot_i + h*(V_i - rho_i) + lambda_4*(u_plus-ubar) 

    for c in [constr_minus, constr, constr_plus]:
        prog.AddSosConstraint(c)

    # Solve the problem
    result = Solve(prog)
    rho_i = result.GetSolution(rho_i)
    fail = not result.is_success()

    return (fail, rho_i)

def TVmultSearch(pendulum, controller, x0_traj, knot, time, rho_t):

    # failing checker
    fail = False

    # Sampled constraints
    t_iplus1 = time[knot]
    t_i = time[knot-1]
    dt = t_iplus1 - t_i

    # K and S matrices from TVLQR control
    K_t = controller.tvlqr.K
    S_t = controller.tvlqr.S
    K_i = K_t.value(t_i)
    S_i = S_t.value(t_i)
    S_iplus1 = S_t.value(t_iplus1)
    Sdot_i = (S_iplus1-S_i)/dt

    # Pendulum parameters
    m = pendulum.m
    l = pendulum.l
    g = pendulum.g
    b = pendulum.b
    torque_limit = pendulum.torque_limit

    # Opt. problem definition
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "state") # shifted system state
    gamma = prog.NewContinuousVariables(1)[0]
    prog.AddCost(gamma)
    prog.AddConstraint(gamma <= 0)

    # Dynamics definition
    xg = np.array(x0_traj).T[knot-1] 
    x = xbar + xg
    ubar = -K_i.dot(xbar)[0]
    Tsin = -(x[0]-xg[0]) + (x[0]-xg[0])**3/6  - (x[0]-xg[0])**5/120 
    fn = [x[1], (ubar-b*x[1]-Tsin*m*g*l)/(m*l*l)]

    # Lyapunov function and its derivative
    V_i = (xbar).dot(S_i.dot(xbar))
    Vdot_i_x = V_i.Jacobian(xbar).dot(fn)
    Vdot_i_t = xbar.dot(Sdot_i.dot(xbar))
    Vdot_i = Vdot_i_x + Vdot_i_t

    # Boundaries due to the saturation 
    u_minus = - torque_limit
    u_plus = torque_limit
    fn_minus = [x[1], (u_minus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_minus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_minus)
    fn_plus = [x[1], (u_plus-b*x[1]-Tsin*m*g*l)/(m*l*l)] 
    Vdot_plus = Vdot_i_t + V_i.Jacobian(xbar).dot(fn_plus)

    # Multipliers definition
    h = prog.NewFreePolynomial(Variables(xbar), 4)
    mu_ij = h.ToExpression()
    lambda_1 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_2 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_3 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()
    lambda_4 = prog.NewSosPolynomial(Variables(xbar), 4)[0].ToExpression()

    # rho dot definition
    rho_i = rho_t[knot-1]
    rho_iplus1 = rho_t[knot]
    rho_dot_i = (rho_iplus1 - rho_i)/dt

    # Optimization constraints 
    constr_minus = - (Vdot_minus) +rho_dot_i + mu_ij*(V_i - rho_i) + lambda_1*(-u_minus+ubar) + gamma
    constr = - (Vdot_i) + rho_dot_i + mu_ij*(V_i - rho_i) + lambda_2*(u_minus-ubar) + lambda_3*(-u_plus+ubar) + gamma
    constr_plus = - (Vdot_plus) +rho_dot_i + mu_ij*(V_i - rho_i) + lambda_4*(u_plus-ubar) + gamma

    for c in [constr_minus, constr, constr_plus]:
        prog.AddSosConstraint(c)

    # Solve the problem and store the polynomials
    result = Solve(prog)
    if result.is_success():

        mb_ = MonomialBasis(h.indeterminates(), 2)
        Qdim = mb_.shape[0]

        coeffs = h.decision_variables()
        Q = np.zeros((Qdim, Qdim))
        row = 0
        col = 0
        for k, coeff in enumerate(coeffs):
            Q[row, col] = result.GetSolution(coeff)
            Q[col, row] = result.GetSolution(coeff)

            if col == Q.shape[0] - 1:
                row += 1
                col = row
            else:
                col += 1

        (fail, Q) = TVmultSearch_StepBack(prog, gamma, result.get_optimal_cost(), 0.1, h)

    else:
        Q = None
        fail = True

    return fail, Q

def TVmultSearch_StepBack(prog, objective, optimalCost, eps, h):

    fail = False

    prog.AddCost(-objective)
    prog.AddConstraint(objective <= optimalCost + eps)

    # Solve the problem and store the polynomials
    result = Solve(prog)
    if result.is_success():

        mb_ = MonomialBasis(h.indeterminates(), 2)
        Qdim = mb_.shape[0]

        coeffs = h.decision_variables()
        Q = np.zeros((Qdim, Qdim))
        row = 0
        col = 0
        for k, coeff in enumerate(coeffs):
            Q[row, col] = result.GetSolution(coeff)
            Q[col, row] = result.GetSolution(coeff)

            if col == Q.shape[0] - 1:
                row += 1
                col = row
            else:
                col += 1

    else:
        Q = None
        fail = True

    return fail, Q
