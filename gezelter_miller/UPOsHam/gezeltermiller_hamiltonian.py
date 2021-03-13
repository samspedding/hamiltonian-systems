# -*- coding: utf-8 -*-
# """
# Created on Tue Jul 30 10:02:48 2019

# @author: Wenyang Lyu and Shibabrat Naik

# Script to define expressions for the De Leon-Berne Hamiltonian
# """

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import math
from scipy import optimize


def init_guess_eqpt_gm(eqNum, parameters):
    """
    Returns guess for solving configuration space coordinates of the equilibrium points.  

    For numerical solution of the equilibrium points, this function returns the guess that be inferred from the potential energy surface. 

    Parameters
    ----------
    eqNum : int
        = 1 for saddle and = 2,3 for centre equilibrium points

    parameters : float (list)
        model parameters

    Returns
    -------
    x0 : float (list of size 2)
        configuration space coordinates of the guess: [x, y] 

    """
    
    if eqNum == 1:
        x0 = [-2.805, 1.506] # centre-saddle
    elif eqNum == 2:
        x0 = [-1.325, 0.075] # centre-centre
    elif eqNum == 3:
        x0 = [-0.547, 0.002] # centre-saddle
    elif eqNum == 4:
        x0 = [0, 0]          # centre-centre
    elif eqNum == 5:
        x0 = [0.547, 0.002]  # centre-saddle
    elif eqNum == 6:
        x0 = [1.325, 0.075]  # centre-centre
    elif eqNum == 7:
        x0 = [2.805, 1.506]  # centre-saddle
    
    return x0


def grad_pot_gm(states, parameters):
    """ Returns the negative of the gradient of the potential energy function 
    
    Parameters
    ----------
    x : float (list of size 2) 
        configuration space coordinates: [x, y]

    parameters : float (list)
        model parameters

    Returns
    -------
    F : float (list of size 2)
        configuration space coordinates of the guess: [x, y]

    """
    mx, my, k, d, a2, a4, a6, c, d0 = parameters
    x, y = states
        
    dVdx = 2*a2*x + 4*a4*x**3 + 6*a6*x**5 - 2*c*x*np.exp(-d*x**2)*(d*x**2-1) +\
            4*d0*x**3*(y+d0*x**4/k)
    dVdy = k*(y+d0*x**4/k)
    
    return [-dVdx, -dVdy]


def pot_energy_gm(x, y, parameters):
    """ Returns the potential energy at the configuration space coordinates 

    Parameters
    ----------
    x : float
        configuration space coordinate

    y : float
        configuration space coordinate

    parameters : float (list)
        model parameters

    Returns
    -------
    float 
        potential energy of the configuration
    
    """
    mx, my, k, d, a2, a4, a6, c, d0 = parameters
    
    return a2*x**2 + a4*x**4 + a6*x**6 + c*x**2*np.exp(-d*x**2) \
        + .5*k*(y+d0*x**4/k)**2


def variational_eqns_gm(t,PHI,parameters):
    """    
    Returns the state transition matrix, PHI(t,t0), where Df(t) is the Jacobian of the Hamiltonian vector field
    
    d PHI(t, t0)/dt =  Df(t) * PHI(t, t0)

    Parameters
    ----------
    t : float 
        solution time

    PHI : 1d numpy array
        state transition matrix and the phase space coordinates at initial time in the form of a vector

    parameters : float (list)
        model parameters 

    Returns
    -------
    PHIdot : float (list of size 20)
        right hand side for solving the state transition matrix 

    """
    
    phi = PHI[0:16]
    phimatrix  = np.reshape(PHI[0:16],(4,4))
    x,y,px,py = PHI[16:20]
    
    mx, my, k, d, a2, a4, a6, c, d0 = parameters
    
    # The first order derivative of the potential energy.
    dVdx = 2*a2*x + 4*a4*x**3 + 6*a6*x**5 - 2*c*x*np.exp(-d*x**2)*(d*x**2-1) + 4*d0*x**3*(y+d0*x**4/k)
    dVdy = k*(y+d0*x**4/k)

    # The second order derivative of the potential energy. 
    d2Vdx2 = 2*a2 + 12*a4*x**2 + 30*a6*x**4 - 4*c*x*np.exp(-d*x**2)*((1-d*x**2)**2 - d*x) - 4*d0*x**2*(3*y + 7*d0*x**4/k)
        
    d2Vdy2 = k

    d2Vdydx = 4*d0*x**3
       
    d2Vdxdy = d2Vdydx    

    Df    = np.array([[0,        0,          1/mx,    0],
                      [0,        0,          0,    1/my],
                      [-d2Vdx2,  -d2Vdydx,   0,       0],
                      [-d2Vdxdy, -d2Vdy2,    0,       0]])

    
    phidot = np.matmul(Df, phimatrix) # variational equation

    PHIdot        = np.zeros(20)
    PHIdot[0:16]  = np.reshape(phidot,(1,16)) 
    PHIdot[16]    = px/mx
    PHIdot[17]    = py/my
    PHIdot[18]    = -dVdx 
    PHIdot[19]    = -dVdy
    
    return list(PHIdot)


def get_coord_gm(x,y, E, parameters):
    """ 
    Function that returns the potential energy for a given total energy with one of the configuration space coordinate being fixed

    Used to solve for coordinates on the isopotential contours using numerical solvers

    Parameters
    ----------
    x : float
        configuration space coordinate

    y : float
        configuration space coordinate

    E : float
        total energy

    parameters :float (list)
        model parameters

    Returns
    -------
        float
        Potential energy
    """
    mx, my, k, d, a2, a4, a6, c, d0 = parameters
    
    return a2*x**2 + a4*x**4 + a6*x**6 + c*x**2*np.exp(-d*x**2) \
        + .5*k*(y+d0*x**4/k)**2 - E

def configdiff_gm(guess1, guess2, ham2dof_model,\
                            half_period_model, n_turn, parameters):
    """
    Returns the difference of x(or y) coordinates of the guess initial condition and the ith turning point

    Used by turning point based on configuration difference method and passed as an argument by the user. Depending on the orientation of a system's bottleneck in the potential energy surface, this function should return either difference in x coordinates or difference in y coordinates is returned as the result.

    Parameters
    ----------
    guess1 : float (list of size 4)
        initial condition # 1

    guess2 : float (list of size 4)
        initial condition # 2 

    ham2dof_model : function name
        function that returns the Hamiltonian vector field  
        
    half_period_model : function name
        function to catch the half period event during integration
    
    n_turn : int 
        index of the number of turn as a trajectory comes close to an equipotential contour
    
    parameters : float (list)
        model parameters 

    Returns
    -------
    (x_diff1, x_diff2) or (y_diff1, y_diff2) : float (list of size 2)
        difference in the configuration space coordinates, either x or y depending on the orientation of the bottleneck.

    """
    
    TSPAN = [0,150 + n_turn * 300]
    RelTol = 3.e-7
    AbsTol = 1.e-7
    
    f1 = lambda t,x: ham2dof_model(t,x,parameters) 
    soln1 = solve_ivp(f1, TSPAN, guess1, method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, parameters), rtol=RelTol, atol=AbsTol)
    te1 = soln1.t_events[0]
    t1 = [0,te1[n_turn]]
    turn1 = soln1.sol(t1)
    x_turn1 = turn1[0,-1] 
    y_turn1 = turn1[1,-1]
    x_diff1 = guess1[0] - x_turn1
    y_diff1 = guess1[1] - y_turn1
    
    f2 = lambda t,x: ham2dof_model(t,x,parameters) 
    soln2 = solve_ivp(f2, TSPAN, guess2,method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, parameters), rtol=RelTol, atol=AbsTol)
    te2 = soln2.t_events[0]
    t2 = [0,te2[n_turn]]#[0,te2[1]]#
    turn2 = soln2.sol(t2)
    x_turn2 = turn2[0,-1] 
    y_turn2 = turn2[1,-1] 
    x_diff2 = guess2[0] - x_turn2
    y_diff2 = guess2[1] - y_turn2
    

    print("Initial guess1 %s, initial guess2 %s, \
            y_diff1 is %s, y_diff2 is %s" %(guess1, guess2, y_diff1, y_diff2))
        
    return y_diff1, y_diff2

def guess_coords_gm(guess1, guess2, i, n, e, \
                            get_coord_model,parameters):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the turning point based on confifuration difference method

    Function to be used by the turning point based on configuration difference method and passed as an argument.

    Parameters
    ----------
    guess1 : float (list of size 4)
        initial condition # 1

    guess2 : float (list of size 4)
        initial condition # 2 

    i : int
        index of the number of partitions of the interval between the two guess coordinates

    n : int
        total number of partitions of the interval between the two guess coordinates 
    
    e : float
        total energy

    get_coord_model : function name
        function that returns the potential energy for a given total energy with one of the configuration space coordinate being fixed

    parameters : float (list)
        model parameters

    Returns
    -------
    xguess, yguess : float 
        configuration space coordinates of the next guess of the initial condition

    """
    
    h = (guess2[1] - guess1[1])*i/n # h is defined for dividing the interval
    print("h is ",h)
    yguess = guess1[1] + h
    f = lambda x: get_coord_model(x,yguess,e,parameters)
    xguess = optimize.newton(f,-3.2) # to find the x coordinate for a given y
    
    return xguess, yguess


def plot_iter_orbit_gm(x, ax, e, parameters):
    """ 
    Plots the orbit in the 3D space of (x,y,p_x) coordinates with the initial and final points marked with star and circle. 

    Parameters
    ----------
    x : 2d numpy array
        trajectory with time ordering along rows and coordinates along columns

    ax : figure object
        3D matplotlib axes

    e : float
        total energy

    parameters :float (list)
        model parameters 
    
    Returns
    -------
    Empty - None

    """

    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    
    ax.plot(x[:,0],x[:,1],x[:,2],'-')
    ax.plot(x[:,0],x[:,1],-x[:,2],'--')
    ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
    ax.scatter(x[-1,0],x[-1,1],x[-1,2],s=20,marker='o')
    ax.set_xlabel(r'$x$', fontsize=axis_fs)
    ax.set_ylabel(r'$y$', fontsize=axis_fs)
    ax.set_zlabel(r'$p_x$', fontsize=axis_fs)

    return


def ham2dof_gm(t, states, parameters):
    """ 
    Returns the Hamiltonian vector field (Hamilton's equations of motion) 
    
    Used for passing to ode solvers for integrating initial conditions over a time interval.

    Parameters
    ----------
    t : float
        time instant

    x : float (list of size 4)
        phase space coordinates at time instant t

    parameters : float (list)
        model parameters

    Returns
    -------
    xDot : float (list of size 4)
        right hand side of the vector field evaluated at the phase space coordinates, x, at time instant, t

    """
    
    mx, my, k, d, a2, a4, a6, c, d0 = parameters
    x, y, px, py = states
        
    dVdx = 2*a2*x + 4*a4*x**3 + 6*a6*x**5 - 2*c*x*np.exp(-d*x**2)*(d*x**2-1) +\
            4*d0*x**3*(y+d0*x**4/k)
    dVdy = k*(y+d0*x**4/k)
    
    return [px/mx, py/my, -dVdx, -dVdy]
    


def half_period_gm(t,x,parameters):
    """
    Returns the event function 
    
    Zero of this function is used as the event criteria to stop integration along a trajectory. For symmetric periodic orbits this acts as the half period event when the momentum coordinate is zero.

    Parameters
    ----------
    t : float
        time instant

    x : float (list of size 4)
        phase space coordinates at time instant t

    parameters : float (list)
        model parameters

    Returns
    -------
    float 
        event function evaluated at the phase space coordinate, x, and time instant, t.

    """
    
    return x[2]

        
half_period_gm.terminal = True # terminate the integration.
half_period_gm.direction = 0 # zero of the event function can be approached from either direction and will trigger the terminate

