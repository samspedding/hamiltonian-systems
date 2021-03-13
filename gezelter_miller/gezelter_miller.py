import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from tqdm import tqdm
from tqdm.auto import trange
import caffeine

import matplotlib.pylab as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib as mpl
from pylab import rcParams
import mpl_toolkits.mplot3d.axes3d as p3

def param_setup():
    '''
    Returns exact parameter values; 
    and positions of the equilibria (approximate, for illustrative plotting only)
    '''
    params_pe = [
    1.0074e-2,  # k
    1.9769e0,   # d
    -2.3597e-3, # a2
    1.0408e-3,  # a4
    -7.5496e-5, # a6
    7.7569e-3,  # c
    -2.45182e-4 # d0
            ]

    # Approximate positions of equilibria (just for plotting purposes)

    centre_centres = np.array([[-1.325, 0.075], # EP2
                               [0,0],           # EP4
                               [1.325, 0.075]]) # EP6

    centre_saddles = np.array([[-2.805, 1.506], # EP1
                               [-0.54669686,  0.00217407], # EP3
                               [0.54669686,  0.00217407],  # EP5
                               [2.805, 1.506]]) # EP7
    
    return params_pe, centre_centres, centre_saddles

def potential_energy(x, y, params_pe):
    '''
    Potential energy function
    '''
    k, d, a2, a4, a6, c, d0 = params_pe
    
    return (a2*x**2 + a4*x**4 + a6*x**6 + c*x**2*np.exp(-d*x**2) + .5*k*(y+d0*x**4/k)**2)

def total_energy(states, masses, params_pe):
    '''
    Total energy function
    '''
    x, y, px, py = states
    mx, my = masses
    
    return potential_energy(x, y, params_pe) + (px**2/(2*mx) + py**2/(2*my))

def vector_field(t, states, *parameters):
    '''
    RHS of Hamilton's equations that determine the trajectories
    '''
    x, y, px, py = states
    mx, my, k, d, a2, a4, a6, c, d0 = parameters
    
    xdot = px/mx
    ydot = py/my
    pxdot = -2*a2*x - 4*a4*x**3 - 6*a6*x**5 + 2*c*x*np.exp(-d*x**2)*(d*x**2-1) - 4*d0*x**3*(y+d0*x**4/k)
    pydot = -k*(y+d0*x**4/k)
    
    return xdot, ydot, pxdot, pydot

def grad_V(x, y, params_pe):
    '''
    Returns the derivative of the potential energy wrt x and y
    '''
    k, d, a2, a4, a6, c, d0 = params_pe
    
    dV_dx = 2*a2*x + 4*a4*x**3 + 6*a6*x**5 - 2*c*x*np.exp(-d*x**2)*(d*x**2-1) + 4*d0*x**3*(y+d0*x**4/k)
    dV_dy = k*(y+d0*x**4/k)
    
    return dV_dx, dV_dy

def bounding_box(energy_level,params_pe, x_padding = 0, y_padding = 0):
    '''
    Returns x min/max and y min/max for a contour plot of the potential in config. space at a given energy level
    Requires scipy.optimize.minimize
    '''
    
    # symmetric in x so only need to minimise x
    x_func = lambda r: r[0]
    
    # need to find min and max of y, so minimise y and -y resp.
    y_top_func = lambda r: -r[1]
    y_btm_func = lambda r: r[1]
    
    # Initial points for minimising algorithm
    r0x = [-2, 0]
    r0y_top = [-2, 1]
    r0y_btm = [-2, -1]

    bounds = ((-3, 0), (-3, 3))
    
    # Constraint is that potential energy equals energy level
    cons = {'type': 'eq', 'fun': lambda r: potential_energy(r[0], r[1], params_pe) - energy_level}
    
    x_bound = -minimize(x_func, r0x, bounds = bounds, constraints = cons).x[0]
    y_bound_top = minimize(y_top_func, r0y_top, bounds = bounds, constraints = cons).x[1]
    y_bound_btm = minimize(y_btm_func, r0y_btm, bounds = bounds, constraints = cons).x[1]
    
    return np.array([[-x_bound - x_padding, x_bound + x_padding], 
                     [y_bound_btm - y_padding, y_bound_top + y_padding]])

def meshes(b_box, params_pe, x_res = 150, y_res = 100):
    '''
    Generates x, y, and V(x,y) meshes for bounding box in config space
    '''
    
    x_range = b_box[0,:] 
    y_range = b_box[1,:] 
    
    x_list = np.linspace(x_range[0], x_range[1], x_res)
    y_list = np.linspace(y_range[0], y_range[1], y_res)
    x_mesh, y_mesh = np.meshgrid(x_list, y_list)
    potential_mesh = potential_energy(x_mesh, y_mesh, params_pe)
    
    return x_mesh, y_mesh, potential_mesh


def plot_example_traj(ax, state0, runtime, masses, params_pe,
                      time_res = 0.01, RelTol = 1e-8, AbsTol = 3e-8, legend = False):
    '''
    Plots an example trajectory with IC at state0 on the input axis
    '''
    
    N_times = int(runtime/time_res)
    traj_energy = total_energy(state0, masses, params_pe)
    
    def event(t, x, *parameters):
        return abs(x[0]) - 5
    event.terminal = 1
    
    sol = solve_ivp(vector_field, [0, runtime], state0, method='RK45',args = masses + params_pe ,\
                    events = event, dense_output=True, rtol=RelTol, atol=AbsTol)

    times = np.linspace(0, runtime, N_times)
    traj = sol.sol.__call__(times)

    x_traj = traj[0,:]
    y_traj = traj[1,:]
   
    # Plot the trajectory:

    ax.plot(x_traj, y_traj, c = 'purple')
    
    return None

def xy_py0_section(energy_level, masses, params_pe, runtime, x_res, y_res, RelTol = 1e-8, AbsTol = 3e-8, label = None):
    '''
    Returns the phase space points for a poincare section in config space for py = 0
    '''

    # meshes for the initial points for PSS:

    section_b_box = bounding_box(energy_level, params_pe)
    x_smesh, y_smesh, potential_smesh = meshes(section_b_box, params_pe, x_res = x_res, y_res = y_res)

    # unravel the points within the allowed region into lists of x and y coords:
    
    x_init_list = np.ravel(x_smesh[potential_smesh <= energy_level])
    y_init_list = np.ravel(y_smesh[potential_smesh <= energy_level]) # s/o to Maurice Ravel
    N_points = len(x_init_list)
    
    # corresponding momentum lists:
    
    px_init_list = np.zeros((N_points,))
    py_init_list = np.zeros((N_points,))
    for i in range(N_points):
        x, y = x_init_list[i], y_init_list[i]
        px_init_list[i] = np.sqrt(2*masses[0]*(energy_level - potential_energy(x, y, params_pe)))
        
    #-------------------------------------------------------------------------------------------------
    
    # event function for reintersection with surface p_y = 0
    
    def py_event(t, state, *parameters):
        return state[3]
    py_event.terminal = 0
    py_event.direction = 0
    
    # initialise intersection array and integrate, collecting events:
    traj_intersect = []

    for i in trange(N_points, smoothing = 0, desc = label):
        state0 = [x_init_list[i], y_init_list[i], px_init_list[i], py_init_list[i]]
        traj = solve_ivp(vector_field, [0, runtime], state0, dense_output = True, events = py_event,
                       args = masses + params_pe, rtol = RelTol, atol = AbsTol)

        numEvents = len(traj.t_events[0]) - 1

        if numEvents >= 1:    # if event has been caught
            event_points = np.zeros((numEvents, 5))
            event_points[:,0] = np.asarray(traj.t_events[0][1:]) # fill first column with event times
            event_points[:,1:] = traj.sol.__call__(traj.t_events[0][1:]).T # fill remaining four with event states
            traj_intersect = np.append(traj_intersect,event_points) # add intersection to list

    traj_intersect = np.reshape(traj_intersect,(int(len(traj_intersect)/5),5))
    
    return traj_intersect

def xy_px0_section(energy_level, masses, params_pe, runtime, x_res, y_res, RelTol = 1e-8, AbsTol = 3e-8, label = None):
    '''
    Returns the phase space points for a poincare section in config space for px = 0
    '''
    # meshes for the initial points for PSS:

    section_b_box = bounding_box(energy_level, params_pe)
    x_smesh, y_smesh, potential_smesh = meshes(section_b_box, params_pe, x_res = x_res, y_res = y_res)

    # unravel the points within the allowed region into lists of x and y coords:
    
    x_init_list = np.ravel(x_smesh[potential_smesh <= energy_level])
    y_init_list = np.ravel(y_smesh[potential_smesh <= energy_level]) # s/o to Maurice Ravel
    N_points = len(x_init_list)
    
    # corresponding momentum lists:
    
    px_init_list = np.zeros((N_points,))
    py_init_list = np.zeros((N_points,))
    for i in range(N_points):
        x, y = x_init_list[i], y_init_list[i]
        py_init_list[i] = np.sqrt(2*masses[1]*(energy_level - potential_energy(x, y, params_pe)))
        
    #-------------------------------------------------------------------------------------------------
    
    # event function for reintersection with surface p_x = 0
    
    def px_event(t, state, *parameters):
        return state[2]
    px_event.terminal = 0
    px_event.direction = 0
    
    # initialise intersection array and integrate, collecting events:
    traj_intersect = []

    for i in trange(N_points, smoothing = 0, desc = label):
        state0 = [x_init_list[i], y_init_list[i], px_init_list[i], py_init_list[i]]
        traj = solve_ivp(vector_field, [0, runtime], state0, dense_output = True, events = px_event,
                       args = masses + params_pe, rtol = RelTol, atol = AbsTol)

        numEvents = len(traj.t_events[0]) - 1

        if numEvents >= 1:    # if event has been caught
            event_points = np.zeros((numEvents, 5))
            event_points[:,0] = np.asarray(traj.t_events[0][1:]) # fill first column with event times
            event_points[:,1:] = traj.sol.__call__(traj.t_events[0][1:]).T # fill remaining four with event states
            traj_intersect = np.append(traj_intersect,event_points) # add intersection to list

    traj_intersect = np.reshape(traj_intersect,(int(len(traj_intersect)/5),5))
    
    return traj_intersect

def bounding_box_xpx(energy_level, masses, params_pe, x_padding = 0, px_padding = 0):
    '''
    Same as bounding_box but for (x, p_x) plane with y = 0
    '''
    
    # here r = (x, p_x)
    
    x_func = lambda r: r[0]
    px_func = lambda r: r[1]
    
    x0 = [-2, 0]
    px0 = [-1, -1]
    
    bounds = ((-4, 0), (-20, 0))
    
    # Constraint is now that H(x, y=0, px, py=0) = E
    cons = {'type': 'eq', 'fun': lambda r: total_energy([r[0], 0, r[1],0], masses, params_pe) - energy_level}
    
    x_bound = minimize(x_func, x0, bounds = bounds, constraints = cons).x[0]
    px_bound = minimize(px_func, px0, bounds = bounds, constraints = cons).x[1]
    
    return np.array([[x_bound - x_padding,-x_bound + x_padding], [px_bound - px_padding,-px_bound + px_padding]])

def bounding_box_ypy(x_val, energy_level, masses, params_pe, y_padding = 0, py_padding = 0):
    '''
    Same as bounding_box but for (y, p_y) plane with x = x_val, p_x = 0
    '''
    
    # here r = (y, p_y)
    
    y_min_func = lambda r: r[0]
    y_max_func = lambda r: -r[0]
    py_func = lambda r: r[1]
    
    y0_min = [-1, 0]
    y0_max = [-1, 0]
    py0 = [-1, -1]
    
    bounds = ((-4, 0), (-20, 0))
    
    # Constraint is now that H(x=x_val, y, px=0, py) = E
    cons = {'type': 'eq', 'fun': lambda r: total_energy([x_val, r[0], 0, r[1]], masses, params_pe) - energy_level}
    
    y_min_bound = minimize(y_min_func, y0_min, bounds = bounds, constraints = cons).x[0]
    y_max_bound = -minimize(y_max_func, y0_max, bounds = bounds, constraints = cons).x[0]
    py_bound = minimize(py_func, py0, bounds = bounds, constraints = cons).x[1]
    
    return np.array([[y_min_bound - y_padding, y_max_bound + y_padding], [py_bound - py_padding,-py_bound + py_padding]])


def xpx_y0_section(energy_level, masses, params_pe, runtime, x_res, px_res, 
                   py_sign = 1, y_event_direction = 0, RelTol = 1e-8, AbsTol = 3e-8, label = None):
    
    # Meshes for x and px values within energy boundary:
    b_box = bounding_box_xpx(energy_level, masses, params_pe)
    x_list = np.linspace(b_box[0,0], b_box[0,1], x_res)
    px_list = np.linspace(b_box[1,0], b_box[1,1], px_res)
    
    x_mesh, px_mesh = np.meshgrid(x_list, px_list)
    energy_mesh = total_energy([x_mesh, 0, px_mesh, 0], masses, params_pe)

    # Reusing variable names here (unraveling meshes into list of points in allowed energy region):
    x_list = np.ravel(x_mesh[energy_mesh <= energy_level])
    px_list = np.ravel(px_mesh[energy_mesh <= energy_level])
    N_points = len(x_list)

    # py>0 value for each point:
    py_list = np.zeros((N_points,))

    for i in range(N_points):
        py_list[i] = py_sign * np.sqrt(2*masses[1]*(energy_level - potential_energy(x_list[i], 0, params_pe) \
                                          - px_list[i]**2 / (2*masses[0])))

    # event function for reintersection with surface y = 0

    def y_event(t, state, *parameters):
        return state[1]
    y_event.terminal = 0
    y_event.direction = y_event_direction

    # initialise intersection array and integrate, collecting events:
    traj_intersect = []

    for i in trange(N_points, smoothing = 0):
        state0 = [x_list[i], 0, px_list[i], py_list[i]]
        traj = solve_ivp(vector_field, [0, runtime], state0, dense_output = True, events = y_event,
                         args = masses + params_pe, rtol = RelTol, atol = AbsTol)

        numEvents = len(traj.t_events[0]) - 1

        if numEvents >= 1:    # if event has been caught
            event_points = np.zeros((numEvents, 5))
            event_points[:,0] = np.asarray(traj.t_events[0][1:]) # fill first column with event times
            event_points[:,1:] = traj.sol.__call__(traj.t_events[0][1:]).T # fill remaining four with event states
            traj_intersect = np.append(traj_intersect,event_points) # add intersection to list

    traj_intersect = np.reshape(traj_intersect,(int(len(traj_intersect)/5),5))
    
    return traj_intersect

def ypy_section(x_val, energy_level, masses, params_pe, runtime, y_res, py_res, 
                   px_sign = 1, x_event_direction = 0, RelTol = 1e-8, AbsTol = 3e-8, label = None):
    
    # Meshes for y and py values within energy boundary:
    b_box = bounding_box_ypy(x_val, energy_level, masses, params_pe)
    y_list = np.linspace(b_box[0,0], b_box[0,1], y_res)
    py_list = np.linspace(b_box[1,0], b_box[1,1], py_res)
    
    y_mesh, py_mesh = np.meshgrid(y_list, py_list)
    energy_mesh = total_energy([x_val, y_mesh, 0, py_mesh], masses, params_pe)

    # Reusing variable names here (unraveling meshes into list of points in allowed energy region):
    y_list = np.ravel(y_mesh[energy_mesh <= energy_level])
    py_list = np.ravel(py_mesh[energy_mesh <= energy_level])
    N_points = len(y_list)

    # px>0 value for each point:
    px_list = np.zeros((N_points,))

    for i in range(N_points):
        px_list[i] = px_sign * np.sqrt(2*masses[0]*(energy_level - potential_energy(x_val, y_list[i], params_pe) \
                                          - py_list[i]**2 / (2*masses[1])))

    # event function for reintersection with surface x = x_val

    def x_event(t, state, *parameters):
        return state[0] - x_val
    x_event.terminal = 0
    x_event.direction = x_event_direction

    # initialise intersection array and integrate, collecting events:
    traj_intersect = []

    for i in trange(N_points, smoothing = 0, desc=label):
        state0 = [x_val, y_list[i], px_list[i], py_list[i]]
        traj = solve_ivp(vector_field, [0, runtime], state0, dense_output = True, events = x_event,
                         args = masses + params_pe, rtol = RelTol, atol = AbsTol)

        numEvents = len(traj.t_events[0]) - 1

        if numEvents >= 1:    # if event has been caught
            event_points = np.zeros((numEvents, 5))
            event_points[:,0] = np.asarray(traj.t_events[0][1:]) # fill first column with event times
            event_points[:,1:] = traj.sol.__call__(traj.t_events[0][1:]).T # fill remaining four with event states
            traj_intersect = np.append(traj_intersect,event_points) # add intersection to list

    traj_intersect = np.reshape(traj_intersect,(int(len(traj_intersect)/5),5))
    
    return traj_intersect