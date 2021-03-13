# deleon_berne.py

import caffeine

import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

from tqdm import tqdm
from tqdm.auto import trange

import matplotlib.pylab as plt
import matplotlib as mpl

def potential_energy(x, y, params_pe):
    '''
    Input: x, y co-ords as float or meshgrid, parameters w/o masses
    params_pe = [lamb, zeta, V, y_w, e_s, D_x]
    
    Returns potential energy as float or meshgrid
    '''
    if np.size(x) == 1:
        nX = 1
        nY = 1
    else:
        nX = np.size(x,1)
        nY = np.size(x,0)
        x = np.ravel(x, order = 'C')
        y = np.ravel(y, order = 'C')
    
    lamb, zeta, V, y_w, e_s, D_x = params_pe

    V = D_x*(1-np.exp(-lamb*x))**2 + V*y**2*(y**2 - 2*y_w**2) / y_w**4 * np.exp(-zeta*lamb*x) + e_s

    pe = np.reshape(V, (nY, nX))
    
    if np.size(x) == 1:
        pe = pe[0,0]

    return pe

def vector_field(t, states, *parameters):
    '''
    Input: time t (not used here), states = [x, y, px, py], parameters inc. masses
    
    Returns vector field from RHS of Ham. equations
    '''
    m_x, m_y = parameters[0:2]
    lamb, zeta, V, y_w, e_s, D_x = parameters[2:]
    x, y, px, py = states

    q1Dot = px/m_x

    q2Dot = py/m_y

    p1Dot = 2*D_x*lamb*np.exp(-lamb*x)*(np.exp(-lamb*x)-1) \
        + V / y_w**4 * zeta * lamb * y**2 * (y**2 - 2*y_w**2) * np.exp(-zeta*lamb*x)

    p2Dot = - 4 * V / y_w**4 * y * (y**2 - y_w**2) * np.exp(-zeta*lamb*x)
    
    return np.array([q1Dot, q2Dot, p1Dot, p2Dot])

def total_energy(states, parameters):
    '''
    Input: phase space coords = states = [x, y, px, py], parameters
    Returns: total energy H = T + V at those coords
    '''    
    x, y, px, py = states
    m_x, m_y = parameters[0:2]
    params_pe = parameters[2:]

    totalEnergy = px**2/(2*m_x) + py**2/(2*m_y) + potential_energy(x, y, params_pe) 

    return totalEnergy

def py_fixed_energy(x, y, px, parameters, energy):
    """ Returns y-momentum (py) for a fixed energy at other phase space coordinates
    """
    m_x, m_y = parameters[0:2]
    params_pe = parameters[2:]
    
    if energy >= px**2/(2*m_x) + potential_energy(x,y,params_pe):
        py = np.sqrt( 2*m_y*( energy - potential_energy(x,y,params_pe) - px**2/(2*m_x)  )   )
    else:
        py = np.NaN
    return py

def px_fixed_energy(x, y, py, parameters, energy):
    """ Returns x-momentum (px) for a fixed energy at other phase space coordinates
    """
    m_x, m_y = parameters[0:2]
    params_pe = parameters[2:]
    
    if energy >= py**2/(2*m_y) + potential_energy(x,y,params_pe):
        px = np.sqrt( 2*m_x*( energy - potential_energy(x,y,params_pe) - py**2/(2*m_y)  )   )
    else:
        px = np.NaN
    return px

def config_bounding_box(parameters, energy):
    '''
    Input: parameters, energy level
    Output: list of x and y axis limits for a contour plot of the potential surface in configuration space
    '''
    
    m_x, m_y, lamb, zeta, V, y_w, e_s, D_x = parameters
    
    def func(x):
        return (D_x*(1-np.exp(-lamb*x))**2-V*np.exp(-zeta*lamb*x) + e_s - energy)

    x_max = float(fsolve(func, 0.5/lamb))
    x_min = float(fsolve(func, -0.667/lamb))

    if zeta == 2:
        x_crit = -1/lamb * np.log(1-(energy-e_s)/D_x)
    else:
        x_crit = -1/lamb * np.log((zeta-1-np.sqrt((1-zeta)**2 - zeta*(zeta-2)*(1-(energy-e_s)/D_x)))/(zeta-2))

    y_min = -y_w * np.sqrt(1+np.sqrt(1+2*D_x/(zeta*V)*np.exp(-lamb*x_crit*(1-zeta))*(1-np.exp(-lamb*x_crit))))
    y_max = - y_min
    return [x_min, x_max, y_min, y_max]

def bounding_box(parameters, energy, section_type, section_constant):
    '''
    Input: parameters, energy level, 
        poincare section type and constant:
        # section_type = 0: (x, y),  px = constant
        # section_type = 1: (x, y),  py = constant
        # section_type = 2: (x, px), y  = constant
        # section_type = 3: (y, py), x  = constant
        
    Output: list of x and y axis limits for a contour plot of the total energy surface in the space defined by the section
    '''
    m_x, m_y, lamb, zeta, V, y_w, e_s, D_x = parameters
    
    if section_type == 0 or section_type == 1:
        def func(x):
            return (D_x*(1-np.exp(-lamb*x))**2-V*np.exp(-zeta*lamb*x) + e_s - energy)
    
        x_max = float(fsolve(func, 0.5/lamb))
        x_min = float(fsolve(func, -0.667/lamb))

        if zeta == 2:
            x_crit = -1/lamb * np.log(1-(energy-e_s)/D_x)
        else:
            x_crit = -1/lamb * np.log((zeta-1-np.sqrt((1-zeta)**2 - zeta*(zeta-2)*(1-(energy-e_s)/D_x)))/(zeta-2))

        y_min = -y_w * np.sqrt(1+np.sqrt(1+2*D_x/(zeta*V)*np.exp(-lamb*x_crit*(1-zeta))*(1-np.exp(-lamb*x_crit))))
        y_max = - y_min
        return [x_min, x_max, y_min, y_max]
    
    if section_type == 2:
        y0 = section_constant
        K = V*y0**2/y_w**4*(y0**2-2*y_w**2)
        def f1(x):
            return (2*D_x*np.exp(-lamb*x)*(1-np.exp(-lamb*x)) \
                    -zeta*K*np.exp(-lamb*zeta*x))
        x_crit = float(fsolve(f1, 0))
        px_max = np.sqrt(2*m_x*(energy-e_s-D_x*(1-np.exp(-lamb*x_crit))**2-K*np.exp(-lamb*zeta*x_crit)))
        px_min = -px_max
        
        def f2(x):
            return (D_x*(1-np.exp(-lamb*x))**2+K*np.exp(-zeta*lamb*x) - energy + e_s)
        
        x_min = float(fsolve(f2, -1/2))
        x_max = float(fsolve(f2, 1/2))

        return [x_min, x_max, px_min, px_max]
        
    if section_type == 3:
        x0 = section_constant
        K = e_s + D_x*(1-np.exp(-lamb*x0))**2
        V_new = V*np.exp(-zeta*lamb*x0)
        
        y_min = -y_w*np.sqrt(1+np.sqrt(1+(energy-K)/V_new))
        y_max = -y_min
        py_min = -np.sqrt((2*m_y)*(energy-K+V_new))
        py_max = -py_min
        return [y_min, y_max, py_min, py_max]
        
    return "Invalid section type"
    
def UPO(parameters, energy, timespan, RelTol = 1.e-10, AbsTol = 3.e-10):
    '''
    Input: parameters, energy, integration time span, solve_ivp tolerances
    Output: callable solution for the unstable PO (along y = 0) and the time period of the PO
    '''
    
    m_x, m_y, lamb, zeta, V, y_w, e_s, D_x= parameters

    x_range = [-1/lamb*np.log(1+np.sqrt((energy-e_s)/D_x)), \
                  -1/lamb*np.log(1-np.sqrt((energy-e_s)/D_x))]

    init = np.array([x_range[0], 0, 0, 0])

    def event(t, x, *parameters):
        return x[2]
    event.terminal = False

    sol = solve_ivp(vector_field, [0,timespan], init, method='RK45',args = parameters,dense_output=True,\
                       events = event, rtol=RelTol, atol=AbsTol)
    t_period = sol.t_events[0][2]
    x = sol.sol.__call__
    
    return x, t_period

def poincare_section(parameters, section_type, section_constant, energy_level, \
                     init_grid_resolution, contour_resolution, trajectory_runtime, pbar_label = 'Processing'):
    
    '''
    Creates the required arrays for plotting the poincare section
    '''
    
    box_range = bounding_box(parameters, energy_level, section_type, section_constant)

    # Axis meshes for contour plot of boundary:
    h_cmesh, v_cmesh = np.meshgrid(np.linspace(box_range[0], box_range[1], num = contour_resolution),  \
                                   np.linspace(box_range[2], box_range[3], num = contour_resolution))

    # Axis meshes for intital data
    h_gmesh, v_gmesh = np.meshgrid(np.linspace(box_range[0], box_range[1], num = init_grid_resolution),  \
                                   np.linspace(box_range[2], box_range[3], num = init_grid_resolution))



    # Free energy mesh for contour plot and mesh for missing momentum co-ordinate

    p_mesh = np.zeros(np.shape(h_gmesh))

    if section_type == 0:
        [hlab, vlab] = ['$x$','$y$']
        #(x, y),  px = constant
        energy_mesh = total_energy([h_cmesh, v_cmesh, section_constant, 0], parameters)
        for i in range(np.size(h_gmesh,0)):
            for j in range(np.size(h_gmesh,1)):
                p_mesh[i,j] = py_fixed_energy(h_gmesh[i,j], v_gmesh[i,j], section_constant, \
                                              parameters, energy_level)
        
        def event_intersect(t, y, *parameters):
            return y[2]-section_constant
        event_intersect.direction = 0

        xMesh = h_gmesh 
        yMesh = v_gmesh
        pxMesh = section_constant* np.ones(np.shape(h_gmesh))
        pyMesh = p_mesh

        coord_nums = [1,2]
        
    if section_type == 1:
        [hlab, vlab] = ['$x$','$y$']
        #(x, y),  py = constant
        energy_mesh = total_energy([h_cmesh, v_cmesh, 0, section_constant], parameters)
        for i in range(np.size(h_gmesh,0)):
            for j in range(np.size(h_gmesh,1)):
                p_mesh[i,j] = px_fixed_energy(h_gmesh[i,j], v_gmesh[i,j], section_constant, \
                                              parameters, energy_level)
        
        def event_intersect(t, y, *parameters):
            return y[3]-section_constant
        event_intersect.direction = 0

        xMesh = h_gmesh 
        yMesh = v_gmesh
        pxMesh = p_mesh
        pyMesh = section_constant* np.ones(np.shape(h_gmesh))

        coord_nums = [1,2]
        
    if section_type == 2:
        [hlab, vlab] = ['$x$','$p_x$']
        #(x, px), y  = constant
        energy_mesh = total_energy([h_cmesh, section_constant, v_cmesh, 0], parameters)
        for i in range(np.size(h_gmesh,0)):
            for j in range(np.size(h_gmesh,1)):
                p_mesh[i,j] = py_fixed_energy(h_gmesh[i,j], section_constant, v_gmesh[i,j], \
                                              parameters, energy_level)

        def event_intersect(t, y, *parameters):
            return y[1]-section_constant
        event_intersect.direction = 1

        xMesh = h_gmesh
        yMesh = section_constant* np.ones(np.shape(h_gmesh))
        pxMesh = v_gmesh
        pyMesh = p_mesh

        coord_nums = [1,3]
        
    if section_type == 3:
        [hlab, vlab] = ['$y$','$p_y$']
        #(y, py), x  = constant
        energy_mesh = total_energy([np.ones(np.shape(h_cmesh))*section_constant, h_cmesh, 0, v_cmesh], parameters)
        for i in range(np.size(h_gmesh,0)):
            for j in range(np.size(h_gmesh,1)):
                p_mesh[i,j] = px_fixed_energy(section_constant, h_gmesh[i,j], v_gmesh[i,j], \
                                              parameters, energy_level)
        
        def event_intersect(t, y, *parameters):
            return y[0]-section_constant
        event_intersect.direction = 1

        xMesh = section_constant* np.ones(np.shape(h_gmesh))
        yMesh = h_gmesh
        pxMesh = p_mesh
        pyMesh = v_gmesh

        coord_nums = [2,4]
        
    event_intersect.terminal = False
    
    # Error tolerances for solve_ivp:
    RelTol = 3.e-7
    AbsTol = 1.e-7 

    # Initialise intersection storage:
    traj_intersect = []

    with tqdm(total = np.sum(~np.isnan(p_mesh)), desc = pbar_label) as pbar: # progress bar
        for i in range(np.size(xMesh,0)):
            for j in range(np.size(xMesh,1)): 
                if ~np.isnan(p_mesh[i,j]): # For each intial point within the energy boundary...

                    # Insert into solve_ivp:
                    traj = solve_ivp(vector_field, [0, trajectory_runtime], \
                                    [xMesh[i,j], yMesh[i,j], pxMesh[i,j], pyMesh[i,j]], \
                                    args = parameters, \
                                    events = event_intersect, dense_output = True, \
                                    rtol = RelTol, atol = AbsTol)

                    # Collect events:
                    if len(traj.t_events[0]) > 1:    # if event has been caught
                            eventCoords = traj.sol.__call__(traj.t_events[0][1:])
                            numEvents = len(traj.t_events[0][1:]) # number of intesections in the traj.
                            eventSol = np.zeros((numEvents, 5))
                            eventSol[:,0] = np.asarray(traj.t_events[0][1:])
                            eventSol[:,1:] = eventCoords.T
                            traj_intersect = np.append(traj_intersect,eventSol) # add intersection to list
                    pbar.update(1) # update progress bar

        # reshape list of intersections across all trajectories computed:
        traj_intersect = np.reshape(traj_intersect,(int(len(traj_intersect)/5),5))
    return [h_cmesh, v_cmesh, energy_mesh, coord_nums, traj_intersect]

def forward_DS(N, parameters, energy, UPO_timespan):
    '''
    Input: 
        N = number of points around the equator of the sphere. Total points on forward DS is roughly N^2.05/2
        parameters, energy level of DS, time to integrate around UPO (100 is usually good)
        
    Output: array of points on DS embedded in 4d phase space
    '''
    
    m_x, m_y, lamb, zeta, V, y_w, e_s, D_x= parameters
    
    x_PO, t_period_PO = UPO(parameters, energy, UPO_timespan)
    times = np.linspace(0, t_period_PO, 2*N)
    
    x_list = []
    y_list = []
    px_list = []
    py_list = []

    px_max_max = (2*m_y*(energy - e_s))**.5

    for j in range(N):

        x_val = x_PO(times[j])[0]
        y_val = x_PO(times[j])[1]

        potential_value = potential_energy(x_val, y_val, parameters[2:])
        
        energy_diff = np.max([energy - potential_value,0])
        
        px_max = (2*m_x*(energy_diff))**.5
        py_max = (2*m_y*(energy_diff))**.5

        N_thetas = int(N*px_max/px_max_max)
        thetas = np.linspace(0, np.pi, N_thetas)

        for k in range(N_thetas):
            theta = thetas[k]
            py_val = (2*m_y*(energy - potential_value))**.5 * np.sin(theta)
            px_val = (2*m_x*(energy - potential_value))**.5 * np.cos(theta)

            if py_val > 0:
                x_list.append(x_val)
                y_list.append(y_val)
                px_list.append(px_val)
                py_list.append(py_val)

    N_points = len(x_list)

    points = np.zeros((N_points, 4))
    points[:,0] = np.array(x_list)
    points[:,1] = np.array(y_list)
    points[:,2] = np.array(px_list)
    points[:,3] = np.array(py_list)
    
    return(points)

def gap_time(init_point, parameters, max_runtime, y_increment=1e-20, RelTol = 1e-7, AbsTol = 3e-7):
    
    '''
    Input:
        init_point = 4d point on the DS to use as initial val for integration
        parameters
        max_runtime = max time to integrate until we return to the DS
        y_increment = tiny shift in init_point away from DS to not trigger the event at t=0
        tolerances for solve_ivp
    '''
    
    x, y, px, py = init_point
    y += y_increment
    
    def event_func(t,x, *parameters):
        return x[1]

    event_func.terminal = True
    event_func.direction = 0

    sol = solve_ivp(vector_field, [0, max_runtime], [x, y, px, py], args = parameters,\
                    method='RK45',dense_output=True, events = event_func, rtol=RelTol, atol=AbsTol)
    
    if len(sol.t_events[0]>0):
        return sol.t_events[0][0]
    else:
        return max_runtime

def flux_theoretical(energy, parameters):
    m_x, m_y, lamb, zeta, V, y_w, e_s, D_x= parameters
    
    k = np.sqrt((energy-e_s)/D_x)
    
    d_theta = 0.001
    theta = -np.pi/2
    integral = 0
    while theta < np.pi/2:
        integral += np.cos(theta)**2/(1-k*np.sin(theta)) * d_theta
        theta += d_theta
    
    flux = 2/lamb*np.sqrt(2*m_x/D_x)*(energy-e_s)*integral
    
    return flux

    