###############################################################################
# Filename: MotionPlanningFinal.py
# Author(s): Riana Gagnon, Sanjana Tewathia
#
# ASEN 5254, Fall 2022
#
# References/Acknowledgements:
#   - 
# 
###############################################################################


#-----------------------------------------------------------------------------#
# Import Statements
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d.art3d as art3d
import hppfcl as fcl

## ODE fix later 


#-----------------------------------------------------------------------------#
# Classes
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Class:    Node
# Purpose:  Node objects are Nodes that represent the vertices in the tree.
#
# Attributes/Properties:
#   - Name:             int- vertex number/name (Ex. 'v0' or 0) (goal = -1)
#   - Point:            np.array- the final state of the agent
#   - Parent:           Node- the parent node or None
#   - DistFromParent:   double- distance of current vertex from goal
#   - Trajectory:       np.array- the ode trajectory from the start to end state
#
# Functions:
#   - __init__: the constructor
#-----------------------------------------------------------------------------#
class Node:
    def __init__(self, name, point, parent, DistFromParent,trajectory):
        self.name = name
        self.point = point
        self.parent = parent
        self.DistFromParent = DistFromParent
        self.trajectory = trajectory
        

#-----------------------------------------------------------------------------#
# Class:    Edge
# Purpose:  Edge objects that connect the string vertices in graphs.
#
# Attributes/Properties:
#   - v1:           Node- the first vertex (start)
#   - v2:           Node- the second vertex (next)
#   - trajectory:   np.array- the ode trajectory from the start to end state
#
# Functions:
#   - __init__: the constructor
#-----------------------------------------------------------------------------#
class Edge:
    def __init__(self, v1, v2, trajectory):
        self.v1 = v1
        self.v2 = v2
        self.trajectory = trajectory

#-----------------------------------------------------------------------------#
# Functions/Methods
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Function: DroneDynamics
# Purpose:  ODE function for the ode45 integrator for the drone's dynamics.
#
# Inputs:
#   - t:        int- required input for ode integrator
#   - state:    np.array- 6x1 current state of the drone
#   - u1:       double- control input for omega angular acceleration
#   - u2:       double- control input for alpha angular acceleration
#   - u3:       double- control input for linear acceleration
#
# Outputs:
#   - next_state: np.array- 6x1 next state of the drone
#-----------------------------------------------------------------------------#
def DroneDynamics(t,state,u1,u2,u3): #ode function
    
    x,y,z,psi,theta,v = state
    omega,alpha,a = u1,u2,u3
    
    #input state: [x,y,z,psi,theta,v]
    xdot = v*np.cos(psi)*np.cos(theta)
    ydot = v*np.sin(psi)*np.cos(theta)
    zdot = v*np.sin(theta) 
    psidot = omega
    thetadot = alpha 
    vdot = a

    return [xdot,ydot,zdot,psidot,thetadot,vdot]


#-----------------------------------------------------------------------------#
# Function: RoverDynamics
# Purpose:  ODE function for the ode45 integrator for the rover's dynamics.
#
# Inputs:
#   - t:        int- required input for ode integrator
#   - state:    np.array- 6x1 current state of the rover
#   - uv:       double- control input for linear acceleration
#   - up:       double- control input for theta angular acceleration
#
# Outputs:
#   - next_state: np.array- 6x1 next state of the rover
#-----------------------------------------------------------------------------#
def RoverDynamics(t,state,uv,up): #ode function
    
    x,y,theta = state
    
    L = 1 #car length
    #input state: [x,y,theta]
    xdot = uv*np.cos(theta)
    ydot = uv*np.sin(theta)
    thetadot = uv/L*np.tan(up)

    return [xdot,ydot,thetadot]


#-----------------------------------------------------------------------------#
# Function: MultiAgentDynamics
# Purpose:  ODE function for the ode45 integrator for the rover's dynamics.
#
# Inputs:
#   - t:        int- required input for ode integrator
#   - state:    np.array- 6x1 current state of the rover
#   - uv:       double- control input for linear acceleration
#   - up:       double- control input for theta angular acceleration
#
# Outputs:
#   - next_state: np.array- 6x1 next state of the rover
#-----------------------------------------------------------------------------#
def MultiAgentDynamics(t,state,u1,u2,u3,uv,up):
    x,y,z,psi,theta,v,xr,yr,tr = state
    omega,alpha,a = u1,u2,u3
    
    #input state: [x,y,z,psi,theta,v]
    xdot = v*np.cos(psi)*np.cos(theta)
    ydot = v*np.sin(psi)*np.cos(theta)
    zdot = v*np.sin(theta) 
    psidot = omega
    thetadot = alpha 
    vdot = a

    L = 1 #car length
    #input state: [x,y,theta]
    xrdot = uv*np.cos(tr)
    yrdot = uv*np.sin(tr)
    trdot = uv/L*np.tan(up)

    return [xdot,ydot,zdot,psidot,thetadot,vdot,xrdot,yrdot,trdot]

#-----------------------------------------------------------------------------#
# Function: generateNode
# Purpose:  ODE function for the ode45 integrator for the rover's dynamics.
#
# Inputs:
#   - Q:        double- probability for introducing goal bias
#   - q_goal:   np.array- 6x1 goal state of the agent
#
# Outputs:
#   - q_rand:   np.array- 6x1 sampled/goal-biased state of the agent
#-----------------------------------------------------------------------------#
def generateNode(Q,q_goal): #generate 6 state 
    
    chance = np.random.uniform(0,1,1)
    y_min = 0
    y_max = 15
    x_min = 0
    x_max = 15
    z_min = 0
    z_max = 15
    v_min = -1
    v_max = 1
    theta_min = -np.pi/3
    theta_max = np.pi/3
    
    if chance < Q:
        x = np.random.uniform(x_min,x_max)
        y = np.random.uniform(y_min,y_max)
        z = np.random.uniform(z_min,z_max)
        v = np.random.uniform(v_min,v_max)
        p = np.random.uniform(-np.pi/2,np.pi/2)
        ta = np.random.uniform(theta_min,theta_max)

        xr = np.random.uniform(x_min,x_max)
        yr = np.random.uniform(y_min,y_max)
        tr = np.random.uniform(-np.pi/2,np.pi/2)

        #single agent tests
        # q_rand = [x,y,z,p,ta,v]
        # q_rand = [xr,yr,tr]

        #for multi agent 
        q_rand = [x,y,z,p,ta,v,xr,yr,tr]
        
        
    elif chance >= Q:
        q_rand = q_goal
        
    return q_rand



#-----------------------------------------------------------------------------#
# Function: GenerateTrajectory
# Purpose:  Function to generate a trajectory between two states that gets as
#           close as possible to the inputted new_state with random control
#           inputs.
#
# Inputs:
#   - state:        np.array- 6x1 initial state of the agent (starting point)
#   - new_state:    np.array- 6x1 new sampled/goal-biased state of the agent
#
# Outputs:
#   - trajectories: np.array- 6x1 states of the agent for the best trajectory
#   - time:         np.array- time array for the trajectory (from ode solver)
#-----------------------------------------------------------------------------#
def GenerateTrajectory(state,new_state): #state is qnear and new_state is q_rand
    
    m = 3 #generate 3 different random controls that extend from state which is qnear
    t_span = (0.0,0.5) #seconds
    min_dist = math.inf
    
    for i in range(m):
        u1 = np.random.uniform(-np.pi/6, np.pi/6)
        u2 = np.random.uniform(-np.pi/6, np.pi/6)
        u3 = np.random.uniform(-1/2, 1/2)
        control = (u1,u2,u3)
        uv = np.random.uniform(-1,1)
        up = np.random.uniform(-np.pi/2, np.pi/2)
        control_rover = (uv,up)
        #.y gets the results, .t gets the time

        control_multi = (u1,u2,u3,up,uv)

        #for one agent
        # result_solve_ivp = solve_ivp(DroneDynamics, t_span, state,args=control, method = 'RK45')
        # result_solve_ivp = solve_ivp(RoverDynamics, t_span, state,args=control_rover, method = 'RK45')

        # traj = result_solve_ivp.y
        # best_con = math.dist(new_state,traj[:,-1])
        # if best_con < min_dist:
        #     min_dist = best_con
        #     trajectories = traj
        #     time = result_solve_ivp.t

        #this will be used for multi agent
        result_solve_ivp = solve_ivp(MultiAgentDynamics, t_span, state,args=control_multi, method = 'RK45')
    
        traj = result_solve_ivp.y
        best_con = math.dist(new_state,traj[:,-1])
        if best_con < min_dist:
            min_dist = best_con
            trajectories = traj
            time = result_solve_ivp.t
    
    return trajectories,time

#-----------------------------------------------------------------------------#
# Function: TrajectoryValid
# Purpose:  Function to check if a generated trajectory between two states is 
#           valid, i.e., no state in the trajectory exceeds the dynamic
#           constraints placed on the agent or collides with an obstacle
#
# Inputs:
#   - trajectories: np.array- 6x1 states of the agent for the best trajectory
#   - time:         np.array- time array for the trajectory (from ode solver)
#   - obstacles:    list of fcl objects used for 3d collision checking
#
# Outputs:
#   - boolean:  true if the trajectory is valid for all states, false if not         
#-----------------------------------------------------------------------------#
def TrajectoryValid(trajectories,time, obstacles = []): 

    #for single agent
    # x,y,z,ta,v = trajectories[0,:],trajectories[1,:],trajectories[2,:],trajectories[4,:],trajectories[5,:]
    # xr,yr = trajectories[0,:],trajectories[1,:]

    # for multi agent
    x,y,z,ta,v,xr,yr =  trajectories[0,:],trajectories[1,:],trajectories[2,:],trajectories[4,:],trajectories[5,:],trajectories[6,:],trajectories[7,:]

    # # DRONE
    # agent = fcl.Sphere(0.2)

    # # ROVER
    # # agent = fcl.Box(np.array([1.0, 1.0, 1.0]))

    # Valid = 1 # Let's assume the trajectory is valid until proven invalid

    # for i in range(len(time)): 
    #     # if (0<=xr[i]<=11) and (0<=yr[i]<=10):
    #     if (0<=x[i]<=11) and (0<=y[i]<=10) and (0<=z[i]<=10) and (-1<=v[i]<=1) and (-np.pi/3<=ta[i]<=np.pi/3):
    #         Valid = 1
    #     else:
    #         Valid = 0
    #         break

    #     # Collision check
    #     # M_agent = fcl.Transform3f(np.eye(3), np.array([xr[i], yr[i], 0.5]))
    #     M_agent = fcl.Transform3f(np.eye(3), np.array([x[i], y[i], z[i]]))

    #     req = fcl.CollisionRequest()
    #     res = fcl.CollisionResult()

    #     # loop over obstacles
    #     for obs_i in range(len(obstacles)):

    #         # create obstacle
    #         obstacle = obstacles[obs_i][0]
    #         M_obstacle = obstacles[obs_i][1]

            
    #         if fcl.collide(agent, M_agent, obstacle, M_obstacle, req, res):
    #             Valid = 0
    #             break

    # return Valid

    # DRONE
    drone = fcl.Sphere(0.2)

    # ROVER
    rover = fcl.Box(np.array([1.0, 1.0, 1.0]))

    Valid = 1 # Let's assume the trajectory is valid until proven invalid

    for i in range(len(time)): 
        # if (0<=xr[i]<=11) and (0<=yr[i]<=10):
        if (0<=x[i]<=11) and (0<=y[i]<=10) and (0<=z[i]<=10) and (-1<=v[i]<=1) and (-np.pi/3<=ta[i]<=np.pi/3) and (0<=xr[i]<=11) and (0<=yr[i]<=10):
            Valid = 1
        else:
            Valid = 0
            break

        # Collision check
        M_rover = fcl.Transform3f(np.eye(3), np.array([xr[i], yr[i], 0.5]))
        M_drone = fcl.Transform3f(np.eye(3), np.array([x[i], y[i], z[i]]))

        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()

        # Check if rover and drone collide
        if fcl.collide(rover, M_rover, drone, M_drone, req, res):
            Valid = 0
            break

        # loop over obstacles
        for obs_i in range(len(obstacles)):

            # create obstacle
            obstacle = obstacles[obs_i][0]
            M_obstacle = obstacles[obs_i][1]

            if fcl.collide(drone, M_drone, obstacle, M_obstacle, req, res):
                Valid = 0
                break

            if fcl.collide(rover, M_rover, obstacle, M_obstacle, req, res):
                Valid = 0
                break

    return Valid


#-----------------------------------------------------------------------------#
# Function: plt_sphere
# Purpose:  Function to plot a sphere in 3D space to visualize the drone.
#
# Inputs:
#   - ax:           matlplotlib axis object
#   - list_center:  np.array- x,y,z location of the drone (its center point)
#   - list_radius:  np.array- radii of the drone at each step
#
# Outputs:
#   - None- just plots spheres in the matplotlib workspace figure
#-----------------------------------------------------------------------------#
def plt_sphere(ax, list_center, list_radius, color, alpha):

  for c, r in zip(list_center, list_radius):
    # ax = fig.gca(projection='3d')

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)

    ax.plot_surface(x+c[0], y+c[1], z+c[2], color = color, alpha = alpha) # color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)


#-----------------------------------------------------------------------------#
# Function: create_rrt
# Purpose:  Implementation of a RRT planner.
#
# Inputs:
#   - start:        np.array- start state vector
#   - goal:         np.array- goal state vector
#   - n:            int- maximum number of iterations the algorithm should run
#   - Q:            double- goal bias probability (how often bias should occur)
#   - obstacles:    list of all obstacles in the workspace
#   - plot_path:    boolean- true if a path should be plotted, false if not
#
# Outputs:
#   - solution_found:   boolean, true if a solution was found, false if not
#   - path:             np.array- vector of nodes for all states in final path
#   - path_length:      double- length of the path
#   - tree_size:        int- the size of the tree, or the number of nodes
#   - kino_path:        np.array- vector of states for all sub-trajectories in
#                       final path
#-----------------------------------------------------------------------------#
def create_rrt(start, goal, n, Q, rover_parked, drone_parked, plot_path = False, obstacles = []):


    # STEP 1: Initialize tree with root/start node
    tree = [Node(0, start, None, 0.0,[])]

    curr_node = tree[0]

    #single agent
    # x,y,z,p,ta,v = start
    # xr,yr,tr = start

    #centralized multi agent
    x,y,z,p,ta,v,xr,yr,tr = start
    solution_found = True

    # STEP 2: Loop until n samples created or goal reached
    #multi agent
    # while (len(tree) < n) and not(9<=x<=11 and 8<=y<=10 and 3<=z<=5 and -1/20<=v<=1/20 and 9<=xr<=11 and 8<=yr<=10):

    #single agent 
    # while (len(tree) < n) and not(9<=xr<=11 and 8<=yr<=10):
    while (len(tree) < n) and not(9<=x<=11 and 8<=y<=10 and 3<=z<=5 and -1/20<=v<=1/20 and 9<=xr<=11 and 8<=yr<=10):

        # print(len(tree))
        # print(math.dist(curr_node.point, goal))
        
        new_state = generateNode(Q,goal)

        if rover_parked:
            new_state[6:] = rover_state_last
    
        if drone_parked:
            new_state[0:6] = drone_state_last


        # Find node from tree that is closest to q_rand

        state = None # TODO:

        min_dist = math.inf

        for node in tree:

            curr_dist = math.dist(node.point, new_state)
            
            if curr_dist < min_dist:
                min_dist = curr_dist
                state = node
         
        #get the trajectory of the sampled state
        trajectories,time =  GenerateTrajectory(state.point,new_state)

        Valid = TrajectoryValid(trajectories,time, obstacles)

        # Check if q_new collides with obstacles
        if Valid:
            # Add new Node
            x_new = trajectories[:,-1]

            #single agent tests
            # x,y,z,p,ta,v = x_new
            # xr,yr,tr = x_new

            #multi agent
            x,y,z,p,ta,v,xr,yr,tr = x_new

            if drone_parked == False and 9<=x<=11 and 8<=y<=10 and 3<=z<=5 and -1/20<=v<=1/20:
                drone_parked = True
                drone_state_last = x_new[0:6]

            if rover_parked == False and 9<=xr<=11 and 8<=yr<=10:
                rover_parked = True
                rover_state_last = x_new[6:]

            #name, new node, parent node, distance from parent node
            curr_node = Node(len(tree), x_new, state, math.dist(x_new, state.point),trajectories)
           
            tree.append(curr_node) #appends an object

    if len(tree)>=n:
        solution_found = False

    path = []
    kino_path = []
    path_length = 0.0

    # STEP 3: Create path from goal to start going up the tree
    if solution_found == True:

        curr_node = tree[-1]

        while curr_node != tree[0]:

            path.insert(0, curr_node)
            kino_path.insert(0,curr_node.trajectory)
            path_length += curr_node.DistFromParent
            curr_node = curr_node.parent
            

        path.insert(0, curr_node)
        
    return solution_found, path, path_length, len(tree),kino_path

    

#Bounds that are allowable 
x_goal = [9,10]
y_goal = [8,9]
z_goal = [4,5]
v_goal = [-1/20,1/20]


#-----------------------------------------------------------------------------#
# Function: plot_rectangular_prism
# Purpose:  Function to plot a rectangular prism in 3D space to visualize the
#           rover and stalagmite/stalactite obstacles.
#
# Inputs:
#   - ax:       matlplotlib axis object
#   - x_bounds: list- the x limits/bounds for the prism
#   - y_bounds: list- the y limits/bounds for the prism
#   - z_bounds: list- the z limits/bounds for the prism
#   - color:    string- the color of the prism's faces
#   - alpha:    double- the transparency of the prism's faces
#
# Outputs:
#   - None- just plots spheres in the matplotlib workspace figure
#-----------------------------------------------------------------------------#
def plot_rectangular_prism(ax, x_bound_arr, y_bound_arr, z_bound_arr, color, alpha):

    # Vertices

    for i in range(np.size(x_bound_arr, 0)):

        x_bounds = x_bound_arr[i,:]
        y_bounds = y_bound_arr[i,:]
        z_bounds = z_bound_arr[i,:]


        # For now, draw 6 rectangles
        face1 = plt.Polygon([[x_bounds[0], z_bounds[0]], \
                            [x_bounds[0], z_bounds[1]], \
                            [x_bounds[1], z_bounds[1]], \
                            [x_bounds[1], z_bounds[0]]], color = color, alpha = alpha)
        
        face2 = plt.Polygon([[x_bounds[0], z_bounds[0]], \
                            [x_bounds[0], z_bounds[1]], \
                            [x_bounds[1], z_bounds[1]], \
                            [x_bounds[1], z_bounds[0]]], color = color, alpha = alpha)

        ax.add_patch(face1)
        ax.add_patch(face2)
        art3d.patch_2d_to_3d(face1, z = y_bounds[0], zdir = "y")
        art3d.patch_2d_to_3d(face2, z = y_bounds[1], zdir = "y")

        face3 = plt.Polygon([[y_bounds[0], z_bounds[0]], \
                            [y_bounds[0], z_bounds[1]], \
                            [y_bounds[1], z_bounds[1]], \
                            [y_bounds[1], z_bounds[0]]], color = color, alpha = alpha)

        face4 = plt.Polygon([[y_bounds[0], z_bounds[0]], \
                            [y_bounds[0], z_bounds[1]], \
                            [y_bounds[1], z_bounds[1]], \
                            [y_bounds[1], z_bounds[0]]], color = color, alpha = alpha)
        
        ax.add_patch(face3)
        art3d.pathpatch_2d_to_3d(face3, z = x_bounds[0], zdir = "x")
        ax.add_patch(face4)
        art3d.pathpatch_2d_to_3d(face4, z = x_bounds[1], zdir = "x")

        face5 = plt.Polygon([[x_bounds[0], y_bounds[0]], \
                            [x_bounds[0], y_bounds[1]], \
                            [x_bounds[1], y_bounds[1]], \
                            [x_bounds[1], y_bounds[0]]], color = color, alpha = alpha)
        
        face6 = plt.Polygon([[x_bounds[0], y_bounds[0]], \
                            [x_bounds[0], y_bounds[1]], \
                            [x_bounds[1], y_bounds[1]], \
                            [x_bounds[1], y_bounds[0]]], color = color, alpha = alpha)
        
        ax.add_patch(face5)
        art3d.pathpatch_2d_to_3d(face5, z = z_bounds[0], zdir = "z")
        ax.add_patch(face6)
        art3d.pathpatch_2d_to_3d(face6, z = z_bounds[1], zdir = "z")

if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Set up inputs for the workspace
    #-----------------------------------------------------------------------------#
    #single agent tests

    # Initial rover location (x,y) and heading (t)
    xr,yr,tr = 1,1,0

    # Initial drone location (x, y, z), orientation (psi, theta), and velocity (v)
    x,y,z,psi,theta,v = 1,1,1.5,0,0,0

    # Obstacles                                                                       x     y    z
                # Stalagmites (on ground)
    obstacles = [[fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([1.0, 5.0, 1.0]))],

                [fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([14.0, 5.0, 1.0]))],
                # Stalactites (on ceiling)
                 [fcl.Box(np.array([15, 2, 8])), fcl.Transform3f(np.eye(3), np.array([7.5, 5.0, 6.0]))]]

    # For single agents:

    # ROVER
    # start = [xr,yr,tr] 
    # goal = [9.5,8.5,0]

    # DRONE
    # start = [x,y,z,psi,theta,v] 
    # goal = [9.5,8.5,4.5,np.random.uniform(-np.pi/2,np.pi/2),0,0]
    
    # For multi-agent, or [DRONE, ROVER]:

    start = [x,y,z,psi,theta,v,xr,yr,tr]
    goal = [9.5,8.5,4.5,np.random.uniform(-np.pi/2,np.pi/2),0,0,9.5,8.5,0]
    # goal = [10,9,4,np.random.uniform(-np.pi/2,np.pi/2),0,0,10,9,0]

    # number of iterations, n
    n = 5000
    
    # goal bias probability
    p_goal = 0.05

    # random sample state probability
    Q = 1-p_goal

    #-----------------------------------------------------------------------------#
    # Run the RRT planner on these inputs
    #-----------------------------------------------------------------------------#

                                                                                    # drone_parked, rover_parked, plot_path
    solution_found, path, path_length, tree_size,kino_path = create_rrt(start, goal, n, Q, False, False, True, obstacles)
    

    #kinopath is the kinodynamic path
    x,y,z,p,ta,v = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    xr,yr,tr = np.array([]),np.array([]),np.array([])

    # Extract each state from the kinodynamic solution path
    for i in range(len(kino_path)):

        # DRONE
        x = np.append(x,kino_path[i][0])
        y = np.append(y,kino_path[i][1])
        z= np.append(z,kino_path[i][2])
        p = np.append(p,kino_path[i][3])
        ta = np.append(ta,kino_path[i][4])
        v = np.append(v,kino_path[i][5])

        # ROVER
        # xr = np.append(xr,kino_path[i][0]) # or index 6
        # yr = np.append(yr,kino_path[i][1]) # or index 7
        # tr = np.append(tr,kino_path[i][2]) #or index at 8 

        xr = np.append(xr,kino_path[i][6]) # or index 6
        yr = np.append(yr,kino_path[i][7]) # or index 7
        tr = np.append(tr,kino_path[i][8]) #or index at 8 

    # Array of constant z values to plot the rover on the same 3D plot
    zr = 0.5 * np.ones(len(xr)) #this will be for plotting on the same graph 

    # Reorder the states to plot spheres/rectangular prisms along path
    drone_radius = [.2]*len(x)
    rover_side_length = [1] * len(x)

    list_center = []

    for i in range(len(x)):
        list_center.insert(0,(x[i],y[i],z[i]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot(x,y,z)
    ax.plot(xr,yr,zr)

    # Plot spheres to represent the drone- only 30 spheres will be plotted along the path (too slow otherwise)
    if len(list_center) > 30:
    # if len(xr) > 30:
        # DRONE
        plt_sphere(ax, list_center[::math.floor(len(list_center) / 30.0)], drone_radius, 'k', 0.7) 

        # ROVER
        x_left = xr[::math.floor(len(xr) / 15.0)] - 0.5
        x_right = xr[::math.floor(len(xr) / 15.0)] + 0.5

        y_left = yr[::math.floor(len(yr) / 15.0)] - 0.5
        y_right = yr[::math.floor(len(yr) / 15.0)] + 0.5

        z_left = zr[::math.floor(len(zr) / 15.0)] - 0.5
        z_right = zr[::math.floor(len(zr) / 15.0)] + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'k', 0.5)

    elif len(list_center) > 0:
        # DRONE
        plt_sphere(ax, list_center, drone_radius, 'k', 0.7) 

        # ROVER
        x_left = xr - 0.5
        x_right = xr + 0.5

        y_left = yr - 0.5
        y_right = yr + 0.5

        z_left = zr - 0.5
        z_right = zr + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left, x_right), axis = 1), np.concatenate((y_left, y_right), axis = 1), np.concatenate((z_left, z_right), axis = 1), 'k', 0.5)


    # Plot start and goal points

    # DRONE
    ax.scatter(start[0], start[1], start[2], color = 'b')
    ax.scatter(goal[0], goal[1], goal[2], color = 'g')

    # ROVER
    ax.scatter(start[0], start[1], 0.5, color = 'b')
    ax.scatter(goal[0], goal[1], 0.5, color = 'g')

    # Plot goal region

    # DRONE
    plot_rectangular_prism(ax, np.array([[8.5, 10.5]]), np.array([[7.5, 9.5]]), np.array([[3.5, 5.5]]), 'g', 0.2)

    # ROVER
    plot_rectangular_prism(ax, np.array([[8.5, 10.5]]), np.array([[7.5, 9.5]]), np.array([[0.0, 2.0]]), 'g', 0.2)

    # Plot obstacles    
    plot_rectangular_prism(ax, np.array([[0.0, 2.0]]), np.array([[4.0, 6.0]]), np.array([[0.0, 2.0]]), 'r', 0.2)
    
    plot_rectangular_prism(ax, np.array([[13.0, 15.0]]), np.array([[4.0, 6.0]]), np.array([[0.0, 2.0]]), 'r', 0.2)
    # plot_rectangular_prism(ax, np.array([[4.0, 6.0]]), np.array([[4.0, 6.0]]), np.array([[8.0, 10.0]]), 'r', 0.2)

    
    # [fcl.Box(np.array([15, 2, 8])), fcl.Transform3f(np.eye(3), np.array([7.5, 3.0, 6.0]))]
    
    plot_rectangular_prism(ax, np.array([[0.0, 15.0]]), np.array([[4.0, 6.0]]), np.array([[2.0, 10.0]]), 'r', 0.2)

    # Label plot and axes
    ax.set_title('Rover and Quadrotor Trajectory')
    # ax.set_title('Quadrotor Trajectory')
    # ax.set_title('Rover Trajectory')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set workspace bounds
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 15.0)
    ax.set_zlim(0.0, 10.0)

    # Show the plot
    plt.show()
