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
import random

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
def MultiAgentDynamics(t,state,u1,u2,u3,uv,up, mode = 1):
    x,y,z,psi,theta,v,xr,yr,tr = state
    omega,alpha,a = u1,u2,u3

    if mode == 1:
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

    elif mode == 0:
        L = 1
        xrdot = uv*np.cos(tr)
        yrdot = uv*np.sin(tr)
        trdot = uv/L*np.tan(up)

        xdot = xrdot
        ydot = yrdot
        zdot = 0.0 #1.2 # z
        psidot = 0 # 0.0 # psi
        thetadot = 0
        vdot = 0
    
    # #input state: [x,y,z,psi,theta,v]
    # xdot = v*np.cos(psi)*np.cos(theta)
    # ydot = v*np.sin(psi)*np.cos(theta)
    # zdot = v*np.sin(theta) 
    # psidot = omega
    # thetadot = alpha 
    # vdot = a

    # L = 1 #car length
    # #input state: [x,y,theta]
    # xrdot = uv*np.cos(tr)
    # yrdot = uv*np.sin(tr)
    # trdot = uv/L*np.tan(up)

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
def generateNode(Q,q_goal, W_bounds): #generate 6 state 
    
    chance = np.random.uniform(0,1,1)
    y_min = 0
    y_max = W_bounds[1]
    x_min = 0
    x_max = W_bounds[0]
    z_min = 0
    z_max = W_bounds[2]
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

def generateNodeLanding(Q,q_goal,bias): #generate 6 state 
    
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

        if bias == False:   
            mode = random.randint(0,1)
        elif bias == True:
            mode = 0

        #single agent tests
        # q_rand = [x,y,z,p,ta,v]
        # q_rand = [xr,yr,tr]

        #for multi agent 
        q_rand = [x,y,z,p,ta,v,xr,yr,tr, mode]
        
        
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
def GenerateTrajectory(state,new_state, mode = 1): #state is qnear and new_state is q_rand
    
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
def GenerateTrajectoryLanding(state,new_state): #state is qnear and new_state is q_rand
    
    m = 3 #generate 3 different random controls that extend from state which is qnear
    t_span = (0.0,0.5) #seconds
    min_dist = math.inf
    mode = state[9]
    state = state[0:9]
    for i in range(m):

        u1 = np.random.uniform(-np.pi/6, np.pi/6)
        u2 = np.random.uniform(-np.pi/6, np.pi/6)
        u3 = np.random.uniform(-1/2, 1/2)
        uv = np.random.uniform(-1,1)
        up = np.random.uniform(-np.pi/2, np.pi/2)
        #.y gets the results, .t gets the time


        control_multi = (u1,u2,u3,up,uv,mode)

        
        #this will be used for multi agent
        result_solve_ivp = solve_ivp(MultiAgentDynamics, t_span, state,args=control_multi, method = 'RK45')
    
        traj = result_solve_ivp.y

        if mode == 0:
            for j in range(np.shape(traj)[1]): # [:,-1]
                traj[0:2, j] = traj[6:8, j]
                traj[2, j] = 1.3
                traj[3, j] = 0.0
                traj[4, j] = 0.0
                traj[5, j] = 0.0

        best_con = math.dist(new_state[0:9],traj[:,-1])
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
def TrajectoryValid(trajectories,time, W_bounds, obstacles = [], mode = 1): 

    #for single agent
    # x,y,z,ta,v = trajectories[0,:],trajectories[1,:],trajectories[2,:],trajectories[4,:],trajectories[5,:]
    # xr,yr = trajectories[0,:],trajectories[1,:]

    # for multi agent
    x,y,z,ta,v,xr,yr =  trajectories[0,:],trajectories[1,:],trajectories[2,:],trajectories[4,:],trajectories[5,:],trajectories[6,:],trajectories[7,:]

    # DRONE
    drone = fcl.Sphere(0.2)

    # ROVER
    rover = fcl.Box(np.array([1.0, 1.0, 1.0]))

    Valid = 1 # Let's assume the trajectory is valid until proven invalid

    if mode == 1:

        for i in range(len(time)): 
            # if (0<=xr[i]<=11) and (0<=yr[i]<=10):
            if (0<=x[i]<=14.5) and (0<=y[i]<=14.5) and (0<=z[i]<=10) and (-1<=v[i]<=1) and (-np.pi/3<=ta[i]<=np.pi/3) and (0<=xr[i]<=14) and (0<=yr[i]<=14):
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

    elif mode == 0:

        for i in range(len(time)): 
            # if (0<=xr[i]<=11) and (0<=yr[i]<=10):
            if (0<=xr[i]<=14) and (0<=yr[i]<=14):
                Valid = 1
            else:
                Valid = 0
                break

            # Collision check
            M_rover = fcl.Transform3f(np.eye(3), np.array([xr[i], yr[i], 0.5]))

            req = fcl.CollisionRequest()
            res = fcl.CollisionResult()

            # loop over obstacles
            for obs_i in range(len(obstacles)):

                # create obstacle
                obstacle = obstacles[obs_i][0]
                M_obstacle = obstacles[obs_i][1]

                if fcl.collide(rover, M_rover, obstacle, M_obstacle, req, res):
                    Valid = 0
                    break

    # for i in range(len(time)): 
    #     # if (0<=xr[i]<=11) and (0<=yr[i]<=10):
    #     if (0 <= x[i] <= W_bounds[0]) and (0 <= y[i] <= W_bounds[1]) and (0 <= z[i] <= W_bounds[2]) and \
    #         (-1<=v[i]<=1) and (-np.pi/3<=ta[i]<=np.pi/3) and \
    #         (0 <= xr[i] <= W_bounds[0]) and (0 <= yr[i] <= W_bounds[1]):
    #         Valid = 1
    #     else:
    #         Valid = 0
    #         break

    #     # Collision check
    #     M_rover = fcl.Transform3f(np.eye(3), np.array([xr[i], yr[i], 0.5]))
    #     M_drone = fcl.Transform3f(np.eye(3), np.array([x[i], y[i], z[i]]))

    #     req = fcl.CollisionRequest()
    #     res = fcl.CollisionResult()

    #     # Check if rover and drone collide
    #     if fcl.collide(rover, M_rover, drone, M_drone, req, res):
    #         Valid = 0
    #         break

    #     # loop over obstacles
    #     for obs_i in range(len(obstacles)):

    #         # create obstacle
    #         obstacle = obstacles[obs_i][0]
    #         M_obstacle = obstacles[obs_i][1]

    #         if fcl.collide(drone, M_drone, obstacle, M_obstacle, req, res):
    #             Valid = 0
    #             break

    #         if fcl.collide(rover, M_rover, obstacle, M_obstacle, req, res):
    #             Valid = 0
    #             break

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
def create_rrt(W_bounds, start, goal, n, Q, rover_parked, drone_parked, plot_path = False, obstacles = []):


    # STEP 1: Initialize tree with root/start node
    tree = [Node(0, start, None, 0.0,[])]

    curr_node = tree[0]

    #single agent
    # x,y,z,p,ta,v = start
    # xr,yr,tr = start

    #centralized multi agent
    x,y,z,p,ta,v,xr,yr,tr = start
    solution_found = True

    x_goal_bounds = [goal[0] - 1.0, goal[0] + 1.0]
    y_goal_bounds = [goal[1] - 1.0, goal[1] + 1.0]
    z_goal_bounds = [goal[2] - 1.0, goal[2] + 1.0]
    xr_goal_bounds = [goal[6] - 1.0, goal[6] + 1.0]
    yr_goal_bounds = [goal[7] - 1.0, goal[7] + 1.0]

    # STEP 2: Loop until n samples created or goal reached
    #multi agent
    # while (len(tree) < n) and not(9<=x<=11 and 8<=y<=10 and 3<=z<=5 and -1/20<=v<=1/20 and 9<=xr<=11 and 8<=yr<=10):

    #single agent 
    # while (len(tree) < n) and not(9<=xr<=11 and 8<=yr<=10):


    while (len(tree) < n) and not(x_goal_bounds[0]<=x<=x_goal_bounds[1] and \
                                y_goal_bounds[0]<=y<=y_goal_bounds[1] and \
                                z_goal_bounds[0]<=z<=z_goal_bounds[1] and \
                                -1/20<=v<=1/20 and \
                                xr_goal_bounds[0]<=xr<=x_goal_bounds[1] and \
                                yr_goal_bounds[0]<=yr<=yr_goal_bounds[1]):

        # print(len(tree))
        # print(math.dist(curr_node.point, goal))
        
        new_state = generateNode(Q,goal, W_bounds)

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

        Valid = TrajectoryValid(trajectories,time, W_bounds, obstacles)

        # Check if q_new collides with obstacles
        if Valid:
            # Add new Node
            x_new = trajectories[:,-1]

            #single agent tests
            # x,y,z,p,ta,v = x_new
            # xr,yr,tr = x_new

            #multi agent
            x,y,z,p,ta,v,xr,yr,tr = x_new

            if drone_parked == False and x_goal_bounds[0]<=x<=x_goal_bounds[1] and \
                                        y_goal_bounds[0]<=y<=y_goal_bounds[1] and \
                                        z_goal_bounds[0]<=z<=z_goal_bounds[1] and -1/20<=v<=1/20:
                drone_parked = True
                drone_state_last = x_new[0:6]

            if rover_parked == False and xr_goal_bounds[0]<=xr<=xr_goal_bounds[1] and yr_goal_bounds[0]<=yr<=yr_goal_bounds[1]:
                rover_parked = True
                rover_state_last = x_new[6:]

            #name, new node, parent node, distance from parent node
            curr_node = Node(len(tree), x_new, state, math.dist(x_new, state.point),trajectories)
        
            tree.append(curr_node) #appends an object

    path = []
    kino_path = []
    path_length = 0.0

    if len(tree)>=n:
        solution_found = False

        return solution_found, path, path_length, len(tree),kino_path

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
def create_rrt_landing(W_bounds, start, goal_array, n, Q, rover_parked, drone_parked, plot_path = False, obstacles = []):


    # STEP 1: Initialize tree with root/start node
    tree = [Node(0, start, None, 0.0,[])]

    curr_node = tree[0]

    #single agent
    # x,y,z,p,ta,v = start
    # xr,yr,tr = start

    #centralized multi agent
    x,y,z,p,ta,v,xr,yr,tr,mode = start
    solution_found = True

    # STEP 2: Loop until n samples created or goal reached
    #multi agent
    # while (len(tree) < n) and not(9<=x<=11 and 8<=y<=10 and 3<=z<=5 and -1/20<=v<=1/20 and 9<=xr<=11 and 8<=yr<=10):

    #single agent 
    # while (len(tree) < n) and not(9<=xr<=11 and 8<=yr<=10):


    i = 0
    bias = False
    while (len(tree) < n) and i < 2: #not(drone_parked and rover_parked and i == 2):

        # print(len(tree))
        # print(math.dist(curr_node.point, goal))

        goal = goal_array[i]

        if i == 0:
            goal[0] = float(xr)
            goal[1] = float(yr)

        
        new_state = generateNodeLanding(Q,goal,bias)

        if rover_parked:
            new_state[6:9] = rover_state_last
    

        # Find node from tree that is closest to q_rand

        state = None # TODO:

        min_dist = math.inf

        for node in tree:

            curr_dist = math.dist(node.point, new_state)
            
            if curr_dist < min_dist:
                min_dist = curr_dist
                state = node
         
        #get the trajectory of the sampled state
        mode = state.point[9]
        trajectories,time =  GenerateTrajectoryLanding(state.point,new_state)

        Valid = TrajectoryValid(trajectories,time, W_bounds, obstacles, mode)

        # Check if q_new collides with obstacles
        if Valid:
            # Add new Node
            x_new = trajectories[:,-1]
            x_new = np.append(x_new,mode)

            #single agent tests
            # x,y,z,p,ta,v = x_new
            # xr,yr,tr = x_new

            #multi agent
            x,y,z,p,ta,v,xr,yr,tr,mode = x_new
            r_low = [[xr-1,yr-1,1.3,0,2],[0,6.5,1.3,0,6.5]] #x,y,z,xr,yr
            r_high = [[xr+1,yr+1,2,15,4],[15,7.5,1.5,15,7.5]] #x,y,z,xr,yr

            if i == 0 and drone_parked == False and r_low[i][0]<=x<=r_high[i][0] and r_low[i][1]<=y<=r_high[i][1]  and r_low[i][2]<=z<=r_high[i][2] and -1/20<=v<=1/20 and r_low[i][4]<=yr<=r_high[i][4]:
                print("Drone parked")
                drone_parked = True
                #drone_state_last = x_new[0:6]
                x_new[9] = 0  #this will only work for my current goal, will need to change with if i = _for other goal
                bias = True


            if i == 0 and rover_parked == False and r_low[i][4]<=yr<=r_high[i][4]:# r_low[i][3]<=xr<=r_high[i][3] and r_low[i][4]<=yr<=r_high[i][4]:
                print("Rover parked")
                rover_parked = True
                rover_state_last = x_new[6:9]

            if i == 1 and rover_parked == False and r_low[i][4]<=yr<=r_high[i][4] and mode == 0:# r_low[i][3]<=xr<=r_high[i][3] and r_low[i][4]<=yr<=r_high[i][4]:
                print("Rover and Drone parked")
                rover_parked = True
                rover_state_last = x_new[6:9]

            #name, new node, parent node, distance from parent node
            curr_node = Node(len(tree), x_new, state, math.dist(x_new, state.point),trajectories)
           
            tree.append(curr_node) #appends an object

        if i == 0 and drone_parked and rover_parked:
            i +=1 #both have met goal, continue to next go, will terminate at i = 2 when both have met the second goal
            rover_parked = False
            drone_parked = False


        if i == 1 and rover_parked:
            i +=1 #both have met goal, continue to next go, will terminate at i = 2 when both have met the second goal


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

    W_bounds = [15.0, 15.0, 10.0] # x, y, z

    # Initial rover location (x,y) and heading (t)
    xr,yr,tr = 1,1,0

    # Initial drone location (x, y, z), orientation (psi, theta), and velocity (v)
    x,y,z,psi,theta,v = 1,1,1.5,0,0,0

    # Obstacles                                                                       x     y    z
                # Stalagmites (on ground)
    obstacles = [[fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([1.0, 5.0, 1.0]))],

                [fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([14.0, 5.0, 1.0]))],
                
                [fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([5.0, 12.0, 1.0]))],
                
                [fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([10.0, 12.0, 1.0]))],
                
                [fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([5.0, 12.0, 9.0]))],
                
                [fcl.Box(np.array([2, 2, 2])), fcl.Transform3f(np.eye(3), np.array([10.0, 12.0, 9.0]))],

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

    mode = 1

    start0 = [x,y,z,psi,theta,v,xr,yr,tr, mode]

    # goal1 = [7.5,8.5,4.5,np.random.uniform(-np.pi/2,np.pi/2),0,0,7.5,8.5,0]

    goal2 = [14.0,10.0,7.0,np.random.uniform(-np.pi/2,np.pi/2),0,0,1.0,10.0,0]

    goal3 = [7.5,14.0,4.5,np.random.uniform(-np.pi/2,np.pi/2),0,0,7.5,14.0,0]

    # goals = [goal1, goal2, goal3]


    # goal = [10,9,4,np.random.uniform(-np.pi/2,np.pi/2),0,0,10,9,0]

    # number of iterations, n
    n = 5000
    # n = 7500
    
    # goal bias probability
    # p_goal = 0.05
    p_goal = 0.10

    # random sample state probability
    Q = 1-p_goal

    #-----------------------------------------------------------------------------#
    # Run the RRT planner on these inputs
    #-----------------------------------------------------------------------------#

    goala = [7.5,3,1.5,np.random.uniform(-np.pi/2,np.pi/2),0,0,7.5,3,0,0] #one for flight 0 for land, mode is at index 6
    goalb = [7.5,6.5,1,np.random.uniform(-np.pi/2,np.pi/2),0,0,7.5,6.5,0,0]
    goalab = [goala,goalb]

    solution_found0 = False

    tries0 = 0

    print("Going to goal 1")

    while not solution_found0:

        solution_found0, path0, path_length0, tree_size0,kino_path0 = create_rrt_landing(W_bounds, start0, goalab, n, Q, False, False, True, obstacles)

        tries0 += 1

        if not solution_found0:

            print(f"Attempt {tries0} did not converge on a solution- trying again...")

    print(f'Attempt {tries0} successful! Size of tree was {tree_size0}')

    #kinopath is the kinodynamic path
    x0,y0,z0,p0,ta0,v0 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    xr0,yr0,tr0 = np.array([]),np.array([]),np.array([])

    # Extract each state from the kinodynamic solution path
    for i in range(len(kino_path0)):

        # DRONE
        x0 = np.append(x0,kino_path0[i][0])
        y0 = np.append(y0,kino_path0[i][1])
        z0 = np.append(z0,kino_path0[i][2])
        p0 = np.append(p0,kino_path0[i][3])
        ta0 = np.append(ta0,kino_path0[i][4])
        v0 = np.append(v0,kino_path0[i][5])

        # ROVER
        # xr = np.append(xr,kino_path[i][0]) # or index 6
        # yr = np.append(yr,kino_path[i][1]) # or index 7
        # tr = np.append(tr,kino_path[i][2]) #or index at 8 

        xr0 = np.append(xr0,kino_path0[i][6]) # or index 6
        yr0 = np.append(yr0,kino_path0[i][7]) # or index 7
        tr0 = np.append(tr0,kino_path0[i][8]) #or index at 8 

    # Array of constant z values to plot the rover on the same 3D plot
    zr0 = 0.5 * np.ones(len(xr0)) #this will be for plotting on the same graph 

    #-----------------------------------------------------------------------------

    # start1 = kino_path0[-1][0:9,-1]

    # solution_found1 = False

    # tries1 = 0

    # print("Going to goal 1")

    # while not solution_found1:
    #                                                                                 # drone_parked, rover_parked, plot_path
    #     solution_found1, path1, path_length1, tree_size1,kino_path1 = create_rrt(W_bounds, start1, goal1, n, Q, False, False, True, obstacles)

    #     tries1 += 1

    #     print(tries1)

    # print(f'Size of tree was {0}', tree_size1)

    # #kinopath is the kinodynamic path
    # x1,y1,z1,p1,ta1,v1 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    # xr1,yr1,tr1 = np.array([]),np.array([]),np.array([])

    # # Extract each state from the kinodynamic solution path
    # for i in range(len(kino_path1)):

    #     # DRONE
    #     x1 = np.append(x1,kino_path1[i][0])
    #     y1 = np.append(y1,kino_path1[i][1])
    #     z1= np.append(z1,kino_path1[i][2])
    #     p1 = np.append(p1,kino_path1[i][3])
    #     ta1 = np.append(ta1,kino_path1[i][4])
    #     v1 = np.append(v1,kino_path1[i][5])

    #     # ROVER
    #     # xr = np.append(xr,kino_path[i][0]) # or index 6
    #     # yr = np.append(yr,kino_path[i][1]) # or index 7
    #     # tr = np.append(tr,kino_path[i][2]) #or index at 8 

    #     xr1 = np.append(xr1,kino_path1[i][6]) # or index 6
    #     yr1 = np.append(yr1,kino_path1[i][7]) # or index 7
    #     tr1 = np.append(tr1,kino_path1[i][8]) #or index at 8 

    # # Array of constant z values to plot the rover on the same 3D plot
    # zr1 = 0.5 * np.ones(len(xr1)) #this will be for plotting on the same graph 

    #-----------------------------------------------------------------------------

    print("Going to goal 2")

    start2 = kino_path0[-1][0:9,-1]
    # start2 = kino_path1[-1][:,-1]

    solution_found2 = False
    
    tries2 = 0

    while not solution_found2:
                                                                                    # drone_parked, rover_parked, plot_path
        solution_found2, path2, path_length2, tree_size2,kino_path2 = create_rrt(W_bounds, start2, goal2, n, Q, False, False, True, obstacles)

        tries2 += 1

        if not solution_found2:

            print(f"Attempt {tries2} did not converge on a solution- trying again...")

    print(f'Attempt {tries2} successful! Size of tree was {tree_size2}')

    #kinopath is the kinodynamic path
    x2,y2,z2,p2,ta2,v2 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    xr2,yr2,tr2 = np.array([]),np.array([]),np.array([])

    # Extract each state from the kinodynamic solution path
    for i in range(len(kino_path2)):

        # DRONE
        x2 = np.append(x2,kino_path2[i][0])
        y2 = np.append(y2,kino_path2[i][1])
        z2= np.append(z2,kino_path2[i][2])
        p2 = np.append(p2,kino_path2[i][3])
        ta2 = np.append(ta2,kino_path2[i][4])
        v2 = np.append(v2,kino_path2[i][5])

        # ROVER
        # xr = np.append(xr,kino_path[i][0]) # or index 6
        # yr = np.append(yr,kino_path[i][1]) # or index 7
        # tr = np.append(tr,kino_path[i][2]) #or index at 8 

        xr2 = np.append(xr2,kino_path2[i][6]) # or index 6
        yr2 = np.append(yr2,kino_path2[i][7]) # or index 7
        tr2 = np.append(tr2,kino_path2[i][8]) #or index at 8 

    # Array of constant z values to plot the rover on the same 3D plot
    zr2 = 0.5 * np.ones(len(xr2)) #this will be for plotting on the same graph 

    #-----------------------------------------------------------------------------

    print("Going to goal 3")

    start3 = kino_path2[-1][:,-1]

    solution_found3 = False
    
    tries3 = 0

    while not solution_found3:
                                                                                    # drone_parked, rover_parked, plot_path
        solution_found3, path3, path_length3, tree_size3,kino_path3 = create_rrt(W_bounds, start3, goal3, n, Q, False, False, True, obstacles)

        tries3 += 1

        if not solution_found3:

            print(f"Attempt {tries3} did not converge on a solution- trying again...")

    print(f'Attempt {tries3} successful! Size of tree was {tree_size3}')

    #kinopath is the kinodynamic path
    x3,y3,z3,p3,ta3,v3 = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    xr3,yr3,tr3 = np.array([]),np.array([]),np.array([])

    # Extract each state from the kinodynamic solution path
    for i in range(len(kino_path3)):

        # DRONE
        x3 = np.append(x3,kino_path3[i][0])
        y3 = np.append(y3,kino_path3[i][1])
        z3 = np.append(z3,kino_path3[i][2])
        p3 = np.append(p3,kino_path3[i][3])
        ta3 = np.append(ta3,kino_path3[i][4])
        v3 = np.append(v3,kino_path3[i][5])

        # ROVER
        # xr = np.append(xr,kino_path[i][0]) # or index 6
        # yr = np.append(yr,kino_path[i][1]) # or index 7
        # tr = np.append(tr,kino_path[i][2]) #or index at 8 

        xr3 = np.append(xr3,kino_path3[i][6]) # or index 6
        yr3 = np.append(yr3,kino_path3[i][7]) # or index 7
        tr3 = np.append(tr3,kino_path3[i][8]) #or index at 8 

    # Array of constant z values to plot the rover on the same 3D plot
    zr3 = 0.5 * np.ones(len(xr3)) #this will be for plotting on the same graph 

    #-----------------------------------------------------------------------------

    # Plot trajectory for task n1
    drone_radius = [.2]*len(x0)
    rover_side_length = [1] * len(x0)

    list_center = []

    for i in range(len(x0)):
        list_center.insert(0,(x0[i],y0[i],z0[i]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot(x0,y0,z0, color = 'r')
    ax.plot(xr0,yr0,zr0, color = 'r')

    # Plot spheres to represent the drone- only 30 spheres will be plotted along the path (too slow otherwise)
    if len(list_center) > 30:
    # if len(xr) > 30:
        # DRONE
        plt_sphere(ax, list_center[::math.floor(len(list_center) / 30.0)], drone_radius, 'r', 0.7) 

        # ROVER
        x_left = xr0[::math.floor(len(xr0) / 15.0)] - 0.5
        x_right = xr0[::math.floor(len(xr0) / 15.0)] + 0.5

        y_left = yr0[::math.floor(len(yr0) / 15.0)] - 0.5
        y_right = yr0[::math.floor(len(yr0) / 15.0)] + 0.5

        z_left = zr0[::math.floor(len(zr0) / 15.0)] - 0.5
        z_right = zr0[::math.floor(len(zr0) / 15.0)] + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'r', 0.5)

    elif len(list_center) > 0:
        # DRONE
        plt_sphere(ax, list_center, drone_radius, 'r', 0.7) 

        # ROVER
        x_left = xr0 - 0.5
        x_right = xr0 + 0.5

        y_left = yr0 - 0.5
        y_right = yr0 + 0.5

        z_left = zr0 - 0.5
        z_right = zr0 + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'r', 0.5)


    #-----------------------------------------------------------------------------

    # # Plot trajectory for task 1
    # drone_radius = [.2]*len(x1)
    # rover_side_length = [1] * len(x1)

    # list_center = []

    # for i in range(len(x1)):
    #     list_center.insert(0,(x1[i],y1[i],z1[i]))

    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')

    # ax.plot(x1,y1,z1, color = 'k')
    # ax.plot(xr1,yr1,zr1, color = 'k')

    # # Plot spheres to represent the drone- only 30 spheres will be plotted along the path (too slow otherwise)
    # if len(list_center) > 30:
    # # if len(xr) > 30:
    #     # DRONE
    #     plt_sphere(ax, list_center[::math.floor(len(list_center) / 30.0)], drone_radius, 'k', 0.7) 

    #     # ROVER
    #     x_left = xr1[::math.floor(len(xr1) / 15.0)] - 0.5
    #     x_right = xr1[::math.floor(len(xr1) / 15.0)] + 0.5

    #     y_left = yr1[::math.floor(len(yr1) / 15.0)] - 0.5
    #     y_right = yr1[::math.floor(len(yr1) / 15.0)] + 0.5

    #     z_left = zr1[::math.floor(len(zr1) / 15.0)] - 0.5
    #     z_right = zr1[::math.floor(len(zr1) / 15.0)] + 0.5

    #     plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
    #                                np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
    #                                np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'k', 0.5)

    # elif len(list_center) > 0:
    #     # DRONE
    #     plt_sphere(ax, list_center, drone_radius, 'k', 0.7) 

    #     # ROVER
    #     x_left = xr1 - 0.5
    #     x_right = xr1 + 0.5

    #     y_left = yr1 - 0.5
    #     y_right = yr1 + 0.5

    #     z_left = zr1 - 0.5
    #     z_right = zr1 + 0.5

    #     plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
    #                                np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
    #                                np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'k', 0.5)

    #-----------------------------------------------------------------------------

    # Plot trajectory for task 2
    drone_radius = [.2]*len(x2)
    rover_side_length = [1] * len(x2)

    list_center = []

    for i in range(len(x2)):
        list_center.insert(0,(x2[i],y2[i],z2[i]))

    ax.plot(x2,y2,z2, color = 'm')
    ax.plot(xr2,yr2,zr2, color = 'm')

    # Plot spheres to represent the drone- only 30 spheres will be plotted along the path (too slow otherwise)
    if len(list_center) > 30:
    # if len(xr) > 30:
        # DRONE
        plt_sphere(ax, list_center[::math.floor(len(list_center) / 30.0)], drone_radius, 'm', 0.7) 

        # ROVER
        x_left = xr2[::math.floor(len(xr2) / 15.0)] - 0.5
        x_right = xr2[::math.floor(len(xr2) / 15.0)] + 0.5

        y_left = yr2[::math.floor(len(yr2) / 15.0)] - 0.5
        y_right = yr2[::math.floor(len(yr2) / 15.0)] + 0.5

        z_left = zr2[::math.floor(len(zr2) / 15.0)] - 0.5
        z_right = zr2[::math.floor(len(zr2) / 15.0)] + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'm', 0.5)

    elif len(list_center) > 0:
        # DRONE
        plt_sphere(ax, list_center, drone_radius, 'm', 0.7) 

        # ROVER
        x_left = xr2 - 0.5
        x_right = xr2 + 0.5

        y_left = yr2 - 0.5
        y_right = yr2 + 0.5

        z_left = zr2 - 0.5
        z_right = zr2 + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'm', 0.5)

    #-----------------------------------------------------------------------------

    # Plot trajectory for task 3
    drone_radius = [.2]*len(x3)
    rover_side_length = [1] * len(x3)

    list_center = []

    for i in range(len(x3)):
        list_center.insert(0,(x3[i],y3[i],z3[i]))

    ax.plot(x3,y3,z3, color = 'c')
    ax.plot(xr3,yr3,zr3, color = 'c')

    # Plot spheres to represent the drone- only 30 spheres will be plotted along the path (too slow otherwise)
    if len(list_center) > 30:
    # if len(xr) > 30:
        # DRONE
        plt_sphere(ax, list_center[::math.floor(len(list_center) / 30.0)], drone_radius, 'c', 0.7) 

        # ROVER
        x_left = xr3[::math.floor(len(xr3) / 15.0)] - 0.5
        x_right = xr3[::math.floor(len(xr3) / 15.0)] + 0.5

        y_left = yr3[::math.floor(len(yr3) / 15.0)] - 0.5
        y_right = yr3[::math.floor(len(yr3) / 15.0)] + 0.5

        z_left = zr3[::math.floor(len(zr3) / 15.0)] - 0.5
        z_right = zr3[::math.floor(len(zr3) / 15.0)] + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'c', 0.5)

    elif len(list_center) > 0:
        # DRONE
        plt_sphere(ax, list_center, drone_radius, 'c', 0.7) 

        # ROVER
        x_left = xr3 - 0.5
        x_right = xr3 + 0.5

        y_left = yr3 - 0.5
        y_right = yr3 + 0.5

        z_left = zr3 - 0.5
        z_right = zr3 + 0.5

        plot_rectangular_prism(ax, np.concatenate((x_left.reshape(len(x_left), 1), x_right.reshape(len(x_right), 1)), axis = 1), \
                                   np.concatenate((y_left.reshape(len(y_left), 1), y_right.reshape(len(y_right), 1)), axis = 1), \
                                   np.concatenate((z_left.reshape(len(z_left), 1), z_right.reshape(len(z_right), 1)), axis = 1), 'c', 0.5)
    #-----------------------------------------------------------------------------

    # Plot start and goal points

    # DRONE
    ax.scatter(start0[0], start0[1], start0[2], color = 'b')
    ax.scatter(goalb[0], goalb[1], goalb[2], color = 'g')

    ax.scatter(start2[0], start2[1], start2[2], color = 'b')
    ax.scatter(goal2[0], goal2[1], goal2[2], color = 'g')

    ax.scatter(start3[0], start3[1], start3[2], color = 'b')
    ax.scatter(goal3[0], goal3[1], goal3[2], color = 'g')

    # ROVER
    ax.scatter(start0[6], start0[7], 0.5, color = 'b')
    ax.scatter(goalb[6], goalb[7], 0.5, color = 'g')

    ax.scatter(start2[6], start2[7], 0.5, color = 'b')
    ax.scatter(goal2[6], goal2[7], 0.5, color = 'g')

    ax.scatter(start3[6], start3[7], 0.5, color = 'b')
    ax.scatter(goal3[6], goal3[7], 0.5, color = 'g')

    # Plot goal region

    plot_rectangular_prism(ax, np.array([[0.0, 15.0]]), np.array([[2.0, 4.0]]), np.array([[0.0, 0.0]]), 'g', 0.2)

    plot_rectangular_prism(ax, np.array([[0.0, 15.0]]), np.array([[6.5, 8.0]]), np.array([[0.0, 0.0]]), 'g', 0.2)

    # DRONE

    # plot_rectangular_prism(ax, np.array([[6.5, 8.5]]), np.array([[7.5, 9.5]]), np.array([[3.5, 5.5]]), 'g', 0.2)
    
    plot_rectangular_prism(ax, np.array([[13.0, 15.0]]), np.array([[9.0, 11.0]]), np.array([[6.0, 8.0]]), 'g', 0.2)
    
    plot_rectangular_prism(ax, np.array([[6.5, 8.5]]), np.array([[13.0, 15.0]]), np.array([[3.5, 5.5]]), 'g', 0.2)

    # ROVER
    # plot_rectangular_prism(ax, np.array([[6.5, 8.5]]), np.array([[7.5, 9.5]]), np.array([[0.0, 2.0]]), 'g', 0.2)
    
    plot_rectangular_prism(ax, np.array([[0.0, 2.0]]), np.array([[9.0, 11.0]]), np.array([[0.0, 2.0]]), 'g', 0.2)

    plot_rectangular_prism(ax, np.array([[6.5, 8.5]]), np.array([[13.0, 15.0]]), np.array([[0.0, 2.0]]), 'g', 0.2)
    

    # Plot obstacles    
    plot_rectangular_prism(ax, np.array([[0.0, 2.0]]), np.array([[4.0, 6.0]]), np.array([[0.0, 2.0]]), 'r', 0.2)
    
    plot_rectangular_prism(ax, np.array([[13.0, 15.0]]), np.array([[4.0, 6.0]]), np.array([[0.0, 2.0]]), 'r', 0.2)
    
    plot_rectangular_prism(ax, np.array([[4.0, 6.0]]), np.array([[11.0, 13.0]]), np.array([[0.0, 2.0]]), 'r', 0.2)
    
    plot_rectangular_prism(ax, np.array([[9.0, 11.0]]), np.array([[11.0, 13.0]]), np.array([[0.0, 2.0]]), 'r', 0.2)
    
    plot_rectangular_prism(ax, np.array([[4.0, 6.0]]), np.array([[11.0, 13.0]]), np.array([[8.0, 10.0]]), 'r', 0.2)
    
    plot_rectangular_prism(ax, np.array([[9.0, 11.0]]), np.array([[11.0, 13.0]]), np.array([[8.0, 10.0]]), 'r', 0.2)
    
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
