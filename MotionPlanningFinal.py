# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:51:32 2022

@author: riana
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math


## ODE fix later 

class Node:
    def __init__(self, name, point, parent, DistFromParent,trajectory):
        self.name = name
        self.point = point
        self.parent = parent
        self.DistFromParent = DistFromParent
        self.trajectory = trajectory
        
        
class Edge:
    def __init__(self, trajectory):
        self.trajectory = trajectory

def ODEFunc(t,state,u1,u2,u3): #ode function
    
    x,y,z,v,psi,theta = state
    omega,alpha,a = u1,u2,u3
    
    #input state: [x,y,z,v,psi,theta]
    xdot = v*np.cos(psi)*np.cos(theta)
    ydot = v*np.sin(psi)*np.cos(theta)
    zdot = v*np.sin(theta) 
    psidot = omega
    thetadot = alpha 
    vdot = a

    return [xdot,ydot,zdot,psidot,thetadot,vdot]


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
        q_rand = [x,y,z,p,ta,v]
        
        
        
    elif chance >= Q:
        q_rand = q_goal
        
    return q_rand



def GenerateTrajectory(state,new_state): #state is qnear and new_state is q_rand
    
    m = 3 #generate 3 different random controls that extend from state which is qnear
    t_span = (0.0,0.5) #seconds
    min_dist = math.inf
    
    for i in range(m):
        u1 = np.random.uniform(-np.pi/6, np.pi/6)
        u2 = np.random.uniform(-np.pi/6, np.pi/6)
        u3 = np.random.uniform(-1/2, 1/2)
        control = (u1,u2,u3)
        #.y gets the results, .t gets the time
        result_solve_ivp = solve_ivp(ODEFunc, t_span, state,args=control, method = 'RK45')
    
        traj = result_solve_ivp.y
        best_con = math.dist(new_state,traj[:,-1])
        if best_con < min_dist:
            min_dist = best_con
            trajectories = traj
            time = result_solve_ivp.t

    
    return trajectories,time

def TrajectoryValid(trajectories,time): #does the trajectory work?
    x,y,z,ta,v = trajectories[0,:],trajectories[1,:],trajectories[2,:],trajectories[4,:],trajectories[5,:]
    
    #trajectory out of given constraints
    for i in range(len(time)): 
        if (0<=x[i]<=11) and (0<=y[i]<=10) and (0<=z[i]<=10) and (-1<=v[i]<=1) and (-np.pi/3<=ta[i]<=np.pi/3):
            Valid = 1
            
        else:   
            Valid = 0
            break
    
    return Valid

def create_rrt(start, goal,n, Q, plot_path):


    # STEP 1: Initialize tree with root/start node
    tree = [Node(0, start, None, 0.0,[])]
    
    


    curr_node = tree[0]
    x,y,z,p,ta,v = start
    solution_found = True
    stopper= 1

    # STEP 2: Loop until n samples created or goal reached
    while (len(tree) < n) and not(9<=x<=11 and 8<=y<=10 and 3<=z<=5 and -1/20<=v<=1/20):

        # print(len(tree))
        # print(math.dist(curr_node.point, goal))
        
        new_state = generateNode(Q,goal)


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

        Valid = TrajectoryValid(trajectories,time)

        # Check if q_new collides with obstacles
        if Valid:
            # Add new Node
            x_new = trajectories[:,-1]
            #name, new node, parent node, distance from parent node
            curr_node = Node(len(tree), x_new, state, math.dist(x_new, state.point),trajectories)
           
            tree.append(curr_node) #appends an object
            x,y,z,p,ta,v = x_new
            
            
        
        if len(tree) == 50*stopper:
            stopper +=1
            


    if len(tree)>=n:
        solution_found = False

    # STEP 3: Create path from goal to start going up the tree
    if solution_found == True:

        path = []
        kino_path = []
        path_length = 0.0

        curr_node = tree[-1]

        while curr_node != tree[0]:

            path.insert(0, curr_node)
            kino_path.insert(0,curr_node.trajectory)
            path_length += curr_node.DistFromParent
            curr_node = curr_node.parent
            

        path.insert(0, curr_node)
        
    return True, path, path_length, len(tree),kino_path

    

#Bounds that are allowable 
x_goal = [9,10]
y_goal = [8,9]
z_goal = [4,5]
v_goal = [-1/20,1/20]


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Exercise 2 (a): Planning problem of HW2 Exercise 2 (W1)
    #-----------------------------------------------------------------------------#
    x,y,z,psi,theta,v = 1,1,1,0,0,0
    start = [x,y,z,psi,theta,v]
    
    #hypothetical goal state that will lead to goal region
    goal = [9.5,8.5,4.5,np.random.uniform(-np.pi/2,np.pi/2),0,0]


    n = 5000
    p_goal = 0.05
    Q = 1-p_goal

    #-----------------------------------------------------------------------------#
    # Run this section to generate 1 plot 
    # -> basically, 1 plot of the workspace, RRT, and path
    # -> This is the DEFAULT
    #-----------------------------------------------------------------------------#

    solution_found, path, path_length, tree_size,kino_path = create_rrt(start, goal, n, Q, True)
    

#kinopath is the kinodynamic path
x,y,z,p,ta,v = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

for i in range(len(kino_path)):
    x = np.append(x,kino_path[i][0])
    y = np.append(y,kino_path[i][1])
    z= np.append(z,kino_path[i][2])
    p = np.append(p,kino_path[i][3])
    ta = np.append(ta,kino_path[i][4])
    v = np.append(v,kino_path[i][5])
    
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(x,y,z)
ax.set_title('Quadrotor Trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')