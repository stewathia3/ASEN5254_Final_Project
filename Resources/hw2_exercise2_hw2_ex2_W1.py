###############################################################################
# Filename: hw7_exercise2.py
# Author: Sanjana Tewathia
#
# ASEN 5254, Fall 2022
#
# References/Acknowledgements:
#   - Line Segment Intersection Code based on:
#       https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
#       https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
# 
###############################################################################

#-----------------------------------------------------------------------------#
# Import Statements
#-----------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import time
from queue import PriorityQueue

#-----------------------------------------------------------------------------#
# Classes
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Class:    Node
# Purpose:  Node objects are Nodes that represent the vertices in the tree.
#
# Attributes/Properties:
#   - Name:             int- vertex number/name (Ex. 'v0' or 0) (goal = -1)
#   - Parent:           Node- the parent node or None
#   - DistFromParent:   double- distance of current vertex from goal
#
# Functions:
#   - __init__: the constructor
#-----------------------------------------------------------------------------#
class Node:
    def __init__(self, name, point, parent, DistFromParent):
        self.name = name
        self.point = point
        self.parent = parent
        self.DistFromParent = DistFromParent

#-----------------------------------------------------------------------------#
# Class:    Edge
# Purpose:  Edge objects that connect the string vertices in graphs.
#
# Attributes/Properties:
#   - v1:           Vertex- the first vertex (start)
#   - v2:           Vertex- the second vertex (next)
#   - weight:       double- the dist from v1 to v2
#
# Functions:
#   - __init__: the constructor
#-----------------------------------------------------------------------------#
class Edge:
    def __init__(self, v1, v2, weight):
        self.v1 = v1
        self.v2 = v2
        self.weight = weight

#-----------------------------------------------------------------------------#
# Function: check_on_line_segment
# Purpose:  Checks if of the three inputted points, the middle one is on the 
#           line segment that connects the first and last points.
#
# NOTE: Based on Geeks For Geeks Example Linked Above!
#
# Inputs:
#   - P1:   Point 1- an (x, y) array containing the first vertex
#   - P2:   Point 2- an (x, y) array containing the second vertex
#   - P3:   Point 3- an (x, y) array containing the third vertex
#
# Outputs:
#   - Boolean:  True if the middle point is on the line segment
#-----------------------------------------------------------------------------#
def check_on_line_segment(P1, P2, P3):

    # Check if the coordinates of Point 2 are between P1 and P3
    if (    (P2[0] <= max(P1[0], P3[0])) and    # P2's x is smaller than larger x coord of line segment
            (P2[0] >= min(P1[0], P3[0])) and    # P2's x is larger than smaller x coord of line segment

            (P2[1] <= max(P1[1], P3[1])) and    # P2's y is smaller than larger y coord of line segment
            (P2[1] >= min(P1[1], P3[1])) ):     # P2's y is larger than smaller y coord of line segment

        return True

    else:

        return False

#-----------------------------------------------------------------------------#
# Function: point_orientation
# Purpose:  Finds the orientation of three inputted points
#
# Inputs:
#   - P1:   Point 1- an (x, y) array containing the first vertex
#   - P2:   Point 2- an (x, y) array containing the second vertex
#   - P3:   Point 3- an (x, y) array containing the third vertex
#
# NOTE: Based on Geeks For Geeks Example Linked Above!
#
# Outputs:
#   - Integer:  2 if the points are CCW relative to each other
#               1 if the points are CW relative to each other
#               0 if the points are collinear
#-----------------------------------------------------------------------------#
def point_orientation(P1, P2, P3):

    # Compare two slopes      
    line_slope_comparison = ((P2[1] - P1[1]) * (P3[0] - P2[0])) - ((P2[0] - P1[0]) * (P3[1] - P2[1]))

    # If the points/vertices are counter-clockwise
    if (line_slope_comparison < 0):

        return 2

    # Else if the points/vertices are clockwise
    elif (line_slope_comparison > 0):
          
        return 1

    # Else: the points/vertices are collinear! Intersection could be happening here
    else:
          
        return 0

#-----------------------------------------------------------------------------#
# Function: check_for_intersection
# Purpose:  Finds out if the two line segments (P1, P2) and (V1, V2) intersect
#
# NOTE: Based on Geeks For Geeks Example Linked Above!
#
# Inputs:
#   - P1:   (x, y) array containing the first link vertex
#   - P2:   (x, y) array containing the second link vertex
#   - V1:   (x, y) array containing the first obstacle vertex
#   - V2:   P(x, y) array containing the second obstacle vertex
#
# Outputs:
#   - Boolean:  True if the line segments instersect
#               False if they do not intersect
#-----------------------------------------------------------------------------#
def check_for_intersection(P1, P2, V1, V2):
      
    # Find orientations for the following groups of interest:

    orientation1 = point_orientation(P1, P2, V1)    # If Obstacle Vertex 1 lies on link
    orientation2 = point_orientation(P1, P2, V2)    # If Obstacle Vertex 2 lies on link
    orientation3 = point_orientation(V1, V2, P1)    # If Link Vertex 1 lies on obstacle line
    orientation4 = point_orientation(V1, V2, P2)    # If Link Vertex 2 lies on obstacle line
  
    # If the obstacle or link lines vertices don't have the same orientation
    if ((orientation1 != orientation2) and (orientation3 != orientation4)):
        
        return True
    
    # Else if link is collinear with obstacle line: obstacle Vertex 1 between link line
    elif ((orientation1 == 0) and check_on_line_segment(P1, V1, P2)):

        return True
  
    # Else if link is collinear with obstacle line: obstacle Vertex 2 between link line
    elif ((orientation2 == 0) and check_on_line_segment(P1, V2, P2)):

        return True
  
    # Else if obstacle is collinear with link: link Vertex 1 between obstacle line
    elif ((orientation3 == 0) and check_on_line_segment(V1, P1, V2)):

        return True
  
    # Else if obstacle is collinear with link: link Vertex 2 between obstacle line
    elif ((orientation4 == 0) and check_on_line_segment(V1, P2, V2)):

        return True
  
    # Else, no intersection!
    else:
        
        return False

#-----------------------------------------------------------------------------#
# Function: check_for_obstacle_intersection
# Purpose:  Determines whether line segment from current step to next step will
#           intersect an obstacle line.
#
# Inputs:
#   - q:            np.array- (x, y) coord. of current step
#   - q_goal:       np.array- (x, y) coord. of goal
#   - obstacles:    list of np arrays- each np array contains vertices
#                   of each obstacle
#
# Outputs:
#   - boolean:  True if the line segments instersect
#               False if they do not intersect
#-----------------------------------------------------------------------------#
def check_for_obstacle_intersection(q, pot_step, obstacles):

    intersection = False

    # For obstacle i
    for obs_i in range(0, len(obstacles)):

        # Extract vertices
        vertices = obstacles[obs_i]

        n_vertices, n_dim = np.shape(vertices)

        # Iterate through vertices and make sure no intersections

        for vert_i in range(0, n_vertices):

            if check_for_intersection(vertices[vert_i], vertices[(vert_i + 1) % n_vertices], q, pot_step):

                intersection = True
                break

        if intersection:
            break

    return intersection

def create_rrt(start, goal, obstacles, n, r, epsilon, p_goal, x_lim, y_lim, plot_path):

    # Computation Time
    start_time = time.time()

    # STEP 1: Initialize tree with root/start node
    tree = [Node(0, start, None, 0.0)]

    valid_x = []
    valid_y = []
    valid_edges = []

    curr_node = tree[0]

    # STEP 2: Loop until n samples created or goal reached
    while (len(tree) < n) and (math.dist(curr_node.point, goal) > epsilon):

        # print(len(tree))
        # print(math.dist(curr_node.point, goal))

        # Goal Bias Check
        p_bias = np.random.uniform(0.0, 1.0)

        if p_bias >= (1 - p_goal):

            # q_rand is q_goal TODO:
            q_rand = goal

        else:
            x_point = np.random.uniform(x_lim[0], x_lim[1])
            y_point = np.random.uniform(y_lim[0], y_lim[1])

            q_rand = np.array([x_point, y_point])

        # Find node from tree that is closest to q_rand

        q_near = None # TODO:

        min_dist = math.inf

        for node in tree:

            curr_dist = math.dist(node.point, q_rand)
            
            if curr_dist < min_dist:
                min_dist = curr_dist
                q_near = node

        # Create q_new that is r away from q_near in direction of q_rand

        # print(q_near.point)
        # print(q_rand)
        # print(np.arange(start = q_near.point, stop = q_rand, step = 0.5))

        # q_new = np.arange(q_near.point, q_rand, 0.5)

        q_new = q_near.point + (r * ((q_rand - q_near.point) / (math.dist(q_rand, q_near.point))))

        # Check if q_new collides with obstacles
        if not check_for_obstacle_intersection(q_new, q_near.point, obstacles):
            # Add new Node
            curr_node = Node(len(tree), q_new, q_near, math.dist(q_new, q_near.point))
            tree.append(curr_node)

            valid_x.append(q_new[0])
            valid_y.append(q_new[1])

            # valid_x.append(q_rand[0])
            # valid_y.append(q_rand[1])
            valid_edges.append(Edge(q_new, q_near.point, curr_node.DistFromParent))

    end_time = (time.time() - start_time)

    # STEP 3: Create path from goal to start going up the tree
    if math.dist(curr_node.point, goal) <= epsilon:

        path = []
        path_length = 0.0

        curr_node = tree[-1]

        while curr_node != tree[0]:

            path.insert(0, curr_node)
            path_length += curr_node.DistFromParent
            curr_node = curr_node.parent

        path.insert(0, curr_node)

        if plot_path:

            fig = plt.figure()
            ax = fig.add_subplot()

            for obs_i in range(0, len(obstacles)):
                ax.add_patch(plt.Polygon(obstacles[obs_i], color = 'gray', alpha = 0.5))


            ax.scatter(valid_x, valid_y, color = 'k')

            for edge in valid_edges:

                ax.plot([edge.v1[0], edge.v2[0]], [edge.v1[1], edge.v2[1]], color = 'k', linewidth = 0.8)


            # Plot the path
            for i in range(0, len(path) - 1):

                path_length += math.dist(path[i].point, path[i + 1].point)

                ax.plot([path[i].point[0], path[i + 1].point[0]], [path[i].point[1], path[i + 1].point[1]], color = 'r', linewidth = '2')

            ax.scatter(start[0], start[1], color = 'b')
            ax.scatter(goal[0], goal[1], color = 'g')

            ax.add_patch(Circle(goal, epsilon, color = 'g', alpha = 0.5))

            ax.set_aspect('equal', 'box')

            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])

            ax.text(x_lim[0] + 0.25, y_lim[1] - 0.25, "Path Length: " + str(path_length))
            ax.text(x_lim[0] + 0.25, y_lim[1] - 0.75, "Tree Size: " + str(len(tree)))

            plt.show()

        return True, path, path_length, len(tree), end_time

    else:

        if plot_path:

            fig = plt.figure()
            ax = fig.add_subplot()

            for obs_i in range(0, len(obstacles)):
                ax.add_patch(plt.Polygon(obstacles[obs_i], color = 'gray', alpha = 0.5))


            ax.scatter(valid_x, valid_y, color = 'k')

            for edge in valid_edges:

                ax.plot([edge.v1[0], edge.v2[0]], [edge.v1[1], edge.v2[1]], color = 'k', linewidth = 0.8)

            ax.scatter(start[0], start[1], color = 'b')
            ax.scatter(goal[0], goal[1], color = 'g')

            ax.add_patch(Circle(goal, epsilon, color = 'g', alpha = 0.5))

            ax.set_aspect('equal', 'box')

            ax.text(x_lim[0], y_lim[1], "Could not reach goal!")

            plt.show()

        # print("Goal could not be reached with this RRT")

        return False, [], 0.0, len(tree)


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Exercise 2 (a): Planning problem of HW2 Exercise 2 (W1)
    #-----------------------------------------------------------------------------#

    start = np.array([0.0,0.0])
    goal = np.array([10.0,10.0])

    # obstacle_tolerance = 0.2

    WO1 = [np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.array([2.0, 5.0]), np.array([1.0, 5.0])]

    WO2 = [np.array([3.0, 3.0]), np.array([4.0, 3.0]), np.array([4.0, 12.0]), np.array([3.0, 12.0])]
    WO3 = [np.array([3.0, 12.0]), np.array([12.0, 12.0]), np.array([12.0, 13.0]), np.array([3.0, 13.0])]
    WO4 = [np.array([12.0, 5.0]), np.array([13.0, 5.0]), np.array([13.0, 13.0]), np.array([12.0, 13.0])]
    WO5 = [np.array([6.0, 5.0]), np.array([12.0, 5.0]), np.array([12.0, 6.0]), np.array([6.0, 6.0])]

    obstacles = [WO1, WO2, WO3, WO4, WO5]

    c_space_x_lim = [-1, 14]
    c_space_y_lim = [-1, 14]

    n = 5000
    r = 0.5
    epsilon = 0.25
    p_goal = 0.05

    #-----------------------------------------------------------------------------#
    # Run this section to generate 1 plot 
    # -> basically, 1 plot of the workspace, RRT, and path
    # -> This is the DEFAULT
    #-----------------------------------------------------------------------------#

    solution_found, path, path_length, tree_size, end_time = create_rrt(start, goal, obstacles, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, True)
    
    #-----------------------------------------------------------------------------#
    # Run this section to generate 3 plots with 100 runs
    #-----------------------------------------------------------------------------#

    # valid_solutions = 0
    # invalid_solutions = 0
    # path_lengths = []
    # end_times = []

    # for i in range(0, 100):

    #     solution_found, path, path_length, tree_size, end_time = create_rrt(start, goal, obstacles, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, False)

    #     if solution_found:
    #         valid_solutions += 1
    #         path_lengths.append(path_length)
    #         end_times.append(end_time)

    #     else:
    #         invalid_solutions += 1


    # path_length_box_plot = plt.figure()

    # plt.boxplot(path_lengths)

    # plt.savefig('path_lengths_ex2_hw2_ex2_W1.png')

    # plt.close(path_length_box_plot)

    # comp_time_box_plot = plt.figure()

    # plt.boxplot(end_times)

    # plt.savefig('computation_time_ex2_hw2_ex2_W1.png')

    # plt.close(comp_time_box_plot)

    # valid_solns_bar_chart = plt.figure()

    # plt.bar(["Valid", "Invalid"], [valid_solutions, invalid_solutions])

    # plt.savefig('valid_solutions_ex2_hw2_ex2_W1.png')

    # plt.close(valid_solns_bar_chart)
