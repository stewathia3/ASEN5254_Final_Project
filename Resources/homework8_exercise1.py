###############################################################################
# Filename: hw8_exercise1.py
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
# Methods/Functions
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Function: within_WO1
# Purpose:  Determines whether current step is within obstacle WO1's bounds.
#
# Inputs:
#   - x:       float- x coord. of current step
#   - y:       float- y coord. of current step
#
# Outputs:
#   - int:     0 if within bounds, 1 if on bounds, and 2 if clear
#-----------------------------------------------------------------------------#
def within_WO1(x, y, tolerance):

    if  (x > (4 - tolerance) and x < (6 + tolerance) and y >= (6 - tolerance)  and y <= (7 + tolerance)) or \
        (y > (6 - tolerance) and y < (7 + tolerance) and x >= (4 - tolerance) and x <= (6 + tolerance)) or \
        \
        (x > (4 - tolerance) and x < (5 + tolerance) and y >= (6 - tolerance)  and y <= (10 + tolerance)) or \
        (y > (6 - tolerance) and y < (10 + tolerance) and x >= (4 - tolerance) and x <= (5 + tolerance)) or \
        \
        (x > (4 - tolerance) and x < (6 + tolerance) and y >= (9 - tolerance)  and y <= (10 + tolerance)) or \
        (y > (9 - tolerance) and y < (10 + tolerance) and x >= (4 - tolerance) and x <= (6 + tolerance)):

        return 0

    elif (x == (4 - tolerance) and (y == (6 - tolerance) or y == (7 + tolerance))) or (x == (6 + tolerance) and (y == (6 - tolerance) or y == (7 + tolerance))) or \
        \
        (x == (4 - tolerance) and (y == (6 - tolerance) or y == (10 + tolerance))) or (x == (5 + tolerance) and (y == (6 - tolerance) or y == (10 + tolerance))) or \
        \
        (x == (4 - tolerance) and (y == (9 - tolerance) or y == (10 + tolerance))) or (x == (6 + tolerance) and (y == (9 - tolerance) or y == (10 + tolerance))):

        return 1

    else:
        return 2


#-----------------------------------------------------------------------------#
# Function: within_WO2
# Purpose:  Determines whether current step is within obstacle WO1's bounds.
#
# Inputs:
#   - x:       float- x coord. of current step
#   - y:       float- y coord. of current step
#
# Outputs:
#   - int:     0 if within bounds, 1 if on bounds, and 2 if clear
#-----------------------------------------------------------------------------#
def within_WO2(x, y, tolerance):

    if  (x > (10 - tolerance) and x < (12 + tolerance) and y >= (6 - tolerance)  and y <= (7 + tolerance)) or \
        (y > (6 - tolerance) and y < (7 + tolerance) and x >= (10 - tolerance) and x <= (12 + tolerance)) or \
        \
        (x > (11 - tolerance) and x < (12 + tolerance) and y >= (6 - tolerance)  and y <= (10 + tolerance)) or \
        (y > (6 - tolerance) and y < (10 + tolerance) and x >= (11 - tolerance) and x <= (12 + tolerance)) or \
        \
        (x > (10 - tolerance) and x < (12 + tolerance) and y >= (9 - tolerance)  and y <= (10 + tolerance)) or \
        (y > (9 - tolerance) and y < (10 + tolerance) and x >= (10 - tolerance) and x <= (12 + tolerance)):

        return 0

    elif (x == (10 - tolerance) and (y == (6 - tolerance) or y == (7 + tolerance))) or (x == (12 + tolerance) and (y == (6 - tolerance) or y == (7 + tolerance))) or \
        \
        (x == (11 - tolerance) and (y == (6 - tolerance) or y == (10 + tolerance))) or (x == (12 + tolerance) and (y == (6 - tolerance) or y == (10 + tolerance))) or \
        \
        (x == (10 - tolerance) and (y == (9 - tolerance) or y == (10 + tolerance))) or (x == (12 + tolerance) and (y == (9 - tolerance) or y == (10 + tolerance))):

        return 1

    else:
        return 2

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
def check_for_obstacle_intersection(q, pot_step, agent_radius, obstacles):

    intersection = False

    # For obstacle i
    for obs_i in range(0, len(obstacles)):

        # Extract vertices
        vertices = obstacles[obs_i]

        n_vertices, n_dim = np.shape(vertices)

        # Iterate through vertices and make sure no intersections

        for vert_i in range(0, n_vertices):

            # # if within_WO1(pot_step[0], pot_step[1], agent_radius) or within_WO2(pot_step[0], pot_step[1], agent_radius):
            # #     intersection = True

            # x = pot_step[0]
            # y = pot_step[1]

            # # print(f"x: {x}, y: {y}")

            # x1 = vertices[vert_i][0]
            # y1 = vertices[vert_i][1]

            # # print(f"x1: {x1}, y1: {y1}")

            # x2 = vertices[(vert_i + 1) % n_vertices][0]
            # y2 = vertices[(vert_i + 1) % n_vertices][1]

            # if ( abs(((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / math.dist(vertices[vert_i], vertices[(vert_i + 1) % n_vertices])) ) <= agent_radius:

            #     # print(f"x2: {x2}, y2: {y2}")

            #     # print(((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)))

            #     # print(math.dist(vertices[vert_i], vertices[(vert_i + 1) % n_vertices]))

            #     # print(((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / math.dist(vertices[vert_i], vertices[(vert_i + 1) % n_vertices]))

            #     intersection = True
            #     break

            if check_for_intersection(vertices[vert_i], vertices[(vert_i + 1) % n_vertices], q, pot_step): #  + np.array([agent_radius, agent_radius])

                intersection = True
                break

        if intersection:
            break

    return intersection

def all_robots_reached_goals(curr_positions, next_positions, epsilon, goal_vector):

    shape = np.shape(curr_positions)

    all_within_epsilon = True

    for i in range(0, int(max(shape)), 2):

        # print(curr_positions[0])

        if not goal_vector[ int(i / 2) ]:

            agent_i = np.array([curr_positions[i], curr_positions[i + 1]])
            goal_i = np.array([next_positions[i], next_positions[i + 1]])

            if np.linalg.norm(agent_i - goal_i) <= epsilon:

                goal_vector[int(i / 2)] = True

            else:

                all_within_epsilon = False
                # break

    if all(goal_vector):

        all_within_epsilon = True


    # index = 0

    # for i in range(0, int(max(shape) / 2)):

    #     # print(curr_positions[0])

    #     if not goal_vector[i]:

    #         agent_i = np.array([curr_positions[index], curr_positions[index + 1]])
    #         goal_i = np.array([next_positions[index], next_positions[index + 1]])

    #         if np.linalg.norm(agent_i - goal_i) <= epsilon:

    #             goal_vector[i] = True

    #         else:

    #             all_within_epsilon = False
    #             # break

    #     index += 1

    return all_within_epsilon, goal_vector

#-----------------------------------------------------------------------------#
# Function: create_rrt
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
def create_rrt(start, goal, obstacles, n, r, epsilon, p_goal, x_lim, y_lim, m, agent_radius, plot_path, plot_obstacles = []):

    # Computation Time
    start_time = time.time()

    # STEP 1: Initialize tree with root/start node
    tree = [Node(0, start, None, 0.0)]

    valid_x = []
    valid_y = []
    valid_edges = []

    curr_node = tree[0]

    total_iters = 0

    goal_vector = np.full(m, False)

    goal_reached, goal_vector = all_robots_reached_goals(curr_node.point, goal, epsilon, goal_vector)

    old_point = start.copy()

    # STEP 2: Loop until n samples created or goal reached
    while (total_iters < n) and not goal_reached: # (math.dist(curr_node.point, goal) > (epsilon * m)): 

        # print(f"Tree length: {len(tree)}")

        # print(total_iters)

        # Goal Bias Check
        p_bias = np.random.uniform(0.0, 1.0)

        if p_bias >= (1 - p_goal):

            # q_rand is q_goal TODO:
            q_rand = goal

        else:

            # NOTE: Assuming x_lim == y_lim (which only works for this workspace!)

            q_rand = np.random.uniform(x_lim[0], x_lim[1], m * 2).transpose() # Shape = m * 2 x 1

        # Find node from tree that is closest to q_rand

        q_near = None

        min_dist = math.inf

        for node in tree:

            curr_dist = np.linalg.norm(node.point - q_rand)
            
            if curr_dist < min_dist:
                min_dist = curr_dist
                q_near = node

        # Create q_new that is r away from q_near in direction of q_rand

        q_new = np.zeros(m * 2)

        q_near_point = q_near.point

        for i in range(0, (m * 2), 2): # range(0, (m * 2) - 1):

            # Check if this agent is "parked" or at the goal, in which case q_new is q_old

            if goal_vector[ int(i / 2)]:

                q_new[i] = old_point[i]
                q_new[i + 1] = old_point[i + 1]

            else:

                q_rand_i = np.array([q_rand[i], q_rand[i + 1]])
                q_near_i = np.array([q_near_point[i], q_near_point[i + 1]])

                # print(math.dist(q_rand_i, q_near_i))

                # print(q_rand_i - q_near_i)

                # print((q_rand_i - q_near_i) / (math.dist(q_rand_i, q_near_i)))

                q_new_i = q_near_i + (r * (q_rand_i - q_near_i) / (np.linalg.norm(q_rand_i - q_near_i))) # (math.dist(q_rand_i, q_near_i)))


                q_new[i] = q_new_i[0]
                q_new[i + 1] = q_new_i[1]

        # q_new = q_near.point + (r * ((q_rand - q_near.point) / (math.dist(q_rand, q_near.point))))

        agent_collision = False
        obstacle_collision = False

        dim = len(q_new)

        # TODO: Check if agents collide with each other: if distance between centroids of two robots <= 2 times radius, collision

        if m > 1:

            for i in range(0, m - 1):

                agent_i = np.array([q_new[(i * 2)], q_new[(i * 2) + 1]])

                # Check collisions with rest of agents 
                for j in range(i + 1, m):
                    agent_i_p_1 = np.array([q_new[j * 2], q_new[(j * 2) + 1]])
                    # agent_i_p_1 = np.array([q_new[(i + 2) % dim], q_new[(i + 3) % dim]])

                    if np.linalg.norm(agent_i - agent_i_p_1) <= (2 * agent_radius):
                        agent_collision = True
                        break

        for i in range(0, m * 2, 2):

            # agent_i = np.array([q_new[i], q_new[i + 1]])
            # agent_i_p_1 = np.array([q_new[(i + 2) % dim], q_new[(i + 3) % dim]])

            # if np.linalg.norm(agent_i - agent_i_p_1) <= (2 * agent_radius):
            #     agent_collision = True
            #     break

            # Only collision check if the robot is parked

            # if not goal_vector[ int(i / 2)]:

            agent_i = np.array([q_new[i], q_new[i + 1]])

            # check_for_obstacle_intersection(np.array([q_near_point[(i + 2) % dim], q_near_point[(i + 3) % dim]]), agent_i_p_1, agent_radius, obstacles)

            if check_for_obstacle_intersection(np.array([q_near_point[i], q_near_point[i + 1]]), agent_i, agent_radius, obstacles):

                obstacle_collision = True
                break

        # Check if q_new collides with obstacles
        if not obstacle_collision and not agent_collision:

            old_point = q_new.copy()
            # Add new Node
            curr_node = Node(len(tree), q_new, q_near, np.linalg.norm(q_new - q_near.point))
            tree.append(curr_node)

            for i in range(0, m * 2, 2):

                # if not goal_vector[int(i / 2)]:
                valid_x.append(q_new[i])
                # valid_x.append(q_new[2])
                valid_y.append(q_new[i + 1])
                    # valid_y.append(q_new[3])

            valid_edges.append(Edge(q_new, q_near.point, curr_node.DistFromParent))

        goal_reached, goal_vector = all_robots_reached_goals(curr_node.point, goal, epsilon, goal_vector)

        # if goal_reached:
        #     print(goal_reached)
        #     print(total_iters)
        
        total_iters += 1

    end_time = (time.time() - start_time)

    # STEP 3: Create path from goal to start going up the tree
    if goal_reached: # math.dist(curr_node.point, goal) <= (epsilon * m):

        path = []
        path_length = 0.0

        curr_node = tree[-1]

        while curr_node != tree[0]:

            path.insert(0, curr_node)
            path_length += curr_node.DistFromParent
            curr_node = curr_node.parent

        path.insert(0, curr_node)

        if plot_path:

            print("Wait for figure to open (this may take a while), and then close figure after to proceed")

            fig = plt.figure()
            ax = fig.add_subplot()

            for obs_i in range(0, len(plot_obstacles)):
                ax.add_patch(plt.Polygon(plot_obstacles[obs_i], color = 'gray', alpha = 0.5))


            ax.scatter(valid_x, valid_y, color = 'k')

            for i in range(0, len(valid_x)):

                ax.add_patch(Circle([valid_x[i], valid_y[i]], agent_radius, color = 'k', alpha = 0.1))

            # for edge in valid_edges:

            #     for i in range(0, m*2, 2):

            #         ax.plot([edge.v1[i], edge.v2[i]], [edge.v1[i + 1], edge.v2[i + 1]], color = 'k', linewidth = 0.8)

            # Plot the paths
            for i in range(0, len(path) - 1):

                path_length += math.dist(path[i].point, path[i + 1].point)

                for j in range(0, m*2, 2):

                    ax.plot([path[i].point[j], path[i + 1].point[j]], [path[i].point[j + 1], path[i + 1].point[j + 1]], color = 'r', linewidth = '2')

            for i in range(0, m*2, 2):

                ax.scatter(start[i], start[i + 1], color = 'b')

                ax.scatter(goal[i], goal[i + 1], color = 'g')

                ax.add_patch(Circle(goal[i:i+2], epsilon, color = 'g', alpha = 0.5))

            ax.set_aspect('equal', 'box')

            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])

            ax.text(x_lim[0] + 0.25, y_lim[1] - 0.25, "Path Length: " + str(path_length))
            ax.text(x_lim[0] + 0.25, y_lim[1] - 0.75, "Tree Size: " + str(len(tree)))

            plt.show()

        return True, path, path_length, len(tree), end_time

    else:

    #     if plot_path:

    #         fig = plt.figure()
    #         ax = fig.add_subplot()

    #         for obs_i in range(0, len(plot_obstacles)):
    #             ax.add_patch(plt.Polygon(plot_obstacles[obs_i], color = 'gray', alpha = 0.5))


    #         ax.scatter(valid_x, valid_y, color = 'k')

    #         for i in range(0, len(valid_x)):

    #             ax.add_patch(Circle([valid_x[i], valid_y[i]], agent_radius, color = 'k', alpha = 0.1))

    #         for edge in valid_edges:

    #             for i in range(0, m*2, 2):

    #                 ax.plot([edge.v1[i], edge.v2[i]], [edge.v1[i + 1], edge.v2[i + 1]], color = 'k', linewidth = 0.8)

    #             # ax.plot([edge.v1[2], edge.v2[2]], [edge.v1[3], edge.v2[3]], color = 'r', linewidth = 0.8)

    #         for i in range(0, m*2, 2):

    #             ax.scatter(start[i], start[i + 1], color = 'b')

    #             ax.scatter(goal[i], goal[i + 1], color = 'g')

    #             ax.add_patch(Circle(goal[i:i+2], epsilon, color = 'g', alpha = 0.5))

    #         # ax.add_patch(Circle(goal[2:4], epsilon, color = 'g', alpha = 0.5))

    #         ax.set_aspect('equal', 'box')

    #         ax.text(x_lim[0], y_lim[1], "Could not reach goal!")

    #         plt.show()

        # print("Goal could not be reached with this RRT")

        return False, [], 0.0, len(tree), end_time

if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Exercise 1 (a): Planning problem of HW5 Exercise 2 (a)
    #-----------------------------------------------------------------------------#

    WO1 = [np.array([4.0, 6.0]), np.array([6.0, 6.0]), np.array([6.0, 7.0]), np.array([4.0, 7.0])]
    WO2 = [np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([5.0, 10.0]), np.array([4.0, 10.0])]
    WO3 = [np.array([4.0, 9.0]), np.array([6.0, 9.0]), np.array([6.0, 10.0]), np.array([4.0, 10.0])]
    WO4 = [np.array([10.0, 6.0]), np.array([12.0, 6.0]), np.array([12.0, 7.0]), np.array([10.0, 7.0])]
    WO5 = [np.array([11.0, 6.0]), np.array([12.0, 6.0]), np.array([12.0, 10.0]), np.array([11.0, 10.0])]
    WO6 = [np.array([10.0, 9.0]), np.array([12.0, 9.0]), np.array([12.0, 10.0]), np.array([10.0, 10.0])]

    WO1_padded = [np.array([3.5, 5.5]), np.array([6.5, 5.5]), np.array([6.5, 7.5]), np.array([3.5, 7.5])]
    WO2_padded = [np.array([3.5, 5.5]), np.array([5.5, 5.5]), np.array([5.5, 10.5]), np.array([3.5, 10.5])]
    WO3_padded = [np.array([3.5, 8.5]), np.array([6.5, 8.5]), np.array([6.5, 10.5]), np.array([3.5, 10.5])]

    # TODO: PAD THESE!
    WO4_padded = [np.array([9.5, 5.5]), np.array([12.5, 5.5]), np.array([12.5, 7.5]), np.array([9.5, 7.5])]
    WO5_padded = [np.array([10.5, 5.5]), np.array([12.5, 5.5]), np.array([12.5, 10.5]), np.array([10.5, 10.5])]
    WO6_padded = [np.array([9.5, 8.5]), np.array([12.5, 8.5]), np.array([12.5, 10.5]), np.array([9.5, 10.5])]


    obstacles = [WO1, WO2, WO3, WO4, WO5, WO6]

    obstacles_padded = [WO1_padded, WO2_padded, WO3_padded, WO4_padded, WO5_padded, WO6_padded]

    agent_radius = 0.5

    c_space_x_lim = [0, 16]
    c_space_y_lim = [0, 16]

    n = 7500 # 7500
    r = 0.5
    epsilon = 0.25
    p_goal = 0.05

    x1_start = [2.0, 2.0]
    x1_goal = [14.0, 14.0]

    x2_start = [2.0, 14.0]
    x2_goal = [14.0, 2.0]

    x3_start = [8.0, 14.0]
    x3_goal = [8.0, 2.0]

    x4_start = [2.0, 8.0]
    x4_goal = [14.0, 8.0]

    x5_start = [11.0, 2.0]
    x5_goal = [5.0, 14.0]

    x6_start = [11.0, 14.0]
    x6_goal = [5.0, 2.0]

    #-----------------------------------------------------------------------------#
    # Default run- plots valid solutions
    #-----------------------------------------------------------------------------#

    print("Press enter to run RRT for m = 1")

    input()

    m = 1

    start_state = np.array(x1_start).transpose()
    goal_state = np.array(x1_goal).transpose()

    plot_output = True
    solution_found = False

    while not solution_found:

        solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, plot_output, obstacles)
        print(f"Solution Found: {solution_found}")
        
    print(f"Solution Found: {solution_found}")
    print(f"Tree Size: {tree_size}")
    print(f"Comp Time: {end_time}")

    #-----------------------------------------------------------------------------#

    print("Press enter to run RRT for m = 2")

    input()

    m = 2

    start_state = np.array(x1_start + x2_start).transpose()
    goal_state = np.array(x1_goal + x2_goal).transpose()

    plot_output = True
    solution_found = False

    while not solution_found:

        solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, plot_output, obstacles)
        print(f"Solution Found: {solution_found}")
        
    print(f"Solution Found: {solution_found}")
    print(f"Tree Size: {tree_size}")
    print(f"Comp Time: {end_time}")

    #-----------------------------------------------------------------------------#

    print("Press enter to run RRT for m = 3")

    input()

    m = 3

    start_state = np.array(x1_start + x2_start + x3_start).transpose()
    goal_state = np.array(x1_goal + x2_goal + x3_goal).transpose()

    plot_output = True
    solution_found = False

    while not solution_found:

        solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, plot_output, obstacles)
        
        if not solution_found:
            print(f"Solution Found: {solution_found}, trying again...")
        else:
            print(f"Solution Found: {solution_found}")
        
    print(f"Tree Size: {tree_size}")
    print(f"Comp Time: {end_time}")

    #-----------------------------------------------------------------------------#

    print("Press enter to run RRT for m = 4")

    input()

    m = 4

    start_state = np.array(x1_start + x2_start + x3_start + x4_start).transpose()
    goal_state = np.array(x1_goal + x2_goal + x3_goal + x4_goal).transpose()

    plot_output = True
    solution_found = False

    while not solution_found:

        solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, plot_output, obstacles)
        print(f"Solution Found: {solution_found}")
        
    print(f"Solution Found: {solution_found}")
    print(f"Tree Size: {tree_size}")
    print(f"Comp Time: {end_time}")

    #-----------------------------------------------------------------------------#

    print("Press enter to run RRT for m = 5")

    input()

    m = 5

    start_state = np.array(x1_start + x2_start + x3_start + x4_start + x5_start).transpose()
    goal_state = np.array(x1_goal + x2_goal + x3_goal + x4_goal + x5_goal).transpose()

    plot_output = True
    solution_found = False

    while not solution_found:

        solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, plot_output, obstacles)
        print(f"Solution Found: {solution_found}")
        
    print(f"Solution Found: {solution_found}")
    print(f"Tree Size: {tree_size}")
    print(f"Comp Time: {end_time}")

    #-----------------------------------------------------------------------------#

    print("Press enter to run RRT for m = 6")

    input()

    m = 6

    start_state = np.array(x1_start + x2_start + x3_start + x4_start + x5_start + x6_start).transpose()
    goal_state = np.array(x1_goal + x2_goal + x3_goal + x4_goal + x5_goal + x6_goal).transpose()

    plot_output = True
    solution_found = False

    while not solution_found:

        solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, plot_output, obstacles)
        print(f"Solution Found: {solution_found}")
        
    print(f"Solution Found: {solution_found}")
    print(f"Tree Size: {tree_size}")
    print(f"Comp Time: {end_time}")

    #-----------------------------------------------------------------------------#
    # 100 runs code
    #-----------------------------------------------------------------------------#

    # run_time_start = time.time()

    # valid_solutions = []
    # tree_sizes = []
    # end_times = []

    # for i in range(0, 100):

    #     print(i)

    #     # solution_found, path, path_length, tree_size, end_time = create_rrt(start, goal, obstacles, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, False)

    #     solution_found, path, path_length, tree_size, end_time = create_rrt(start_state, goal_state, obstacles_padded, n, r, epsilon, p_goal, c_space_x_lim, c_space_y_lim, m, agent_radius, False, obstacles)

    #     tree_sizes.append(tree_size)
    #     end_times.append(end_time)

    #     if solution_found:
    #         valid_solutions.append(1)
            
    #     else:
    #         valid_solutions.append(0)

    # # Save data
    # np.savetxt(f"valid_solns_m_{m}.csv", valid_solutions)
    # np.savetxt(f"tree_size_m_{m}.csv", tree_sizes) # np.array([0.0, tree_size]))
    # np.savetxt(f"comp_time_m_{m}.csv", end_times) # np.array([0.0, end_time]))

    # run_time_end = time.time() - run_time_start

    # print(f"Run complete! Took {run_time_end} seconds for 100 runs of m = {m}")

    # tree_sizes_box_plot = plt.figure()

    # plt.boxplot(tree_sizes)

    # plt.savefig(f"tree_sizes_ex1_m_{m}.png")

    # plt.close(tree_sizes_box_plot)

    # comp_time_box_plot = plt.figure()

    # plt.boxplot(end_times)

    # plt.savefig(f"computation_time_ex1_m_{m}.png")

    # plt.close(comp_time_box_plot)
