import pygame
import heapq
import math
import numpy as np

class Planner:
    def get_path(self, surface, world_height, world_width, start, goal):
        """
        A* path planning algorithm to find optimal path from start to goal
        
        Args:
            surface: The pygame surface containing the map
            world_height (int): Height of the world
            world_width (int): Width of the world
            start (list): Starting coordinates [x, y]
            goal (list): Goal coordinates [x, y]
        
        Returns:
            list: List of coordinates representing the path
        """
        # Define constants
        WHITE = (255, 255, 255)
        # Include diagonal movements for more natural paths
        DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        # Convert start and goal to integers
        start_int = (int(start[0]), int(start[1]))
        goal_int = (int(goal[0]), int(goal[1]))
        
        # Check if start or goal is valid (within bounds and not in obstacle)
        if (start_int[0] < 0 or start_int[0] >= world_width or 
            start_int[1] < 0 or start_int[1] >= world_height or
            goal_int[0] < 0 or goal_int[0] >= world_width or 
            goal_int[1] < 0 or goal_int[1] >= world_height):
            return []
        
        try:
            # Check if start and goal are in free space (white)
            if surface.get_at(start_int)[:3] != WHITE or surface.get_at(goal_int)[:3] != WHITE:
                return []
        except IndexError:
            return []
        
        # Initialize data structures for A*
        open_set = []  # Priority queue for frontier
        closed_set = set()  # Set of evaluated nodes
        came_from = {}  # Parent pointers for path reconstruction
        g_score = {start_int: 0}  # Cost from start to each node
        f_score = {start_int: self._heuristic(start_int, goal_int)}  # Estimated total cost
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[start_int], start_int))
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if self._is_close_enough(current, goal_int):
                return self._reconstruct_path(came_from, current, start_int)
            
            # Mark current node as evaluated
            closed_set.add(current)
            
            # Check all neighbors
            for dx, dy in DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if invalid position
                if (neighbor[0] < 0 or neighbor[0] >= world_width or 
                    neighbor[1] < 0 or neighbor[1] >= world_height):
                    continue
                
                # Skip if obstacle or already evaluated
                try:
                    if surface.get_at(neighbor)[:3] != WHITE or neighbor in closed_set:
                        continue
                except IndexError:
                    continue
                
                # Calculate costs
                move_cost = math.sqrt(dx**2 + dy**2)  # Euclidean distance to neighbor
                tentative_g_score = g_score.get(current, float('inf')) + move_cost
                
                # If this path is better, record it
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_int)
                    
                    # Update open set
                    in_open_set = False
                    for i, (_, node) in enumerate(open_set):
                        if node == neighbor:
                            in_open_set = True
                            open_set[i] = (f_score[neighbor], neighbor)
                            heapq.heapify(open_set)
                            break
                    
                    if not in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found, find closest possible approach
        if closed_set:
            closest_node = min(closed_set, key=lambda node: self._heuristic(node, goal_int))
            return self._reconstruct_path(came_from, closest_node, start_int)
        return []
    
    def get_rrt_path(self, surface, world_height, world_width, start, goal, max_iterations=5000):
        """
        RRT path planning algorithm for subtask 3-6, more efficient for exploration
        
        Args:
            surface: The pygame surface containing the map
            world_height (int): Height of the world
            world_width (int): Width of the world
            start (list): Starting coordinates [x, y]
            goal (list): Goal coordinates [x, y]
            max_iterations (int): Maximum iterations to try
            
        Returns:
            list: List of coordinates representing the path
        """
        WHITE = (255, 255, 255)
        
        # Convert start and goal to tuples
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        # Initialize RRT
        nodes = [start]  # List of nodes in the tree
        parent = {start: None}  # Parent of each node
        
        # Main RRT loop
        for _ in range(max_iterations):
            # Bias towards goal with 10% probability
            if np.random.random() < 0.1:
                random_point = goal
            else:
                random_point = (
                    np.random.randint(0, world_width),
                    np.random.randint(0, world_height)
                )
            
            # Find closest node in the tree
            closest_node = min(nodes, key=lambda n: self._distance(n, random_point))
            
            # Create new node in the direction of random point
            theta = math.atan2(random_point[1] - closest_node[1], 
                              random_point[0] - closest_node[0])
            
            step_size = 10  # Adjustable step size
            new_node = (
                int(closest_node[0] + step_size * math.cos(theta)),
                int(closest_node[1] + step_size * math.sin(theta))
            )
            
            # Check if new node is valid
            if (new_node[0] < 0 or new_node[0] >= world_width or 
                new_node[1] < 0 or new_node[1] >= world_height):
                continue
            
            # Check if path to new node is clear
            if not self._is_path_clear(surface, closest_node, new_node, WHITE):
                continue
            
            # Add new node to tree
            nodes.append(new_node)
            parent[new_node] = closest_node
            
            # Check if goal reached
            if self._distance(new_node, goal) < 20:
                # Construct path
                path = [new_node]
                current = new_node
                while parent[current] is not None:
                    current = parent[current]
                    path.append(current)
                
                # Return reversed path (start to goal)
                return list(reversed(path))
        
        # If max iterations reached without finding goal, return path to closest node
        closest_to_goal = min(nodes, key=lambda n: self._distance(n, goal))
        path = [closest_to_goal]
        current = closest_to_goal
        
        while parent[current] is not None:
            current = parent[current]
            path.append(current)
        
        return list(reversed(path))
    
    def _is_path_clear(self, surface, start, end, free_color):
        """
        Check if path between two points is clear (no obstacles)
        
        Args:
            surface: Map surface
            start: Start point (x, y)
            end: End point (x, y)
            free_color: Color representing free space
            
        Returns:
            bool: True if path is clear
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return True
        
        x_step = dx / steps
        y_step = dy / steps
        
        # Check points along the line
        for i in range(1, steps + 1):
            x = int(start[0] + i * x_step)
            y = int(start[1] + i * y_step)
            
            try:
                if surface.get_at((x, y))[:3] != free_color:
                    return False
            except IndexError:
                return False
        
        return True
    
    def _distance(self, a, b):
        """Calculate Euclidean distance between points"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _heuristic(self, a, b):
        """Euclidean distance heuristic for A*"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _is_close_enough(self, a, b, threshold=5):
        """Check if points are close enough for goal test"""
        return self._heuristic(a, b) <= threshold
    
    def _reconstruct_path(self, came_from, current, start):
        """
        Reconstruct path from start to goal
        
        Args:
            came_from: Dictionary mapping each node to its parent
            current: Current (goal) node
            start: Start node
            
        Returns:
            list: Path from start to goal
        """
        path = [current]
        while current in came_from and current != start:
            current = came_from[current]
            path.append(current)
        
        if path[-1] != start:
            path.append(start)
        
        return path[::-1]  # Reverse to get path from start to goal
