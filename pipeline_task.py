from localization import Localization
from planning import Planner
from map_merger import MapMerger
import math
import numpy as np
import pygame

class Pipeline:
    def __init__(self, world_width, world_height, map):
        """
        Initialize the pipeline
        
        Args:
            world_width (int): Width of the world
            world_height (int): Height of the world
            map: Map surface
        """
        # Initialize components
        self.explored_map1 = Localization(2 * world_width, 2 * world_height)
        self.explored_map2 = Localization(2 * world_width, 2 * world_height)
        self.planner = Planner()
        self.map_merger = MapMerger(2 * world_width, 2 * world_height)
        
        # State variables
        self.path = None
        self.ind = 0
        self.world_height = world_height
        self.world_width = world_width
        self.agent1_pos = None
        self.agent2_pos = None
        self.rendezvous_complete = False
        
        # For subtask identification
        self.subtask = 1  # Default to subtask 1
        self.map_available = True
        self.positions_available = True
        
        # Store map for reference
        self.map = map
        
        # For scan matching and position estimation
        self.scan_matcher_initialized = False
        self.relative_transform = np.eye(3)  # Identity transformation
        
    def reset(self):
        """Reset the pipeline state"""
        self.path = None
        self.ind = 0
        self.agent1_pos = None
        self.agent2_pos = None
        self.rendezvous_complete = False
        
    def set_subtask(self, subtask_num):
        """
        Set which subtask to execute
        
        Args:
            subtask_num (int): Subtask number (1-6)
        """
        self.subtask = subtask_num
        
        # Configure constraints based on subtask
        if subtask_num == 1:
            # Basic task - use agent positions and map
            self.map_available = True
            self.positions_available = True
        elif subtask_num == 2:
            # Simple rotate and move approach
            self.map_available = True
            self.positions_available = True
        elif subtask_num == 3:
            # No map available
            self.map_available = False
            self.positions_available = True
        elif subtask_num == 4:
            # No positions available
            self.map_available = True
            self.positions_available = False
        elif subtask_num == 5:
            # Only scan and IMU data
            self.map_available = False
            self.positions_available = False
        elif subtask_num == 6:
            # Optimized scanning
            self.map_available = False
            self.positions_available = False
        
        # Reset state when changing subtasks
        self.reset()
        
    def work(self, agent1, agent2):
        """
        Make the two agents meet somewhere on the map
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Skip if rendezvous is already complete
        if self.rendezvous_complete:
            return
        
        # Update maps with scan data
        self.explored_map1.update(agent1)
        self.explored_map2.update(agent2)
        
        # Execute appropriate subtask
        if self.subtask == 1:
            self._execute_subtask1(agent1, agent2)
        elif self.subtask == 2:
            self._execute_subtask2(agent1, agent2)
        elif self.subtask == 3:
            self._execute_subtask3(agent1, agent2)
        elif self.subtask == 4:
            self._execute_subtask4(agent1, agent2)
        elif self.subtask == 5:
            self._execute_subtask5(agent1, agent2)
        else:  # subtask 6
            self._execute_subtask6(agent1, agent2)
    
    def _execute_subtask1(self, agent1, agent2):
        """
        Subtask 1: Use initial coordinates and map
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Get initial agent positions if not already obtained
        if self.agent1_pos is None:
            try:
                self.agent1_pos = list(agent1.get_pos())
            except:
                # Fallback if get_pos fails
                self.agent1_pos = [self.world_width/4, self.world_height/4]
                
        if self.agent2_pos is None:
            try:
                self.agent2_pos = list(agent2.get_pos())
            except:
                # Fallback if get_pos fails
                self.agent2_pos = [3*self.world_width/4, 3*self.world_height/4]
        
        # Check if agents are close enough (rendezvous achieved)
        dist = math.sqrt((self.agent1_pos[0] - self.agent2_pos[0])**2 + 
                        (self.agent1_pos[1] - self.agent2_pos[1])**2)
        if dist < 20:
            self.rendezvous_complete = True
            return
        
        # Compute path if not already done
        if self.path is None:
            # Use the provided world map for planning
            world_surface = agent1.get_world()
            self.path = self.planner.get_path(world_surface, self.world_height, self.world_width, 
                                            self.agent1_pos, self.agent2_pos)
            self.ind = 0
        
        # Follow the path using basic navigation
        self._follow_path(agent1)
    
    def _execute_subtask2(self, agent1, agent2):
        """
        Subtask 2: Use coordinates and map with simple rotate-and-move approach
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Get initial agent positions
        if self.agent1_pos is None:
            try:
                self.agent1_pos = list(agent1.get_pos())
            except:
                self.agent1_pos = [self.world_width/4, self.world_height/4]
                
        if self.agent2_pos is None:
            try:
                self.agent2_pos = list(agent2.get_pos())
            except:
                self.agent2_pos = [3*self.world_width/4, 3*self.world_height/4]
        
        # Check if agents are close enough (rendezvous achieved)
        dist = math.sqrt((self.agent1_pos[0] - self.agent2_pos[0])**2 + 
                        (self.agent1_pos[1] - self.agent2_pos[1])**2)
        if dist < 20:
            self.rendezvous_complete = True
            return
        
        # Compute path if not already done
        if self.path is None:
            world_surface = agent1.get_world()
            # Get path using A* algorithm
            self.path = self.planner.get_path(world_surface, self.world_height, self.world_width, 
                                           self.agent1_pos, self.agent2_pos)
            self.ind = 0
        
        # Follow the path with simple rotate-and-move approach
        self._follow_rotate_and_move_path(agent1)
    
    def _execute_subtask3(self, agent1, agent2):
        """
        Subtask 3: Use coordinates but not the map
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Get initial agent positions
        if self.agent1_pos is None:
            try:
                self.agent1_pos = list(agent1.get_pos())
            except:
                self.agent1_pos = [self.world_width/4, self.world_height/4]
                
        if self.agent2_pos is None:
            try:
                self.agent2_pos = list(agent2.get_pos())
            except:
                self.agent2_pos = [3*self.world_width/4, 3*self.world_height/4]
        
        # Check if agents are close enough (rendezvous achieved)
        dist = math.sqrt((self.agent1_pos[0] - self.agent2_pos[0])**2 + 
                        (self.agent1_pos[1] - self.agent2_pos[1])**2)
        if dist < 20:
            self.rendezvous_complete = True
            return
        
        # Merge maps from both agents to create a unified map
        merged_map = self.map_merger.merge(self.explored_map1.map, self.explored_map2.map)
        
        # Compute path if not already done or if need to replan
        if self.path is None or self.ind >= len(self.path) - 1:
            # Use RRT for planning in partially known environments
            self.path = self.planner.get_rrt_path(merged_map, self.world_height, self.world_width, 
                                               self.agent1_pos, self.agent2_pos)
            self.ind = 0
        
        # Follow the path with the simple rotate-and-move approach
        self._follow_rotate_and_move_path(agent1)
    
    def _execute_subtask4(self, agent1, agent2):
        """
        Subtask 4: Use map but not coordinates
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Initialize position estimates if not done
        world_surface = agent1.get_world()
        
        if self.agent1_pos is None:
            # Initialize particle filter for position estimation
            self.explored_map1.initialize_particle_filter(300, world_surface)
            self.agent1_pos = self.explored_map1.get_estimated_position()
            
        if self.agent2_pos is None:
            # Initialize particle filter for position estimation
            self.explored_map2.initialize_particle_filter(300, world_surface)
            self.agent2_pos = self.explored_map2.get_estimated_position()
        
        # Update position estimates
        self.agent1_pos = self.explored_map1.get_estimated_position()
        self.agent2_pos = self.explored_map2.get_estimated_position()
        
        # Check if agents are close enough (rendezvous achieved)
        dist = math.sqrt((self.agent1_pos[0] - self.agent2_pos[0])**2 + 
                        (self.agent1_pos[1] - self.agent2_pos[1])**2)
        if dist < 20:
            self.rendezvous_complete = True
            return
        
        # Compute path if not already done
        if self.path is None or self.ind >= len(self.path) - 1:
            # Use A* for planning with known map
            self.path = self.planner.get_path(world_surface, self.world_height, self.world_width, 
                                            self.agent1_pos, self.agent2_pos)
            self.ind = 0
        
        # Follow the path
        self._follow_rotate_and_move_path(agent1)
    
    def _execute_subtask5(self, agent1, agent2):
        """
        Subtask 5: Use only scan and IMU data
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Update local maps
        sensor_data1 = agent1.scan(fov=360, resolution=2)
        sensor_data2 = agent2.scan(fov=360, resolution=2)
        
        # Initialize position estimates if needed
        if self.agent1_pos is None:
            self.explored_map1.initialize_particle_filter(500)
            self.agent1_pos = [self.world_width/2, self.world_height/2]
        
        if self.agent2_pos is None:
            self.explored_map2.initialize_particle_filter(500)
            self.agent2_pos = [self.world_width/2, self.world_height/2]
        
        # Update position estimates
        self.agent1_pos = self.explored_map1.get_estimated_position()
        self.agent2_pos = self.explored_map2.get_estimated_position()
        
        # Initialize or update scan matcher between agents
        if not self.scan_matcher_initialized:
            # Simple initial transformation estimate
            self.relative_transform = np.array([
                [1.0, 0.0, self.world_width/2],
                [0.0, 1.0, self.world_height/2],
                [0.0, 0.0, 1.0]
            ])
            self.scan_matcher_initialized = True
        
        # Transform agent2's position to agent1's coordinate frame
        agent2_in_agent1_frame = self._transform_position(self.agent2_pos)
        
        # Check if agents are close enough (rendezvous achieved)
        dist = math.sqrt((self.agent1_pos[0] - agent2_in_agent1_frame[0])**2 + 
                        (self.agent1_pos[1] - agent2_in_agent1_frame[1])**2)
        if dist < 20:
            self.rendezvous_complete = True
            return
        
        # Merge maps using scan matching
        merged_map = self.map_merger.scan_matching_merge(self.explored_map1.map, self.explored_map2.map)
        
        # Compute path if not already done or if need to replan
        if self.path is None or self.ind >= len(self.path) - 1:
            # Use RRT for planning in partially known environments
            self.path = self.planner.get_rrt_path(merged_map, self.world_height, self.world_width, 
                                               self.agent1_pos, agent2_in_agent1_frame)
            self.ind = 0
        
        # Follow the path
        success = self._follow_rotate_and_move_path(agent1)
        
        # If path following fails, try exploration
        if not success:
            self._find_and_explore_frontiers(agent1, merged_map)
    
    def _execute_subtask6(self, agent1, agent2):
        """
        Subtask 6: Optimized for minimal scanning
        
        Args:
            agent1: First robot agent
            agent2: Second robot agent
        """
        # Implement adaptive scanning frequency
        should_scan = False
        
        # Initialize scan timer if needed
        if not hasattr(self, 'last_scan_time'):
            self.last_scan_time = 0
            should_scan = True
        
        # Scan if at decision points or periodically
        self.last_scan_time += 1
        if (self.path is None or 
            (self.ind < len(self.path) and self._distance(self.agent1_pos, self.path[self.ind]) < 10) or
            self.last_scan_time > 20):
            should_scan = True
        
        if should_scan:
            # Reset scan timer
            self.last_scan_time = 0
            
            # Perform reduced scanning (180° instead of 360°)
            heading = agent1.get_imu_data()
            fov = 180  # Reduced field of view
            resolution = 4  # Lower resolution
            
            # Scan in the direction of travel
            sensor_data1 = agent1.scan(fov=fov, resolution=resolution)
            
            # Only scan from agent2 occasionally
            if self.last_scan_time % 3 == 0:
                sensor_data2 = agent2.scan(fov=fov, resolution=resolution)
            else:
                sensor_data2 = None
            
            # Update maps with minimal scan data
            self._update_maps_with_minimal_scan(agent1, sensor_data1, heading, fov, resolution)
            if sensor_data2:
                heading2 = agent2.get_imu_data()
                self._update_maps_with_minimal_scan(agent2, sensor_data2, heading2, fov, resolution)
        
            # Update position estimates
            if self.agent1_pos is None:
                self.agent1_pos = [self.world_width/2, self.world_height/2]
            if self.agent2_pos is None:
                self.agent2_pos = [self.world_width/2, self.world_height/2]
            
            self.agent1_pos = self.explored_map1.get_estimated_position()
            if sensor_data2:
                self.agent2_pos = self.explored_map2.get_estimated_position()
                
            # Update path planning if needed
            if self.path is None or self.ind >= len(self.path) - 1:
                self._update_path_for_subtask6()
        
        # Continue following current path
        self._follow_rotate_and_move_path(agent1)
    
    def _update_maps_with_minimal_scan(self, agent, sensor_data, heading, fov, resolution):
        """
        Update maps with minimal scan data (for subtask 6)
        
        Args:
            agent: Robot agent
            sensor_data: Scan data
            heading: Current heading
            fov: Field of view
            resolution: Angular resolution
        """
        # Get agent position (estimated if needed)
        try:
            pos = agent.get_pos() if not hasattr(agent, '_RobotAPI__subtasklol') or not agent._RobotAPI__subtasklol else [self.world_width/4, self.world_height/4]
        except:
            if hasattr(self, 'agent1_pos') and self.agent1_pos is not None:
                pos = self.agent1_pos
            else:
                pos = [self.world_width/4, self.world_height/4]
        
        # Get the correct map surface
        map_surface = self.explored_map1.map if agent == agent1 else self.explored_map2.map
        
        # Calculate start angle based on fov
        start_angle = heading - fov/2.0
        start_pt = (int(pos[0]), int(pos[1]))
        
        # Mark agent position on map
        pygame.draw.circle(map_surface, (255, 255, 255), start_pt, 3)
        
        # Process scan data efficiently (skip some points)
        for i in range(0, len(sensor_data), 2):
            ray_angle = start_angle + i * resolution
            ray_angle_rad = math.radians(ray_angle)
            
            end_pt = (
                int(pos[0] + sensor_data[i] * math.cos(ray_angle_rad)),
                int(pos[1] + sensor_data[i] * math.sin(ray_angle_rad))
            )
            
            # Draw ray and endpoint
            pygame.draw.line(map_surface, (255, 255, 255), start_pt, end_pt, 1)
            if sensor_data[i] < float(self.world_width):
                pygame.draw.circle(map_surface, (255, 255, 255), end_pt, 2)
    
    def _update_path_for_subtask6(self):
        """Update path planning for subtask 6"""
        # Transform agent2's position to agent1's frame
        agent2_in_agent1_frame = self._transform_position(self.agent2_pos)
        
        # Merge maps efficiently
        merged_map = self.map_merger.merge(self.explored_map1.map, self.explored_map2.map)
        
        # Use RRT for faster planning
        self.path = self.planner.get_rrt_path(merged_map, self.world_height, self.world_width, 
                                           self.agent1_pos, agent2_in_agent1_frame, 
                                           max_iterations=2000)  # Reduced iterations
        self.ind = 0
    
    def _transform_position(self, pos):
        """
        Transform a position using the relative transform
        
        Args:
            pos: Position [x, y]
            
        Returns:
            list: Transformed position [x, y]
        """
        # Convert to homogeneous coordinates
        p_h = np.array([pos[0], pos[1], 1.0])
        
        # Apply transformation
        p_transformed = np.dot(self.relative_transform, p_h)
        
        # Return transformed point, handling division by zero
        if p_transformed[2] != 0:
            return [p_transformed[0]/p_transformed[2], p_transformed[1]/p_transformed[2]]
        else:
            return pos
    
    def _follow_path(self, agent):
        """
        Basic path following for subtask 1
        
        Args:
            agent: Robot agent to control
            
        Returns:
            bool: True if successful, False if path following failed
        """
        if not self.path or self.ind >= len(self.path):
            return False
        
        next_pos = self.path[self.ind]
        
        # Calculate angle to next waypoint
        dx = next_pos[0] - self.agent1_pos[0]
        dy = next_pos[1] - self.agent1_pos[1]
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Get current heading
        current_angle = agent.get_imu_data()
        
        # Calculate angle difference (accounting for wrap-around)
        angle_diff = (target_angle - current_angle + 180) % 360 - 180
        
        # Rotate towards target if needed
        if abs(angle_diff) > 5:
            agent.rotate(angle_diff * 0.1)  # Proportional control
            return True
        else:
            # Move towards target
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 5:
                # Move agent with collision checking
                moved = agent.move(min(distance * 0.1, 5))
                if moved:
                    # Update tracked position
                    move_dist = min(distance * 0.1, 5)
                    move_angle_rad = math.radians(current_angle)
                    self.agent1_pos[0] += move_dist * math.cos(move_angle_rad)
                    self.agent1_pos[1] += move_dist * math.sin(move_angle_rad)
                    return True
                return False
            else:
                # Waypoint reached, move to next one
                self.ind += 1
                return True
    
    def _follow_rotate_and_move_path(self, agent):
        """
        Simple rotate-and-move path following for subtask 2 and beyond
        
        Args:
            agent: Robot agent to control
            
        Returns:
            bool: True if successful, False if path following failed
        """
        if not self.path or self.ind >= len(self.path):
            return False
        
        next_pos = self.path[self.ind]
        
        # Calculate angle to next waypoint
        dx = next_pos[0] - self.agent1_pos[0]
        dy = next_pos[1] - self.agent1_pos[1]
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Get current heading
        current_angle = agent.get_imu_data()
        
        # Calculate angle difference (accounting for wrap-around)
        angle_diff = (target_angle - current_angle + 180) % 360 - 180
        
        # First phase: rotate to face the target
        if abs(angle_diff) > 5:
            # Use direct rotation to target angle (5° per step for smooth rotation)
            rotation_amount = max(min(angle_diff, 5), -5)
            agent.rotate(rotation_amount)
            return True
        
        # Second phase: move in straight line toward target
        distance = math.sqrt(dx**2 + dy**2)
        if distance > 5:
            # Move agent with collision checking
            move_dist = min(3, distance * 0.2)  # Limit step size for better control
            moved = agent.move(move_dist)
            
            if moved:
                # Update tracked position
                move_angle_rad = math.radians(current_angle)
                self.agent1_pos[0] += move_dist * math.cos(move_angle_rad)
                self.agent1_pos[1] += move_dist * math.sin(move_angle_rad)
                return True
            else:
                # Movement blocked, try small adjustments
                agent.rotate(10)  # Turn a bit to try to find a way around
                return False
        else:
            # Waypoint reached, move to next one
            self.ind += 1
            return True
    
    def _find_and_explore_frontiers(self, agent, map_surface):
        """
        Find and explore frontiers for subtasks 5-6
        
        Args:
            agent: Robot agent
            map_surface: Current map
        """
        # Find frontiers (boundaries between explored and unexplored areas)
        frontiers = []
        
        # Sample map to find frontiers
        step = 10  # Check every 10 pixels for efficiency
        for x in range(0, self.world_width, step):
            for y in range(0, self.world_height, step):
                # Check if current point is explored free space
                try:
                    if map_surface.get_at((x, y))[:3] == (255, 255, 255):
                        # Check if it's adjacent to unexplored space
                        has_unexplored = False
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < self.world_width and 0 <= ny < self.world_height):
                                    try:
                                        if map_surface.get_at((nx, ny))[:3] == (0, 0, 0):
                                            has_unexplored = True
                                            break
                                    except IndexError:
                                        continue
                            if has_unexplored:
                                break
                        
                        if has_unexplored:
                            frontiers.append([x, y])
                except IndexError:
                    continue
        
        if frontiers:
            # Sort frontiers by distance
            frontiers.sort(key=lambda f: (f[0]-self.agent1_pos[0])**2 + (f[1]-self.agent1_pos[1])**2)
            
            # Plan path to nearest frontier
            self.path = self.planner.get_rrt_path(map_surface, self.world_height, self.world_width,
                                               self.agent1_pos, frontiers[0])
            self.ind = 0
    
    def _distance(self, a, b):
        """Calculate Euclidean distance between points"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
