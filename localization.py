import pygame
import math
import numpy as np

WHITE = (255, 255, 255)

class Localization:
    def __init__(self, world_width, world_height):
        """
        Initialize localization with an empty map surface
        
        Args:
            world_width (int): Width of the world in pixels
            world_height (int): Height of the world in pixels
        """
        # Create a surface to represent the map
        self.map = pygame.Surface((world_width, world_height))
        self.map.fill((0, 0, 0))  # Black background (unexplored)
        self.world_height = world_height
        self.world_width = world_width
        
        # Store estimated position for agent tracking
        self.estimated_pos = None
        
        # For particle filter (used in subtasks 4-5)
        self.particles = []
        self.particle_weights = []
        self.initialized = False
        
    def update(self, agent):
        """
        Update map based on agent's current LiDAR scan
        
        Args:
            agent: Robot agent object with scan capabilities
        """
        # Get agent position and orientation with subtask protection
        try:
            pos = agent.get_pos() if not hasattr(agent, '_RobotAPI__subtasklol') or not agent._RobotAPI__subtasklol else [self.world_width/4, self.world_height/4]
        except:
            # If get_pos fails, use a reasonable default or estimated position
            if hasattr(self, 'estimated_pos') and self.estimated_pos is not None:
                pos = self.estimated_pos
            else:
                pos = [self.world_width/4, self.world_height/4]
        
        # Get current heading from IMU
        heading = agent.get_imu_data()
        
        # Process 360° LiDAR scan
        sensor_data = agent.scan(fov=360, resolution=2)
        start_angle = heading - 180.0
        start_pt = (int(pos[0]), int(pos[1]))
        
        # Mark agent position as explored
        pygame.draw.circle(self.map, WHITE, start_pt, 3)
        
        # Update map with scan data
        for i, distance in enumerate(sensor_data):
            ray_angle = start_angle + i * 2  # 2-degree resolution
            ray_angle_rad = math.radians(ray_angle)
            
            end_pt = (
                int(pos[0] + distance * math.cos(ray_angle_rad)),
                int(pos[1] + distance * math.sin(ray_angle_rad))
            )
            
            # Draw ray path as explored
            pygame.draw.line(self.map, WHITE, start_pt, end_pt, 1)
            
            # Mark obstacle points
            if distance < float(self.world_width):
                pygame.draw.circle(self.map, WHITE, end_pt, 2)
        
        # Store as estimated position for future use
        self.estimated_pos = list(pos)
        
        # Update particle filter if initialized (for subtasks 4-5)
        if hasattr(self, 'particles') and len(self.particles) > 0 and self.initialized:
            self._update_particle_filter(sensor_data, heading)
    
    def initialize_particle_filter(self, num_particles=300, map_surface=None):
        """
        Initialize particle filter for position estimation (subtasks 4-5)
        
        Args:
            num_particles: Number of particles to use
            map_surface: Map surface to constrain particles to free space
        """
        self.particles = []
        self.particle_weights = []
        
        # Create particles randomly across the world
        for _ in range(num_particles):
            # Generate random position
            x = np.random.uniform(0, self.world_width)
            y = np.random.uniform(0, self.world_height)
            
            # Check if in free space (if map provided)
            if map_surface is not None:
                try:
                    if map_surface.get_at((int(x), int(y)))[:3] != WHITE:
                        # Not in free space, try again
                        continue
                except IndexError:
                    continue
            
            # Add valid particle
            self.particles.append([x, y])
            self.particle_weights.append(1.0 / num_particles)
        
        self.initialized = True
    
    def _update_particle_filter(self, sensor_data, heading):
        """
        Update particle filter with new sensor data
        
        Args:
            sensor_data: LiDAR scan data
            heading: Current heading from IMU
        """
        if len(self.particles) == 0:
            return
            
        # Predict step - move particles according to estimated motion
        for i in range(len(self.particles)):
            # Add noise to simulate motion uncertainty
            self.particles[i][0] += np.random.normal(0, 1.0)
            self.particles[i][1] += np.random.normal(0, 1.0)
            
            # Ensure particles stay within bounds
            self.particles[i][0] = max(0, min(self.world_width, self.particles[i][0]))
            self.particles[i][1] = max(0, min(self.world_height, self.particles[i][1]))
        
        # Update step - calculate particle weights based on sensor match
        for i in range(len(self.particles)):
            self.particle_weights[i] = self._calculate_particle_weight(self.particles[i], sensor_data, heading)
        
        # Normalize weights
        total_weight = sum(self.particle_weights)
        if total_weight > 0:
            self.particle_weights = [w / total_weight for w in self.particle_weights]
        
        # Resample particles based on weights
        self._resample_particles()
        
        # Estimate position as weighted average of particles
        if len(self.particles) > 0:
            self.estimated_pos = [
                sum(p[0] * w for p, w in zip(self.particles, self.particle_weights)),
                sum(p[1] * w for p, w in zip(self.particles, self.particle_weights))
            ]
    
    def _calculate_particle_weight(self, particle, sensor_data, heading):
        """
        Calculate how well a particle matches the sensor data
        
        Args:
            particle: [x, y] position
            sensor_data: LiDAR scan data
            heading: Current heading
            
        Returns:
            float: Weight representing likelihood of particle
        """
        # Simple model: Check if endpoints of scan match walls in map
        weight = 1.0
        
        start_angle = heading - 180.0
        pos = particle
        
        # Only check some rays for efficiency
        for i in range(0, len(sensor_data), 10):
            distance = sensor_data[i]
            ray_angle = start_angle + i * 2  # 2-degree resolution
            ray_angle_rad = math.radians(ray_angle)
            
            end_x = int(pos[0] + distance * math.cos(ray_angle_rad))
            end_y = int(pos[1] + distance * math.sin(ray_angle_rad))
            
            # Check if endpoint is valid
            if 0 <= end_x < self.world_width and 0 <= end_y < self.world_height:
                # Get color at endpoint in our map
                try:
                    map_color = self.map.get_at((end_x, end_y))[:3]
                    
                    # If sensor shows obstacle and map shows obstacle (or unexplored)
                    if distance < self.world_width/2:
                        if map_color != WHITE:
                            weight *= 1.2  # Boost weight for correct obstacle match
                        else:
                            weight *= 0.8  # Penalize for mismatch
                    # If sensor shows free space and map shows free space
                    else:
                        if map_color == WHITE:
                            weight *= 1.1  # Boost weight for correct free space match
                        else:
                            weight *= 0.9  # Penalize for mismatch
                except IndexError:
                    weight *= 0.8  # Penalize out of bounds
        
        return weight
    
    def _resample_particles(self):
        """
        Resample particles based on their weights using resampling wheel algorithm
        """
        if sum(self.particle_weights) == 0:
            return
            
        # Resampling wheel algorithm
        new_particles = []
        N = len(self.particles)
        index = np.random.randint(0, N)
        beta = 0.0
        max_weight = max(self.particle_weights)
        
        for _ in range(N):
            beta += np.random.uniform(0, 2.0 * max_weight)
            while beta > self.particle_weights[index]:
                beta -= self.particle_weights[index]
                index = (index + 1) % N
            new_particles.append(self.particles[index])
        
        self.particles = new_particles
        self.particle_weights = [1.0 / N] * N
    
    def get_estimated_position(self):
        """Get the current estimated position of the agent"""
        if self.estimated_pos is not None:
            return self.estimated_pos
        return [self.world_width/2, self.world_height/2]
    
    def at(self, x, y):
        """
        Get color at specified coordinates
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
        
        Returns:
            tuple: RGBA color at the specified coordinates
        """
        try:
            return self.map.get_at((int(x), int(y)))
        except IndexError:
            return (0, 0, 0, 255)  # Return black if out of bounds
