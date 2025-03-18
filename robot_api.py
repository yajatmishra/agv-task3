import math
import pygame

# Color definitions.
WHITE = (255, 255, 255)
BROWN = (181, 101, 29)  # Light brown walls.
EPS = 1e-3

class RobotAPI:
    def __init__(self, world_width=600, world_height=600, world_surface=None, start_pos=None):
        """
        Creates a continuous world. If a shared world_surface is provided, it is used;
        otherwise a new white surface is created.
        The robot’s state is stored as continuous (pixel) coordinates plus an orientation (in degrees).
        
        IMPORTANT: Only sensor streams (scan() and get_imu_data()) and control commands (move, rotate)
        are meant for external use. Visualization updates occur internally without exposing an absolute pose.
        """
        self.__subtasklol = False
        self.__world_width = world_width
        self.__world_height = world_height
        self.__max_move_speed = 5
        if world_surface is None:
            self.__world_surface = pygame.Surface((world_width, world_height))
            self.__world_surface.fill(WHITE)
        else:
            self.__world_surface = world_surface
        if start_pos is None:
            self.__pos = [world_width / 2.0, world_height / 2.0]
        else:
            self.__pos = list(start_pos)
        self.__angle = 0.0  # 0° means facing right.
        self.__max_range = float(world_width)
    
    def move(self, distance):
        """
        Moves the robot forward (or backward if negative) by dist units along its current heading.
        The move is blocked if the destination is out-of-bounds or if that pixel is WALL_COLOR.
        Returns True if successful.
        """
        distance = abs(distance)
        dist = min(distance, self.__max_move_speed)
        new_x = self.__pos[0] + dist * math.cos(math.radians(self.__angle))
        new_y = self.__pos[1] + dist * math.sin(math.radians(self.__angle))
        if new_x < 0 or new_x >= self.__world_width or new_y < 0 or new_y >= self.__world_height:
            return False
        if self.__world_surface.get_at((int(new_x), int(new_y)))[:3] != WHITE:
            return False
        self.__pos = [new_x, new_y]
        if (abs(dist-distance) < EPS): return True
        else: return self.move(distance-dist)

    def rotate(self, deg):
        """
        Rotates the robot by deg degrees clockwise (pass a negative value for counterclockwise).
        """
        self.__angle = (self.__angle + deg) % 360

    def scan(self, fov=360, resolution=2):
        """
        Performs a LIDAR scan: casts rays over fov degrees (default 360°) at intervals of resolution degrees.
        Each ray returns the distance (in pixels) from the robot’s current position to the first encountered wall.
        Returns a list of distance measurements.
        """
        num_rays = int(fov / resolution)
        start_angle = self.__angle - fov / 2.0
        measurements = []
        for i in range(num_rays):
            ray_angle = start_angle + i * resolution
            ray_angle_rad = math.radians(ray_angle)
            d = self.__raycast(ray_angle_rad)
            measurements.append(d)
        return measurements

    def __raycast(self, angle_rad):
        """
        Helper: casts a ray (stepping 1 pixel at a time) from the current position along angle_rad.
        Returns the measured distance to the first wall (WALL_COLOR) or to the maximum range.
        """
        step = 1.0
        distance = 0.0
        while distance < self.__max_range:
            test_x = self.__pos[0] + distance * math.cos(angle_rad)
            test_y = self.__pos[1] + distance * math.sin(angle_rad)
            ix, iy = int(test_x), int(test_y)
            if ix < 0 or iy < 0 or ix >= self.__world_width or iy >= self.__world_height:
                return self.__max_range
            if self.__world_surface.get_at((ix, iy))[:3] == BROWN:
                return distance
            distance += step
        return self.__max_range

    def get_imu_data(self):
        """
        Returns simulated IMU data (the current heading in degrees).
        """
        return self.__angle

    def edit_wall(self, world_x, world_y, brush_radius=5):
        """
        Draws a wall onto the world using a circular brush in WALL_COLOR.
        """
        pygame.draw.circle(self.__world_surface, BROWN, (int(world_x), int(world_y)), brush_radius)

    def erase_wall(self, world_x, world_y, brush_radius=5):
        """
        Erases a wall by painting a filled circle in WHITE onto the world.
        """
        pygame.draw.circle(self.__world_surface, WHITE, (int(world_x), int(world_y)), brush_radius)

    # Visualization methods (for internal use only – not to be used for SLAM/mapping).
    def draw_agent(self, target_surface, offset=(0,0), agent_color=(0, 0, 255), heading_color=(255, 0, 0)):
        """
        Draws the agent on target_surface using its internal state.
        The offset is applied to account for UI regions.
        """
        x = int(self.__pos[0]) + offset[0]
        y = int(self.__pos[1]) + offset[1]
        pygame.draw.circle(target_surface, agent_color, (x, y), 8)
        ex = int(x + math.cos(math.radians(self.__angle)) * 15)
        ey = int(y + math.sin(math.radians(self.__angle)) * 15)
        pygame.draw.line(target_surface, heading_color, (x, y), (ex, ey), 3)

    def update_explored(self, explored_surface):
        """
        Updates the explored_surface based solely on this agent's current lidar scan.
        White lines (and small white circles) are drawn from the agent's current location to the endpoints
        indicated by each scan measurement. (This simulates how much of the environment has been "seen".)
        """
        sensor_data = self.scan(fov=360, resolution=2)
        fov = 360
        resolution = 2
        heading = self.get_imu_data()
        start_angle = heading - fov/2.0
        # Use the internal state (private __pos) for computing the scan origin.
        start_pt = (int(self.__pos[0]), int(self.__pos[1]))
        for i, distance in enumerate(sensor_data):
            ray_angle = start_angle + i * resolution
            ray_angle_rad = math.radians(ray_angle)
            end_pt = (int(self.__pos[0] + distance * math.cos(ray_angle_rad)),
                      int(self.__pos[1] + distance * math.sin(ray_angle_rad)))
            pygame.draw.line(explored_surface, WHITE, start_pt, end_pt, 1)
            if distance < self.__max_range:
                pygame.draw.circle(explored_surface, WHITE, end_pt, 2)

    def get_world(self):
        """
        Returns the full underlying world surface.
        """
        return self.__world_surface
    
    def get_pos(self):
        """
        Returns position of the agent. Can be used only once.
        """
        if not self.__subtasklol:
            self.__subtasklol = True
            return self.__pos
        else:
            print("It's over for you")
