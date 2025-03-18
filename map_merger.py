import pygame
import numpy as np
import cv2

WHITE = (255, 255, 255)

class MapMerger:
    def __init__(self, world_width, world_height):
        """
        Initialize map merger
        
        Args:
            world_width (int): Width of the world
            world_height (int): Height of the world
        """
        self.world_width = world_width
        self.world_height = world_height
        self.merged_map = pygame.Surface((world_width, world_height))
        self.merged_map.fill((0, 0, 0))
        
        # For scan matching in subtasks 3-5
        self.transform = np.eye(3)  # Identity transformation
    
    def merge(self, map1, map2):
        """
        Merge two maps together using a simple union operation
        
        Args:
            map1: First map surface
            map2: Second map surface
        
        Returns:
            pygame.Surface: Merged map
        """
        # Create a new surface for the merged map
        merged = pygame.Surface((self.world_width, self.world_height))
        merged.fill((0, 0, 0))
        
        # Simple merging algorithm: if either map has a white pixel, make it white in the merged map
        for x in range(self.world_width):
            for y in range(self.world_height):
                try:
                    # Check both maps and take union
                    if (map1.get_at((x, y))[:3] == WHITE or 
                        map2.get_at((x, y))[:3] == WHITE):
                        merged.set_at((x, y), WHITE)
                except IndexError:
                    continue
        
        self.merged_map = merged
        return merged
    
    def scan_matching_merge(self, map1, map2):
        """
        Advanced map merging using scan matching (for subtasks 3-5)
        
        Args:
            map1: First map surface
            map2: Second map surface
            
        Returns:
            pygame.Surface: Merged map with proper alignment
        """
        try:
            # Convert Pygame surfaces to numpy arrays for OpenCV processing
            map1_array = pygame.surfarray.array3d(map1)
            map2_array = pygame.surfarray.array3d(map2)
            
            # Convert to grayscale
            map1_gray = cv2.cvtColor(map1_array, cv2.COLOR_RGB2GRAY)
            map2_gray = cv2.cvtColor(map2_array, cv2.COLOR_RGB2GRAY)
            
            # Binary thresholding
            _, map1_binary = cv2.threshold(map1_gray, 128, 255, cv2.THRESH_BINARY)
            _, map2_binary = cv2.threshold(map2_gray, 128, 255, cv2.THRESH_BINARY)
            
            # Find feature points in both maps using ORB
            orb = cv2.ORB_create()
            
            # Detect keypoints and compute descriptors
            kp1, des1 = orb.detectAndCompute(map1_binary, None)
            kp2, des2 = orb.detectAndCompute(map2_binary, None)
            
            # If enough features found, match them
            if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
                # Match descriptors using Brute Force matcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                # Sort by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Use best matches to find transformation
                if len(matches) > 10:
                    # Extract matched keypoints
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]])
                    
                    # Find homography matrix
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if H is not None:
                        # Store transformation
                        self.transform = H
                        
                        # Warp map2 to align with map1
                        warped_map2 = cv2.warpPerspective(
                            map2_binary, H, (self.world_width, self.world_height)
                        )
                        
                        # Merge the two aligned maps
                        merged_binary = cv2.bitwise_or(map1_binary, warped_map2)
                        
                        # Convert back to Pygame surface
                        merged_surface = pygame.Surface((self.world_width, self.world_height))
                        merged_surface.fill((0, 0, 0))
                        
                        for x in range(self.world_width):
                            for y in range(self.world_height):
                                if x < merged_binary.shape[1] and y < merged_binary.shape[0]:
                                    if merged_binary[y, x] > 0:
                                        merged_surface.set_at((x, y), WHITE)
                        
                        self.merged_map = merged_surface
                        return merged_surface
        
        except Exception as e:
            print(f"Scan matching error: {e}")
        
        # Fallback to simple merge if scan matching fails
        return self.merge(map1, map2)
    
    def transform_point(self, point):
        """
        Transform a point from map2 coordinate system to map1 coordinate system
        
        Args:
            point: [x, y] coordinate in map2 system
            
        Returns:
            list: [x, y] coordinate in map1 system
        """
        # Apply homography transformation to the point
        pt = np.array([[point[0], point[1], 1.0]])
        transformed_pt = np.dot(self.transform, pt.T).T
        
        # Normalize homogeneous coordinates
        if transformed_pt[0][2] != 0:
            transformed_pt = transformed_pt[0] / transformed_pt[0][2]
            return [transformed_pt[0], transformed_pt[1]]
        else:
            return point  # Return original point if transformation fails
