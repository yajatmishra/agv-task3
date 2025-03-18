import pygame
import sys
import math
from robot_api import RobotAPI
from pipeline_task import Pipeline 
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()

WHITE = (255, 255, 255)
BROWN = (181, 101, 29) 

def handle_image_upload(shared_world, world_width, world_height, agent):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            print(file_path)
            img = pygame.image.load(file_path)
            img = pygame.transform.scale(img, (world_width, world_height))
            for x in range(world_width):
                for y in range(world_height):
                    if img.get_at((x, y))[:3] == BROWN:
                        agent.edit_wall(x, y, 1)
                    else:
                        agent.erase_wall(x, y, 1)
            print("Loaded map from", file_path)
        except Exception as e:
            print("Error loading image:", str(e))

def handle_image_save(shared_world):
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png")])
    if file_path:
        try:
            pygame.image.save(shared_world, file_path)
            print("Map saved to", file_path)
        except Exception as e:
            print("Error saving image:", str(e))

def main():
    pygame.init()
    # World and UI parameters.
    world_width, world_height = 800, 800      
    button_height = 50                         # Top UI area height.
    window_width = world_width
    window_height = world_height + button_height
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("2D Mapping & Pipeline Task")
    clock = pygame.time.Clock()

    # Create a shared ground-truth world for wall editing.
    shared_world = pygame.Surface((world_width, world_height))
    shared_world.fill((255, 255, 255))        

    # Instantiate two agents with different starting positions.
    agent1 = RobotAPI(world_width, world_height, shared_world, start_pos=[world_width * 0.6, world_height * 0.6])
    agent2 = RobotAPI(world_width, world_height, shared_world, start_pos=[world_width * 0.3, world_height * 0.3])

    # Create a class instance for the pipeline
    pipeline = Pipeline(world_height, world_width, shared_world)

    # Create an explored map surface for pipeline mode.
    explored_surface = pygame.Surface((world_width, world_height))
    explored_surface.fill((0, 0, 0))            # Unexplored areas appear black.

    # Button colors.
    GREEN = (125, 250, 145)           
    LIGHT_ORANGE = (255, 200, 150)      
    BLUE = (52, 204, 235)
    PINKISH = (209, 143, 235)

    # UI Button definitions.
    add_button_rect    = pygame.Rect(10, 5, 120, 30)
    erase_button_rect  = pygame.Rect(140, 5, 120, 30)
    start_button_rect  = pygame.Rect(270, 5, 140, 30)
    reset_button_rect  = pygame.Rect(270, 5, 140, 30)
    view_button_rect   = pygame.Rect(420, 5, 140, 30)
    upload_button_rect = pygame.Rect(570, 5, 110, 30)
    save_button_rect   = pygame.Rect(685, 5, 110, 30)

    # Modes.
    simulation_mode = "edit"    # "edit" or "pipeline"
    edit_mode = "none"          # For wall editing: "add", "erase", or "none"
    viz_mode = "full"           # "full" (complete map) or "explored" (only scanned areas)
    brush_radius = 8

    # Manual control parameters (for both modes).
    rotation_speed_manual = 5   # Degrees per frame.
    move_speed_manual = 2       # Units per frame.

    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        # --- Event Processing ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if simulation_mode == "edit":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        if add_button_rect.collidepoint(mx, my):
                            edit_mode = "add" if edit_mode != "add" else "none"
                        elif erase_button_rect.collidepoint(mx, my):
                            edit_mode = "erase" if edit_mode != "erase" else "none"
                        elif start_button_rect.collidepoint(mx, my):
                            simulation_mode = "pipeline"
                            explored_surface.fill((0, 0, 0))
                            edit_mode = "none"
                        elif view_button_rect.collidepoint(mx, my):
                            viz_mode = "explored" if viz_mode == "full" else "full"
                        elif upload_button_rect.collidepoint(mx, my):
                            handle_image_upload(shared_world, world_width, world_height, agent1)
                        elif save_button_rect.collidepoint(mx, my):
                            handle_image_save(shared_world)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_e:
                        edit_mode = "none"
            elif simulation_mode == "pipeline":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        if reset_button_rect.collidepoint(mx, my):
                            simulation_mode = "edit"
                        elif view_button_rect.collidepoint(mx, my):
                            viz_mode = "explored" if viz_mode == "full" else "full"
                        elif upload_button_rect.collidepoint(mx, my):
                            handle_image_upload(shared_world, world_width, world_height, agent1)
                        elif save_button_rect.collidepoint(mx, my):
                            handle_image_save(shared_world)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        viz_mode = "explored" if viz_mode == "full" else "full"

        # --- Manual Control for Agents (Both Modes) ---
        keys = pygame.key.get_pressed()
        action = False
        # Agent1 controlled via arrow keys.
        if keys[pygame.K_LEFT]:
            agent1.rotate(-rotation_speed_manual)
            action = True
        if keys[pygame.K_RIGHT]:
            agent1.rotate(rotation_speed_manual)
            action = True
        if keys[pygame.K_UP]:
            if agent1.move(move_speed_manual):
                action = True
        if keys[pygame.K_DOWN]:
            if agent1.move(-move_speed_manual):
                action = True
        # Agent2 controlled via WASD.
        if keys[pygame.K_a]:
            agent2.rotate(-rotation_speed_manual)
            action = True
        if keys[pygame.K_d]:
            agent2.rotate(rotation_speed_manual)
            action = True
        if keys[pygame.K_w]:
            if agent2.move(move_speed_manual):
                action = True
        if keys[pygame.K_s]:
            if agent2.move(-move_speed_manual):
                action = True
        if keys[pygame.K_k]:
            agent1.get_pos()

        # --- can uncomment for debugging 
        # if action:
            # print("Agent1 LIDAR:", agent1.scan())
            # print("Agent1 IMU:", agent1.get_imu_data())
            # print("Agent2 LIDAR:", agent2.scan())
            # print("Agent2 IMU:", agent2.get_imu_data())

        # --- Wall Editing in Edit Mode ---
        if simulation_mode == "edit":
            if pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                if my >= button_height:
                    wx, wy = mx, my - button_height
                    if edit_mode == "add":
                        agent1.edit_wall(wx, wy, brush_radius)
                    elif edit_mode == "erase":
                        agent1.erase_wall(wx, wy, brush_radius)

        # --- Pipeline Mode: Update Explored Map and Invoke Pipeline ---
        if simulation_mode == "pipeline":
            for agent in [agent1, agent2]:
                agent.update_explored(explored_surface)
            pipeline.work(agent1, agent2)
        else: pipeline.reset()

        # --- Drawing Section ---
        screen.fill((128, 128, 128))
        # Draw UI Buttons.
        if simulation_mode == "edit":
            pygame.draw.rect(screen, LIGHT_ORANGE if edit_mode=="add" else GREEN, add_button_rect)
            add_text = font.render("Add Walls", True, (0, 0, 0))
            screen.blit(add_text, (add_button_rect.x+5, add_button_rect.y+5))
            
            pygame.draw.rect(screen, LIGHT_ORANGE if edit_mode=="erase" else GREEN, erase_button_rect)
            erase_text = font.render("Erase Walls", True, (0, 0, 0))
            screen.blit(erase_text, (erase_button_rect.x+5, erase_button_rect.y+5))
            
            pygame.draw.rect(screen, GREEN, start_button_rect)
            start_text = font.render("Start Pipeline", True, (0, 0, 0))
            screen.blit(start_text, (start_button_rect.x+5, start_button_rect.y+5))
        else:
            pygame.draw.rect(screen, LIGHT_ORANGE, reset_button_rect)
            reset_text = font.render("Reset", True, (0, 0, 0))
            screen.blit(reset_text, (reset_button_rect.x+5, reset_button_rect.y+5))
        
        pygame.draw.rect(screen, BLUE, view_button_rect)
        view_text = font.render("View: " + viz_mode.capitalize(), True, (0, 0, 0))
        screen.blit(view_text, (view_button_rect.x+5, view_button_rect.y+5))
        
        # Draw new Upload and Save buttons (always visible).
        pygame.draw.rect(screen, PINKISH, upload_button_rect)
        upload_text = font.render("Upload Map", True, (0, 0, 0))
        screen.blit(upload_text, (upload_button_rect.x+5, upload_button_rect.y+5))
        
        pygame.draw.rect(screen, PINKISH, save_button_rect)
        save_text = font.render("Save Map", True, (0, 0, 0))
        screen.blit(save_text, (save_button_rect.x+5, save_button_rect.y+5))
        
        # Blit the environment.
        if viz_mode == "full":
            screen.blit(shared_world, (0, button_height))
        else:
            screen.blit(explored_surface, (0, button_height))
        
        # Draw agents.
        agent1.draw_agent(screen, offset=(0, button_height), agent_color=(0, 184, 176))
        agent2.draw_agent(screen, offset=(0, button_height), agent_color=(124, 60, 130))
        
        pygame.display.flip()
        clock.tick(40)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
