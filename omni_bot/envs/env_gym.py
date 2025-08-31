from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import pygame
from omni_bot.omni_bot import OmniBot  

# Version-0
class OmniBotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, SIZE=[5,5], goal_pose = np.array([1.5,1.0,0.0])):
        super(OmniBotEnv, self).__init__()
        # env size
        self.size = SIZE # size of the env is 5x5 measured in metres
        self.window_size = 512 # pygame default window size of the environment
        self.scale = self.size[0]/self.window_size # resolution from metres to pixels
        # creating out main star/centre of attraction # pose(x,y,yaw=theta=phi)
        self.robot = OmniBot(a=(0.1/2)/2, L=(0.6/2)/2, W=(0.6/2)/2, t=(0.036)/2, initial_pos=(0,0,0)) 
        self.target_pose = goal_pose.reshape(3,1)
        self.robot_trajectory = [] # in [m]
        self.robot_vel_global = np.array([0.0, 0.0, 0.0]).reshape(3,1) # in [m/s, m/s, rad/s]
        # action space init
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        ) # action for the bot continuous values for the bot frame velocities [u, v, r]

        # observation space init
        # for version-0 of the env let the observation space consist of bot pose vector and target/goal pose vector
        self.observation_space = gym.spaces.Dict(
            {
                "bot_pose": gym.spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32),
                "robot_vel_global": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "target_pose": gym.spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
            }
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {
            "bot_pose": self.robot.pose.flatten().astype(np.float32),
            "robot_vel_global": self.robot_vel_global.flatten().astype(np.float32),
            "target_pose": self.target_pose.flatten().astype(np.float32)
        }

    def _get_info(self):
        return {
            "distance_to_target": np.linalg.norm(self.target_pose[0:2]-self.robot.pose[0:2], ord=2).astype(np.float32) # eucledian distance since units are in [m]
            # maybe add some reward type informations later on
        }
    
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        # set the robot to the origin
        self.robot.pose = np.array([0.0, 0.0, 0.0]).reshape(3,1)
        # random robot pose as well as target pose on reset
        # self.robot.pose[0:2] = self.size[0]*np.random.random(size=(2,1)) - self.size[0]/2
        # self.robot.pose[2,0] = np.random.uniform(-np.pi, np.pi) # randomize orientation
        self.robot_trajectory = [self.robot.pose[0:2].flatten().tolist()] # reset trajectory
        self.target_pose = self.target_pose # set initial target pose
        self.target_pose[0:2] = self.size[0]*np.random.random(size=(2,1)) - self.size[0]/2
        self.target_pose[2,0] = np.random.uniform(-np.pi, np.pi) # randomize orientation
        
        observation = self._get_obs()
        info = self._get_info()

        # reset rendering too
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    # main physics and action stepping function lomgic
    def step(self, action):
        # action ---> basically np.array [u, v, r]
        # move bot
        omega = self.robot.inverse_kinematics(body_vel=action) # get wheel vel[rad/s]
        global_bot_vel = self.robot.forward_kinematics(omega=omega)
        # update odometry/pose of bot at physics step rate
        self.robot.update_odom(vel_global=global_bot_vel, dt=1/60) # phys up_rate=60Hz(for now) --> if less than render frames then no ping/lag in viz
        # record observations
        observations = self._get_obs()
        info = self._get_info()
        # print(f"Bot Distance from origin: {np.linalg.norm(self.robot.pose[0:2], ord=2):.3f} m")      
        
        # check for termination conditions
        if np.linalg.norm(self.robot.pose[0:2]-self.target_pose[0:2]) < 1.0e-4: # within 1cm radius
            terminated = True
        elif np.linalg.norm(self.robot.pose[0:2], ord=2) > 3.5: 
            terminated = True # if bot goes out of bounds
        else:
            terminated = False
        truncated = False

        # reward model (here) (for now add a sparse reward model)
        if np.linalg.norm(self.robot.pose[0:2]-self.target_pose[0:2]) < 1.0e-2: # within 10cm radius
            rewards = 2.0
            if abs(self.robot.pose[2,0]-self.target_pose[2,0]) < 1.0e-1: # within 0.1 rads
                rewards = 2.0
        else:
            rewards = -1.0*np.linalg.norm(self.robot.pose[0:2]-self.target_pose[0:2], ord=2) # negative reward for being far away from target
        
        # record trajectory of bot for visualization in window
        self.robot_trajectory.append(self.robot.pose[0:2].flatten().tolist()) # in [m]
        # render the viz frames
        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # adding origin axes annotations for reference
        pygame.draw.line(canvas, (255, 0, 0),self.world_to_window(np.array([0.0, 0.0])),self.world_to_window(np.array([0.5, 0.0])), 2)  # X axis
        pygame.draw.line(canvas, (0, 255, 0),self.world_to_window(np.array([0.0, 0.0])),self.world_to_window(np.array([0.0, 0.5])), 2)  # Y axis
        # adding marker for target pose
        target = self.world_to_window(world_coords=(self.target_pose[0], self.target_pose[1]))
        pygame.draw.circle(canvas, color=(0, 255, 0), center=target, radius=5, width=0)
        # add bot outline 
        bot_body_position = self.robot.get_bot_outline() # (4,2)
        # convert to pixel coords for scaling to pygame window
        bot_body = [self.world_to_window((pos[0],pos[1])) for pos in bot_body_position]
        pygame.draw.polygon(canvas, color=(255,255,0), points=bot_body)
        # adding wheel outlines        
        wheel_positions = self.robot.get_wheel_positions() # (4,2) for 4 wheel
        # canvasing each wheel
        for wheel_pos in wheel_positions:
            wheel_center = self.world_to_window((wheel_pos[0], wheel_pos[1]))
            pygame.draw.circle(canvas, color=(0,0,0), center=wheel_center, radius=5)
        # marking robot trajectory as trace
        robot_trace = [self.world_to_window((pos[0], pos[1])) for pos in self.robot_trajectory]
        if len(robot_trace) > 1:
            pygame.draw.lines(canvas, color=(0,0,255), closed=False, points=robot_trace, width=1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # ---------------------------------------------------------------
    # some helper functions
    def world_to_window(self, world_coords):
        xw, yw = world_coords[0], world_coords[1]
        xp = int((1/self.scale)*(xw + (self.size[0]/2)))
        yp = int(-(1/self.scale)*(yw - (self.size[1]/2)))
        return xp, yp

    def window_to_world(self, pixel_coords):
        xp, yp = pixel_coords[0], pixel_coords[1]
        xw = (self.scale)*(xp - (self.window_size/2))
        yw = (self.scale)*(yp + (self.window_size/2))
        return xw, yw
    # ---------------------------------------------------------------

# version-1 under development
# (down here are the envs for the next versions)
# class OmniBotEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

#     def __init__(self, render_mode=None, SIZE=[5, 5], dt=0.05, grid_size=(20, 20), cell_size=25):
#         super(OmniBotEnv, self).__init__()

#         self.robot = OmniBot(a=0.05, L=0.3, W=0.3, t=0.036, m=4, Icz=0.1)
#         self.robot_trajectory = [] # in [m]
#         self.robot_trace = [] # in pygame pixel units
#         self.distances_to_obstacles = None # in [m]
    
#         [self.size_x, self.size_y] = SIZE # size of the env in [m]
#         self.grid_size = grid_size
#         self.cell_size = cell_size

#         # Define the occupancy grid (0 = free, 1 = occupied, 0.5 = unknown (not in use for now))
#         self.occupancy_grid = np.random.choice([0, 1], size=grid_size, p=[0.99, 0.01])
#         self.resolution = self.size_x/self.grid_size[0] # square physical env
#         self.num_obstacles = self.occupancy_grid[self.occupancy_grid == 1].shape[0]
#         print("*"*25+"OCCUPANCY GRID INFO"+"*"*25)
#         print(f"\nnumber of obstalces: {self.num_obstacles}\n")
#         print("*"*50)

#         # defining the gymnasium observation and action spaces
#         self.observation_space = gym.spaces.Dict({
#             "robot_pose": gym.spaces.Box(-self.size_x+1, self.size_x-1, shape=(3,), dtype=float),
#             "robot_vel": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=float), # global frame velocity
#             "distances_to_obstacles": gym.spaces.Box(0.0, self.size_x*np.sqrt(2), shape=(self.num_obstacles,), dtype=float)
#         })
#         self.action_space = gym.spaces.Discrete(6)
#         self._action_to_direction = {
#             0: np.array([1.0, 0.0, 0.0]), # forward
#             1: np.array([-1.0, 0.0, 0.0]), # backward
#             2: np.array([0.0, 1.0, 0.0]), # strafe forward
#             3: np.array([0.0, -1.0, 0.0]), # strafe backward
#             4: np.array([0.0, 0.0, 1.0]), # turn left
#             5: np.array([0.0, 0.0, -1.0]), # turn right
#         }

#         # ----------------------------------------------------
#         self.window_size = 500
#         self.scale = self.window_size/(self.size_x) # from [m] to [pygame pixels]
#         self.clock = None
#         self.window =None

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode
#         self.dt = 1/self.metadata["render_fps"]

#     def world_to_window(self, pose): # unit conversion from m->pixels
#         screen_x = int((pose[0] + self.size_x/2) * self.scale)
#         screen_y = int((self.size_y/2 - pose[1]) * self.scale)
#         return screen_x,screen_y

#     def occgrid_to_world(self, pixel, map_origin=(0.0,0.0)): # map origin is in metres scale ,  pixel in (y_pix, x_pix)
#         x_m = (pixel[1] - self.grid_size[1] // 2 + 0.5) * self.resolution - map_origin[0]
#         y_m = -(pixel[0] - self.grid_size[0] // 2 + 0.5) * self.resolution - map_origin[1]
#         return x_m, y_m
    
#     def occgrid_to_window(self, pixel):
#         pose = self.occgrid_to_world(pixel)
#         return self.world_to_window(pose)
    
#     def get_obstacle_info(self):
#         indices = np.where(self.occupancy_grid==1)
#         obstacle_coords = list(zip(indices[0], indices[1]))
#         return obstacle_coords
        
#     def _draw_obstacle_circle_cost(self, canvas, epsilon=5.0):
#         obstacle_coords = self.get_obstacle_info()
#         for obs in obstacle_coords:
#             # world_x, world_y = self.occgrid_to_world(obs)
#             # print(f"Obstacle at grid {obs} is at world coordinates: ({world_x:.2f}, {world_y:.2f})") 
#             x, y = self.occgrid_to_window(obs)
#             pygame.draw.circle(canvas, (255, 0, 0), (x, y), epsilon/self.resolution, 2)
    
#     def calculate_distance_to_obs(self, obstacle_coords):
#         # taking in obstacle coords in pixel format [y, x] (occupancy grid frame)
#         dists_to_obs = []
#         for i, obs in enumerate(obstacle_coords):
#             xm, ym = self.occgrid_to_world(obs)
#             dist_to_obs = np.linalg.norm(self.robot.pose[0:2, 0]-np.array([xm, ym]))
#             # dist_to_obs = np.sqrt((self.robot.pose[0,0]-xm)**2 + (self.robot.pose[1,0]-ym)**2)
#             dists_to_obs.append((i, dist_to_obs))
#         return dists_to_obs

#     def step(self, action): # Apply action to robot, update state, reward and return observations
#         # one step process
#         actions = self._action_to_direction[action]
#         omega = self.robot.inverse_kinematics(body_vel=actions)
#         eta_dot = self.robot.forward_kinematics(omega=omega)
#         self.robot.update_odom(vel_global=eta_dot, dt=self.dt)
        
#         # storing robot trajectory as real world or pixel coords
#         self.robot_trace.append(self.world_to_window([self.robot.pose[0,0], self.robot.pose[1,0]])) # in pygame pixel coords
#         self.robot_trajectory.append([self.robot.pose[0,0], self.robot.pose[1,0]]) # in real world coords

#         # calculating the distance to obstacles, basically the distance to black/occupied grid cell
#         self.distances_to_obstacles = self.calculate_distance_to_obs(self.get_obstacle_info())
#         if self.distances_to_obstacles is None:
#             print(f"waiting to calculate distances to obstalces...")
#             return

#         observations = {
#             "robot_pose": self.robot.pose,
#             "robot_vel": eta_dot,
#             "distances_to_obstacles": self.distances_to_obstacles 
#         }
#         reward = 0.0
#         terminated = np.linalg.norm(self.robot.pose[0:2]) > 2.4 # done if bot goes out of bounds
#         truncated = False
#         info = {}

#         if self.render_mode == "human":
#             self._render_frame()

#         return observations, reward, terminated, truncated, info

#     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#         # Reset all robots to the origin and other observations to initial states of the env.
#         super().reset(seed=seed, options=options)

#         self.robot.pose = np.array([0.0, 0.0, 0.0]).reshape(3,1)
#         robot_vel = np.array([0.0, 0.0, 0.0]).reshape(3,1)
#         # initial values 
#         self.distances_to_obstacles = self.calculate_distance_to_obs(self.get_obstacle_info())
#         observations = {
#             "robot_pose": self.robot.pose,
#             "robot_vel": robot_vel,
#             "distances_to_obstacles": self.distances_to_obstacles
#         }
#         info = {}
#         return observations, info

#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_frame()
    
#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             pygame.display.set_caption('omni_bot_env')
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()
        
#         # Create a canvas
#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))
        
#         # creating the occupancy grid based map-env
#         # Draw the occupancy grid
#         self._draw_occupancy_grid(canvas)
#         self._draw_obstacle_circle_cost(canvas)

#         # Draw origin axes
#         pygame.draw.line(canvas, (255, 0, 0), self.world_to_window([0,0]), self.world_to_window([0.3, 0]), 3)  # X-axis
#         pygame.draw.line(canvas, (0, 255, 0), self.world_to_window([0,0]), self.world_to_window([0, 0.3]), 3)  # Y-axis

#         bot_body = self.robot.get_bot_outline() # (4x2) matrix (array of [x,y])
#         body_outline = [self.world_to_window(corner) for corner in bot_body]
#         # adding bot_outline
#         pygame.draw.polygon(canvas, (255, 216, 0), body_outline)

#         # wheel outlines (going with black circular wheels for now)
#         wheel_positions = self.robot.get_wheel_positions()
#         wheel_centres = [self.world_to_window(wheel_pose) for wheel_pose in wheel_positions]
#         for wheel_centre in wheel_centres:
#             pygame.draw.circle(canvas, (0, 0, 0), wheel_centre, 5)

#         # robot trajectory trace
#         for trace in self.robot_trace:
#             pygame.draw.circle(canvas, (0, 255, 0), trace, 1, 1)
        
#         if self.render_mode == "human":
#             # The following line copies our drawings from `canvas` to the visible window
#             self.window.blit(canvas, (0,0))
#             pygame.event.pump()
#             pygame.display.update()
#             # We need to ensure that human-rendering occurs at the predefined framerate.
#             # The following line will automatically add a delay to keep the framerate stable.
#             self.clock.tick(self.metadata["render_fps"])
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )
    
#     def _draw_occupancy_grid(self, canvas):
#         for y in range(self.grid_size[0]):
#             for x in range(self.grid_size[1]):
#                 pixel_value = self.occupancy_grid[y, x]
#                 if pixel_value == 0:
#                     color = (255, 255, 255) # free space pixel_values < (0,255) >
#                 elif pixel_value == 1:
#                     color = (0, 0, 0)

#                 # drawing the grid-cell (each cell is a square)
#                 pygame.draw.rect(
#                     canvas,
#                     color,
#                     pygame.Rect(
#                         x*self.cell_size,
#                         y*self.cell_size,
#                         self.cell_size,
#                         self.cell_size
#                     )
#                 )

#     def save_trajectory(self, filename="robot_trajectory.npy"):
#         #Save robot trajectories to a file.
#         np.save(filename, np.array(self.robot_trajectory))

#     def load_trajectory(self, filename="robot_trajectory.npy"):
#         #Load and visualize previous trajectories.
#         data = np.load(filename, allow_pickle=True)
#         plt.plot(data[:, 0], data[:, 1], 'r--', label=f'Robot')
#         plt.axis('equal')
#         plt.legend()
#         plt.grid()
#         plt.show()
        
#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()