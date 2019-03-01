# define task

import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
    
    def get_reward(self):        
        # try implementing eculidean distance to get reward
        cur_x, cur_y, cur_z = self.sim.pose[:3]
        tar_x, tar_y, tar_z = self.target_pos
                
        reward = 0
        penalties = 0
        
        # calculate differences from target  
        x_distance = abs((cur_x - tar_x)) ** 2
        y_distance = abs((cur_y - tar_y)) ** 2
        z_distance = abs((cur_z - tar_z)) ** 2        
        
        # distance from target
        distance = np.sqrt(x_distance + y_distance + z_distance)
        
        # calculate reasonable proximity from target, 
        # start with 30% region around target (assumes starting from origin)
        target = abs(tar_x) + abs(tar_y) + abs(tar_z)         
        proximity = target * 0.30
         
        if (distance < proximity):
            # find % distance from position relative to proximity       
            proportion = 1 - min((distance / proximity), 1.0)
            # reward becomes greater as the agent approaches target
            reward += (200 * proportion)

        # penalise large velocities, a smooth flight will produce better results    
        x_vel, y_vel, v_vel = self.sim.v
        if(abs(x_vel) > 12.0 or abs(y_vel) > 12.0 or abs(v_vel) > 12.0):
            penalties += 20
        
        return (reward - penalties)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state