import torch
import torchvision.transforms as transforms
import numpy as np

import environment
from agent import Agent


class Hw3Env(environment.BaseEnv):
    def __init__(self, render_mode="offscreen", fast_mode=False) -> None:
        super().__init__(render_mode=render_mode)
        self._delta = 0.05
        self._goal_thresh = 0.075  # easier goal detection
        self._max_timesteps = 300  # allow more steps
        self._prev_obj_pos = None  # track object movement
        self._fast_mode = fast_mode  # Track if we're in fast mode
        
        # Adjust simulation parameters for fast mode
        if fast_mode:
            self._max_timesteps = 100  # Shorter episodes
            print("Running in fast mode with simplified physics")
    
    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene
    
    def reset(self):
        super().reset()
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()  # initialize previous position
        self._t = 0

        try:
            return self.high_level_state()
        except:
            return None

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    # def reward(self):
    #     state = self.high_level_state()
    #     ee_pos = state[:2]
    #     obj_pos = state[2:4]
    #     goal_pos = state[4:6]
    #     ee_to_obj = max(10*np.linalg.norm(ee_pos - obj_pos), 1)
    #     obj_to_goal = max(10*np.linalg.norm(obj_pos - goal_pos), 1)
    #     goal_reward = 100 if self.is_terminal() else 0
    #     return 1/(ee_to_obj) + 1/(obj_to_goal) + goal_reward

    def reward(self):
        
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # More informative reward structure:
        # 1. Reward for being close to the object (more shaped)
        r_ee_to_obj = 0.5 * np.exp(-5.0 * d_ee_to_obj)  # Exponential reward for closeness
        
        # 2. Reward for moving object closer to goal
        r_obj_to_goal = 0.5 * np.exp(-5.0 * d_obj_to_goal)
        
        # 3. Reward for pushing in the right direction (more weight)
        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        
        # Only reward movement when it's significant
        if np.linalg.norm(obj_movement) > 1e-4:
            movement_alignment = np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal)
            r_direction = 1.0 * max(0, movement_alignment)  # Increased weight
        else:
            r_direction = 0.0

        # 4. Higher terminal reward
        r_terminal = 20.0 if self.is_terminal() else 0.0
        
        # 5. Smaller step penalty
        r_step = -0.05
        
        # Save object position for next step
        self._prev_obj_pos = obj_pos.copy()
        
        # Total reward
        reward = r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step
        
        return reward

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps
    
    def step(self, action):
        action = action.clamp(-1, 1).cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)            

        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        if result:  # If the action is successful
            truncated = self.is_truncated()
        else:  # If didn't realize the action
            truncated = True
        return state, reward, terminal, truncated

    def _set_ee_in_cartesian(self, position, rotation=None, max_iters=10000, threshold=0.04, n_splits=30):
        # In fast mode, reduce the number of splits for faster simulation
        if self._fast_mode:
            n_splits = max(5, n_splits // 3)  # Reduce trajectory points
            max_iters = max_iters // 2  # Reduce max iterations
            
        # Call the parent method with adjusted parameters
        return super()._set_ee_in_cartesian(position, rotation, max_iters, threshold, n_splits)


if __name__ == "__main__":
    env = Hw3Env(render_mode="offscreen")
    agent = Agent()
    num_episodes = 100

    rews = []

    for i in range(num_episodes):        
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            agent.add_reward(reward)
            cumulative_reward += reward
            done = is_terminal or is_truncated
            
            state = next_state
            episode_steps += 1

        print(f"Episode={i}, reward={cumulative_reward}")
        rews.append(cumulative_reward)
        agent.update_model()

    ## Save the model and the training statistics
    torch.save(agent.model.state_dict(), "model.pt")
    np.save("rews.npy", np.array(rews))
        

