import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor


class CurveFittingEnv(gym.Env):
    """
    Custom Environment for solving y = a * sin(b * x) + c using RL.
    """
    def __init__(self):
        super(CurveFittingEnv, self).__init__()
        
        # True parameters we want the agent to find
        self.true_a = 2.5
        self.true_b = 1.3
        self.true_c = 0.8
        
        # Generate synthetic training data
        self.x_data = np.linspace(-5, 5, 100)
        self.y_true = self.true_a * np.sin(self.true_b * self.x_data) + self.true_c
        
        # Action space: The agent can change a, b, and c by a value between -0.5 and 0.5
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)
        
        # Observation space: [current_a, current_b, current_c, current_mse]
        # We set broad limits for the observation space
        self.observation_space = spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)
        
        self.max_steps = 50
        self.current_step = 0
        
    def _get_mse(self, a, b, c):
        y_pred = a * np.sin(b * self.x_data) + c
        return np.mean((self.y_true - y_pred) ** 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize with a random bad guess
        self.current_params = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        self.current_mse = self._get_mse(*self.current_params)
        
        obs = np.array([*self.current_params, self.current_mse], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Apply the agent's action (update parameters)
        self.current_params += action
        
        # 2. Calculate new error
        new_mse = self._get_mse(*self.current_params)
        
        # 3. Calculate Reward: Positive if MSE decreases, negative if it increases
        # We scale it by 10 to give the neural network stronger gradients
        reward = 10.0 * (self.current_mse - new_mse)
        
        self.current_mse = new_mse
        
        # 4. Check Termination conditions
        terminated = False
        truncated = False
        
        if self.current_mse < 0.01:
            terminated = True
            reward += 50.0  # Massive bonus for solving the problem!
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = np.array([*self.current_params, self.current_mse], dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}

# ==========================================
# Training and Testing the Agent
# ==========================================
if __name__ == "__main__":
    print("Initializing Environment...")
 #   env = CurveFittingEnv()
    
 #   print("Building PPO Agent...")
 #   # PPO is a robust policy-gradient algorithm perfect for continuous actions
 #   model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    
 #   print("Training Agent (this will take a few seconds)...")
 #   # Train for 30,000 steps. In a real complex scenario, this might be 1M+ steps.
 #   model.learn(total_timesteps=30000)
   
    # 2nd version starts 
    # 1. Wrap the environment in a Monitor so the callback can read the episode rewards
    env = Monitor(CurveFittingEnv())

    # 2. Define the stopping condition
    # Our environment gives a +50 bonus for solving it. A well-trained agent 
    # should easily score > 50 per episode.
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=55.0, verbose=1)

    # 3. Define the Evaluation Callback
    # This will test the agent every 5,000 steps. If it hits the reward threshold, it stops.
    eval_callback = EvalCallback(
        env, 
        callback_on_new_best=stop_callback, 
        eval_freq=5000, 
        best_model_save_path='./logs/', 
        verbose=1
    )

    print("Building PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0005) # Slightly lower LR for stability

    print("Training Agent with Monitoring...")
    # Increased timesteps, but it will stop early if it solves the problem
    model.learn(total_timesteps=500000, callback=eval_callback)
    print("Training Complete!\n")

    # Load the best model found during monitoring
    model = PPO.load('./logs/best_model.zip')  
   
    # 2nd version ends 

    print("Training Complete!\n")
    
    # --- Let's test the trained agent ---
    print("--- Testing the Agent ---")
    obs, info = env.reset()
    print(f"Initial Random Guess -> a: {obs[0]:.2f}, b: {obs[1]:.2f}, c: {obs[2]:.2f} | Initial MSE: {obs[3]:.2f}")
    print(f"Target Parameters    -> a: 2.50, b: 1.30, c: 0.80\n")
    
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    
    while not (terminated or truncated):
        # The agent predicts the best parameter adjustments based on current state
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    print(f"Finished in {steps} steps.")
    print(f"Final Agent Guess    -> a: {obs[0]:.2f}, b: {obs[1]:.2f}, c: {obs[2]:.2f}")
    print(f"Final MSE            -> {obs[3]:.4f}")


# output will be something like below.
#--- Testing the Agent ---
#Initial Random Guess -> a: -0.43, b: 0.92, c: 0.38 | Initial MSE: 3.88
#Target Parameters    -> a: 2.50, b: 1.30, c: 0.80
#
#Finished in 8 steps.
#Final Agent Guess    -> a: 2.41, b: 1.31, c: 0.76
#Final MSE            -> 0.0085
