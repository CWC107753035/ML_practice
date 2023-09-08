import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
final_reward = 0
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   final_reward += reward
   if terminated or truncated:
      print(final_reward)
      final_reward = 0
      observation, info = env.reset()