# Reinforcement Learning Based Path Planning on Mobile Robots

- So this project is basically something I do in my off time. Started off by trying to learn DRL based path planning on robots.
- Picked a 4 wheeled omni bot for this project, created an custom gym env after the entire kinematics model was modularized for this bot.
- Chose a simple 2D env with minimal physics info; the only physics based info included in the environment is that of the kinematics of the bot hence, 4-wheeled-Omni-Bot.

## Tech Stack
  - Python 3.10 (lang)
  - StableBaselines3 (RL Framework)
  - StableBaselines3-Imitation (Imitation Learning Framework) --> will upgrade to robomimic (thought it was better)
  - Pygame (for the animation and environment creation)
  - Gymnasium (for uniformizing the env apis)
    
## Kinematics 4-wheeled-Omni-Drive
 - This particular wheel config was documented in an isaacsim based simulation for autonomizing a similar bot in good physics based enviroment previously, so just adding the diagrams and equations of the model.
 - A module was created to consolidate the omni-bot model with kinematics; currently working on adding dynamics too to add effects of friction and damping of motor (simulated motors) to make it better, or just to basically satiatiate my distractions. The module is [here]().
   <div>
     <img src="" alt="frame-diagram"/>
   </div>





**Note**: I haven't added comments to my codebase just yet, atleast an organized version. The project is basically done in my freetime for learning purposes so its a little rough around the edges and its in python 3.11.0.






## ========================================================================================================
## An RL developer module specifically for 4wheeled-OmniBot using python (just for fun 😐)  --> started off as a combined offshoot of a another project involving standard robotics path planning involving the normal search-based and probabilistic based path planning. Thought of testing RL on the side and now keep a separate repo for tracking and saving progress right here

- Here is a working demo of a sample multi-bot environment 

<div align="center">
  <img src="https://github.com/user-attachments/assets/a90cd675-712b-48ba-964b-8ab58b674637" alt="multi-bot-feature-test-demo">
</div>

- The module is still subject to changes and is a bit unpolished
