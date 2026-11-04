## Imitation-learned pick in a Restricted Cabinet



https://github.com/user-attachments/assets/d6d03c50-84f1-49f8-a2bb-018cb3b780a6



A Piper 6-DoF arm reaches into a cabinet it has never seen, grasps a cube on the deck
and withdraws it through the sash. The ACT policy was trained only on the fridge-shelf
pick task and was **not** fine-tuned — cube spawns were constrained to lie inside the
training envelope measured in the robot's base frame, so cabinet geometry, lighting and
robot yaw all differ from training while the target's pose relative to the arm does not.


### Training data — ACT camera views

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f4a26d9-63ea-4b07-8df0-ef7a4a67645d" width="400">
  <img src="https://github.com/user-attachments/assets/bc43d98d-0e21-4046-ab64-75289ad08cab" width="400">
</p>
<p align="center"><em>Right: wrist camera (<code>gripcam</code>). Left: external camera (<code>povcam</code>).</em></p>



