\usepackage{graphicx}
\usepackage{hyperref}

\begin{figure}[htbp]
  \centering
  \href{run:media/Immitationact.mp4}{%
    \includegraphics[width=0.85\linewidth]{media/Immitationact_poster.png}}
  \caption{\textbf{Imitation-learned pick in a laminar-flow cabinet.} A Piper 6-DoF arm
  reaches into a cabinet it has never seen, grasps a cube on the deck and withdraws it
  through the sash. The ACT policy was trained only on the fridge-shelf pick task and
  was not fine-tuned; cube spawns were constrained to lie inside the training envelope
  measured in the robot's base frame, so cabinet geometry, lighting and robot yaw all
  differ from training while the target's pose relative to the arm does not. Across the
  three episodes shown ($139$\,s), the arm grasped and lifted the cube every time
  ($+8.7$, $+7.6$, $+8.5$\,cm) and cleared the cabinet mouth without contacting the
  glass, but released it during withdrawal in two of three. Grasp acquisition
  transfers; grasp retention does not.
  \href{run:media/Immitationact.mp4}{(click to play)}}
  \label{fig:hood-rollout}
\end{figure}
