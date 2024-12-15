# Chess Arm Coach

We are thrilled to finally submit version 0 of our **Chess Arm Coach** project, developed by the **Rook and Rule** team! This project combines computer vision, machine learning, IoT, and chess expertise to create an educational chess robot arm that detects moves, provides feedback, and helps players improve their skills.

## Project Overview

The Chess Arm Coach project aims to:
1. Detect chessboards and identify the pieces and their positions.
2. Provide real-time feedback and ratings on moves during a game.
3. Control a robotic arm to play moves and interact with players.
4. Serve as an educational tool by analyzing games and offering tailored learning experiences.

Some of the code files in this repository are theoretical and under development, serving as a description of ideas rather than functional implementations.

---

## File Descriptions

### Core Functionality

- **`board_detection.py`**:
  - Uses OpenCV to detect the chessboard and crop the image into a square containing 64 smaller squares.
  - Implements the `get_squares()` function to prepare the board for further analysis.

- **`get_position.py`**:
  - Uses the models in the `models` folder to predict the color and type of each chess piece.
  - Includes the `get_fen_from_board` function to generate the FEN (Forsyth-Edwards Notation) formula representing the board state.

- **`image_to_fen.ipynb`**:
  - Demonstrates the process of cropping images, obtaining the FEN formula, and visualizing the board.
  - Explains how this formula is used for virtualizing the board in the Python code.

- **`comments.py`**:
  - Provides real-time feedback and move ratings based on the board state and moves played during games.

- **`GUIPlayer.py`**:
  - Implements a graphical user interface for visualization and user interaction.

- **`code_Arm.py`**:
  - Handles the IoT components of the robotic arm.
  - Calculates inverse kinematics and provides functions for controlling the arm's components.

### Supplementary Codes

- **`SLO.ipynb`** (Student Learning Outcomes):
  - Located in the `models_based_on_previous_games` folder.
  - Analyzes players' previous games to build models that aid in teaching and improvement.

- **`cluster.ipynb`**:
  - Explores clustering techniques to minimize the number of LSTM models by grouping players into clusters.
  - Proposes creating one LSTM model per cluster rather than individual models for each player.

### Real-Time Game Files

1. **`game_via_camera.py`**:
   - Captures moves from camera images to analyze board states in real time.

2. **`arm_vs_arm.py`**:
   - Simulates games between two robotic arms using a chess engine, providing a convenient way to collect game data.

3. **`ChessArmCoach_V0.ipynb`**:
   - Describes the full functionality of the chess arm coach.
   - Includes board detection via camera, real-time feedback provided through a speaker, and arm movement.

---

## Repository Structure

```
ChessArmCoach/
|-- board_detection.py
|-- get_position.py
|-- image_to_fen.ipynb
|-- comments.py
|-- GUIPlayer.py
|-- code_Arm.py
|-- SLO.ipynb
|-- cluster.ipynb
|-- game_via_camera.py
|-- arm_vs_arm.py
|-- ChessArmCoach_V0.ipynb
|-- models/
    |-- [piece type prediction models]
    |-- [color prediction models]
|-- models_based_on_previous_games/
    |-- SLO.ipynb
|-- data/
    |-- [training and game data]
```

---

## Notes

- Some parts of the code are still under development or theoretical. These are meant to describe the concepts and ideas we aim to implement in future versions.
- Contributions and feedback are welcome to improve the functionality and robustness of this project.

---

## Future Work

1. Enhance the clustering techniques for player analysis.
2. Improve real-time move detection and feedback mechanisms.
3. Optimize the robotic arm's movement for faster and smoother gameplay.
4. Expand the teaching models for more personalized learning experiences.

---
https://zenodo.org/records/6656212 :the link of pieces data set that we transform it to 4 folders in the models folder color train - color valid piece train - piece valid

https://stockfishchess.org/download/ link where to donwlowd stockfish enging

https://drive.google.com/drive/u/0/folders/1OYTOxkC-sxIyRHNxlIBy591P4MafCznS compressed models folder

versions:
Python 3.12.2
chess 0.31.3
pytorch 2.4.1
keras 3.4.1

