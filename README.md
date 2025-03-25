
🎟️ 3D Container Packing Optimization with Q-Learning
This project implements a 3D container packing optimization algorithm using Q-Learning, a reinforcement learning technique. The goal is to efficiently pack packages of varying sizes into a container, maximizing the fill ratio while adhering to physical constraints (e.g., packages must be placed from bottom to top and should fill the container from the center outward). The project includes a 3D visualization using OpenGL for real-time rendering of the packing process.

✨ Features
Q-Learning Algorithm: Utilizes reinforcement learning to learn an optimal policy for placing packages in a 3D container.

3D Visualization: Displays the container and packages in a 3D environment using OpenGL (via PyOpenGL and Pygame).

Customizable Package Data: Load package dimensions and counts from a CSV file or use default data.

Performance Optimization: Includes optimizations to reduce lag, such as fast placement checks and display list rendering.

Flexible Options: Supports various command-line arguments to customize the training and visualization process.

Physical Constraints: Ensures packages are placed from bottom to top and prioritizes filling the container from the center outward.

Model Persistence: Saves the best Q-table and packing configuration to a file for future use.

🛠️ Requirements
Software
Python 3.x: Ensure you have Python 3 installed.

Operating System: Compatible with Windows, macOS, and Linux.

Dependencies
Install the required Python libraries using pip:

pip install gym numpy pygame PyOpenGL
gym: For creating the reinforcement learning environment.

numpy: For numerical computations and array operations.

pygame: For handling the graphical interface and user input.

PyOpenGL: For 3D rendering of the container and packages.

Hardware
A system with a decent CPU for running the Q-Learning algorithm.

A GPU is recommended for smooth 3D rendering, though not strictly required.

💻 Installation
Clone this repository or download the source code as a ZIP file and extract it.

Install dependencies:

pip install -r requirements.txt
If a requirements.txt file is not provided, manually install the dependencies listed above.

Prepare package data (optional):

The program can load package data from a CSV file (packages.csv). If this file is not provided, default package data will be used.

Example packages.csv format:

name,length,width,height,count
small,2,2,1,5
medium,4,3,3,5
large,6,4,3,5
Place the packages.csv file in the same directory as the script.

⚡ Usage
Running the Program
Run the script using Python with the following command:

python packing.py
Command-Line Options
Option	Description
--episodes <int>	Number of training episodes (default: 20).
--alpha <float>	Learning rate for Q-Learning (default: 0.1).
--gamma <float>	Discount factor for Q-Learning (default: 0.99).
--epsilon-start <float>	Initial epsilon value for exploration (default: 1.0).
--epsilon-end <float>	Final epsilon value after decay (default: 0.01).
--epsilon-decay <float>	Epsilon decay rate per episode (default: 0.95).
--no-graphics	Run in non-graphic mode (disables 3D visualization).
--fullscreen	Run the visualization in fullscreen mode.
--auto-train	Automatically train without requiring user input.
--fast-mode	Enable fast mode to reduce rendering quality and improve performance.
--eval-only	Evaluate the best saved model without training.
--reset-model	Delete the saved model and start training from scratch.
--packages-csv <path>	Path to the CSV file containing package data.
--max-state-size <int>	Maximum size of the state space for Q-Learning (default: 10000).
Example Commands
Run with default settings:

python packing.py
Run with 50 episodes, auto-training, and fast mode:

python packing.py --episodes 50 --auto-train --fast-mode
Evaluate the best saved model:

python packing.py --eval-only
🛰 Controls (3D Visualization)
Left Mouse Button: Rotate the 3D model.

Mouse Wheel: Zoom in/out.

R Key: Reset the camera view.

A Key: Toggle auto mode.

Space/N Key: Proceed to the next step.

Esc Key: Exit the program.

⚙ How It Works
Container and Package Setup
The container has fixed dimensions: 200 cm x 100 cm x 150 cm.

Packages are loaded from a CSV file or default values.

Q-Learning Algorithm
State: The remaining count of each package type.

Action: Selecting a package type to place.

Reward:

Volume of the placed package.

Increase in fill ratio.

Penalty for empty space.

Novelty bonus for unique placements.

Packing Strategy:

Packages are placed bottom-up.

Prioritizes filling the container from the center outward.

Encourages tight packing by maximizing touching faces.

3D Visualization
The container and packages are rendered in 3D using OpenGL.

Packages are displayed with distinct colors.

Optimized rendering for smoother performance.

Output
Console: Training progress, fill ratio, placed packages.

3D View: Real-time rendering of the packing process.

Model Persistence: Saves the best Q-table (best_q_model.pkl).

🎧 Troubleshooting
Common Issues
Slow Performance: Use --fast-mode or --no-graphics.

Packages Not Fitting Well: Adjust package dimensions and increase episodes.

OpenGL Errors: Ensure PyOpenGL is installed.

CSV File Errors: Check format and ensure it's placed in the correct directory.

Debugging
Enable logging in the script.

Use --eval-only to inspect the saved model.

🌟 Future Improvements
Multithreading: Separate Q-Learning and rendering.

Advanced Heuristics: Implement rotation and better gap-filling logic.

Custom Rendering Frequency: Reduce rendering overhead.

📚 License
This project is licensed under the MIT License.

👨‍💻 Contributing
Contributions are welcome! Open an issue or submit a pull request.

📢 Contact
For support, open an issue or contact the maintainers.


