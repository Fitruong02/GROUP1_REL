# 📦 3D Container Packing Optimization with Q-Learning

This project implements a **3D container packing optimization algorithm** using **Q-Learning**, a reinforcement learning technique. The goal is to efficiently pack packages of varying sizes into a container, maximizing the fill ratio while adhering to physical constraints (e.g., packages must be placed from bottom to top and should fill the container from the center outward). The project includes a **3D visualization** using OpenGL for real-time rendering of the packing process.

---

## ✨ Features

- **Q-Learning Algorithm**: Utilizes reinforcement learning to learn an optimal policy for placing packages in a 3D container.
- **3D Visualization**: Displays the container and packages in a 3D environment using OpenGL (via PyOpenGL and Pygame).
- **Customizable Package Data**: Load package dimensions and counts from a CSV file or use default data.
- **Performance Optimization**: Includes optimizations to reduce lag, such as fast placement checks and display list rendering.
- **Flexible Options**: Supports various command-line arguments to customize the training and visualization process.
- **Physical Constraints**: Ensures packages are placed from bottom to top and prioritizes filling the container from the center outward.
- **Model Persistence**: Saves the best Q-table and packing configuration to a file for future use.

---

## 🛠️ Requirements

### Software
- **Python 3.x**: Ensure you have Python 3 installed.
- **Operating System**: Compatible with Windows, macOS, and Linux.

### Dependencies
Install the required Python libraries using `pip`:

```bash
pip install gym numpy pygame PyOpenGL
```

- `gym`: For creating the reinforcement learning environment.
- `numpy`: For numerical computations and array operations.
- `pygame`: For handling the graphical interface and user input.
- `PyOpenGL`: For 3D rendering of the container and packages.

### Hardware
- A system with a decent CPU for running the Q-Learning algorithm.
- A GPU is recommended for smooth 3D rendering, though not strictly required.

---

## 🚀 Installation

### Clone or Download the Repository:
Clone this repository or download the source code as a ZIP file and extract it.

### Install Dependencies:
Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, manually install the dependencies listed above.

### Prepare Package Data (Optional):
The program can load package data from a CSV file (`packages.csv`). If this file is not provided, default package data will be used.

Example `packages.csv` format:

```csv
name,length,width,height,count
small,2,2,1,5
medium,4,3,3,5
large,6,4,3,5
```

Place the `packages.csv` file in the same directory as the script.

---

## 📌 Usage

### Running the Program
Run the script using Python with the following command:

```bash
python packing.py
```

### Command-Line Options
The program supports various command-line arguments to customize its behavior. Below is a list of available options:

```bash
--episodes <int>          # Number of training episodes (default: 20)
--alpha <float>          # Learning rate for Q-Learning (default: 0.1)
--gamma <float>          # Discount factor for Q-Learning (default: 0.99)
--epsilon-start <float>  # Initial epsilon value for exploration (default: 1.0)
--epsilon-end <float>    # Final epsilon value after decay (default: 0.01)
--epsilon-decay <float>  # Epsilon decay rate per episode (default: 0.95)
--no-graphics            # Run in non-graphic mode (disables 3D visualization)
--fullscreen             # Run the visualization in fullscreen mode
--auto-train             # Automatically train without requiring user input
--fast-mode              # Enable fast mode to improve performance
--eval-only              # Evaluate the best saved model without training
--reset-model            # Delete the saved model and start training from scratch
--packages-csv <path>    # Path to the CSV file containing package data
--max-state-size <int>   # Maximum size of the state space for Q-Learning (default: 10000)
```

### Example Commands

Run with default settings:

```bash
python packing.py
```

Run with 50 episodes, auto-training, and fast mode:

```bash
python packing.py --episodes 50 --auto-train --fast-mode
```

Run in non-graphic mode with a custom package CSV:

```bash
python packing.py --no-graphics --packages-csv my_packages.csv
```

Evaluate the best saved model:

```bash
python packing.py --eval-only
```

---

## 🎮 Controls (3D Visualization)

- **Left Mouse Button**: Rotate the 3D model.
- **Mouse Wheel**: Zoom in/out.
- **R Key**: Reset the camera view.
- **A Key**: Toggle auto mode.
- **Space/N Key**: Proceed to the next step.
- **Esc Key**: Exit the program.

---

## 📊 How It Works

### Container and Package Setup
- The container has fixed dimensions: 200 cm (length) x 100 cm (width) x 150 cm (height).
- Packages are loaded from a CSV file or use default values.
- The total volume is calculated to determine the theoretical maximum fill ratio.

### Q-Learning Algorithm
- **State**: Represented by the remaining count of each package type.
- **Action**: Selecting a package type to place in the container.
- **Reward**: Based on volume placement, fill ratio, penalties for empty space, and compact packing.

### Packing Strategy
- **Placement Rules**:
  - Bottom to top.
  - Center outward.
- **Optimization**:
  - Encourages compact packing using a scoring function.
  - Uses fast placement checks to reduce computation time.

### 3D Visualization
- The container and packages are rendered in 3D using OpenGL.
- Display lists optimize rendering performance.

---

## 📌 Troubleshooting

### Common Issues
- **Lag or Slow Performance**:
  - Use `--fast-mode`.
  - Reduce the container size.
  - Use `--no-graphics` to disable visualization.

- **Packages Not Filling the Container**:
  - Ensure package dimensions allow for a high fill ratio.
  - Increase `--episodes`.

- **OpenGL Errors**:
  - Ensure PyOpenGL is installed correctly.
  - Use `--no-graphics` if necessary.

---

## 🔥 Future Improvements
- **Multithreading**: Separate Q-Learning and rendering into different threads.
- **Advanced Heuristics**: Rotate packages and improve space utilization.
- **Customizable Rendering Frequency**: Control how often 3D visualization updates.

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing
Contributions are welcome! Please submit a pull request or open an issue.

---

## 📬 Contact
For questions or support, please open an issue on the repository or contact the maintainers directly.
