# Genetic Algorithms


[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

For Introduction to Artificial Intelligence
by AI Group 2


## Emergency Unit

### 1. Introduction
The Emergency Unit project demonstrates the use of a Genetic Algorithm (GA) to optimize the placement of an emergency response unit (such as an ambulance or fire station) within a city grid. The goal is to minimize the average response time to emergencies, taking into account the frequency and distribution of incidents across different city sections. This project provides an interactive visualization using Streamlit, allowing users to experiment with GA parameters and observe the optimization process in real time.

### 2. How code works
- The core logic is implemented in `emergency_unit/emergency.py` as the `EmergencyUnitGA` class.
- The city is modeled as a grid (default 10x10), with each cell representing a city section and having a simulated emergency frequency.
- The genetic algorithm evolves a population of candidate locations (coordinates) for the emergency unit:
  - **Initialization:** Random locations are generated as the initial population.
  - **Fitness Evaluation:** Each location is evaluated using a cost function that sums the weighted response times to all city sections, weighted by emergency frequency.
  - **Selection:** Tournament selection is used to choose parents for the next generation.
  - **Crossover:** Arithmetic crossover combines parent coordinates to produce offspring.
  - **Mutation:** Gaussian mutation introduces small random changes to offspring locations.
  - **Elitism:** The best solution from each generation is preserved.
- The process repeats for a specified number of generations, tracking the best solution and cost at each step.
- The Streamlit app (`emergency_unit/main.py`) provides a user interface to set GA parameters, run the optimization, and visualize results (evolution chart, city layout, emergency hotspots, and generation table).

### 3. Requirements
- Python 3.x
- Required packages:
  - `numpy`
  - `pandas`
  - `streamlit`
  - `plotly`
- To install dependencies, run:
  ```sh
  pip install -r requirements.txt
  ```
  or, if using `uv`:
  ```sh
  uv pip install -r requirements.txt
  ```
- To launch the app:
  ```sh
  streamlit run emergency_unit/main.py
  ```

## Pinoy Henyo

### 1. Introduction

### 2. How code works

### 3. Requirements

## Travelling Salesman

### 1. Introduction
The Travelling Salesman Problem (TSP) project uses a Genetic Algorithm (GA) to find a near-optimal route for a salesman to visit a set of cities exactly once and return to the starting point, minimizing the total travel distance. This project includes a visualizer built with Pygame, allowing users to step through the evolution of solutions and observe how the GA improves the route over generations.

### 2. How code works
- The main logic is implemented in `tsp/genetic_algorithm.py` and `tsp/tsp.py`.
- **Cities and Distance Matrix:** Cities are defined as coordinates, and a distance matrix is computed for all city pairs.
- **Genetic Algorithm:**
  - **Chromosome Representation:** Each chromosome represents a possible route (permutation of city indices).
  - **Initialization:** The population is initialized with random routes.
  - **Fitness Evaluation:** Fitness is the inverse of the total route distance (shorter routes are fitter).
  - **Selection:** Tournament selection is used to choose parents.
  - **Crossover:** Order Crossover (OX) combines parent routes to produce valid offspring.
  - **Mutation:** Swap mutation randomly swaps two cities in a route.
  - **Elitism:** The best individual is always carried over to the next generation.
  - **Stagnation Detection:** The algorithm stops early if no improvement is seen for a set number of generations.
- **Visualization:**
  - The `TSPVisualizer` class in `tsp/tsp.py` uses Pygame to display the cities, current route, and route sequence.
  - Users can step through generations using the SPACE key and reset with R.
  - The best route, distance, and generation number are displayed on the screen.

### 3. Requirements
- Python 3.x
- Required packages:
  - `numpy`
  - `pygame`
  - `matplotlib`
- To install dependencies, run:
  ```sh
  pip install -r requirements.txt
  ```
  or, if using `uv`:
  ```sh
  uv pip install -r requirements.txt
  ```
- To run the visualizer:
  ```sh
  python tsp/tsp.py
  ```