# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

From-scratch implementation of a Large Language Model (LLM) using Jupyter notebooks. The project progressively builds all fundamental neural network components without using ML frameworks.

## Development Commands

### Running Jupyter Notebook
```bash
uv tool run jupyter notebook
```

### Package Management
Uses `uv` package manager:
- Add dependency: `uv add <package-name>`
- Install dependencies: `uv sync`
- Python version: 3.12
- Virtual environment: `.venv/` (activate with `.venv/Scripts/activate` on Windows)

## Architecture

### Progressive Implementation

Each notebook builds upon previous concepts. Current and planned components:

1. **Autograd System & Backpropagation** (`grad.ipynb`) - Current


**Operations**: Addition and multiplication

**Visualization**:
- `trace(root)`: Traverse computational graph
- `draw_dot(root)`: GraphViz visualization showing values, gradients, and operations

**GraphViz Setup** (Windows):
```python
graphviz_path = r"c:\\Program Files\\Graphviz\\bin"
```

## Key Concepts

1. **Computational Graph**: Tracks computation history through `_prev`
2. **Gradient Tracking**: `grad` attribute for backpropagation
3. **From-Scratch Philosophy**: No PyTorch, TensorFlow, or other ML frameworks
4. **Visual Debugging**: GraphViz integration for understanding graph structure

## Dependencies

- **graphviz**: Computational graph visualization
- **ipykernel**: Jupyter notebook kernel
- **matplotlib**: Training curves and data visualization
- **numpy**: Numerical computing

## Development Notes

- GraphViz must be installed at `c:\Program Files\Graphviz\bin` on Windows
- All development happens in Jupyter notebooks (not `main.py`)
- Maintain progressive learning approach: simple → complex
- Comment code to explain mathematical concepts
- Keep notebooks self-contained with clear visualizations
