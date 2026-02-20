---
name: notebook-refresh
description: Clears and re-runs Jupyter notebooks using the project's uv environment. Use when the user wants to validate, refresh, or "clean and run" notebooks.
---

# Notebook Refresh Skill

## Model Preference

- **IMPORTANT**: This is a high-token/repetitive task.
- **DO NOT** use Claude Opus or expensive reasoning models for this task.
- **PREFER**: Claude 3.5/3.7 Sonnet or "Auto" mode to save credits.
- If the user has Claude Opus selected, ask for confirmation or suggest switching to a cheaper model before proceeding.

## Instructions

1. **Locate Notebooks**: Find the target `.ipynb` files in the specified directory.
2. **Prerequisites**: Ensure the `uv` environment has `nbconvert` and `ipykernel` installed. If not, prompt to run `uv add --dev nbconvert ipykernel`.
3. **Execution**: For each notebook, run the following commands from the project root:
   - **Step 1 (Clear)**: `uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <path_to_notebook>.ipynb`
   - **Step 2 (Run)**: `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <path_to_notebook>.ipynb`
4. **Validation**:
   - If a notebook fails, capture the error message from the terminal output.
   - Report exactly which cell or module caused the failure.
5. **Final Report**: Provide a summary table of results (Success/Failure) for all processed files.
