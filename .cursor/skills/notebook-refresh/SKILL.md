---
name: notebook-refresh
description: Clears and re-runs Jupyter notebooks using the project's uv environment. Use when the user wants to validate, refresh, or "clean and run" notebooks.
---

# Notebook Refresh Skill

## Instructions

1. **Determine Scope**: Decide which notebooks to process based on the user's request:
   - **All notebooks**: Find every `.ipynb` file in the specified directory.
   - **Changed notebooks only**: If the user asks to refresh only changed/modified notebooks, use git to identify them:
     - `git diff --name-only HEAD -- '*.ipynb'` for uncommitted changes vs the last commit.
     - `git diff --name-only main...HEAD -- '*.ipynb'` for all changes on the current branch vs `main` (adjust base branch name if needed).
     - If no notebooks have changed, report that and stop.
2. **Prerequisites**: Ensure the `uv` environment has `nbconvert` and `ipykernel` installed. If not, prompt to run `uv add --dev nbconvert ipykernel`.
3. **Execution**: For each notebook, run the following commands from the project root:
   - **Step 1 (Clear)**: `uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <path_to_notebook>.ipynb`
   - **Step 2 (Run)**: `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <path_to_notebook>.ipynb`
4. **Validation**:
   - If a notebook fails, capture the error message from the terminal output.
   - Report exactly which cell or module caused the failure.
5. **Final Report**: Provide a summary table of results (Success/Failure) for all processed files, noting whether the scope was "all" or "changed only".
