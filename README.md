# Age Prediction Project

Fresh repository initialization. Previous git history and large dependency folders (venv, node_modules) removed.

## Structure
- back-end: Python FastAPI (or similar) service and ML model file `model.keras`.
- front-end: React (Vite) application.

## After Cloning / Fresh Start
1. Create Python virtual environment inside `back-end` (do NOT commit it):
   ```powershell
   cd back-end
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Install front-end dependencies:
   ```powershell
   cd ..\front-end
   npm install
   ```
3. Run back-end (example):
   ```powershell
   cd ..\back-end
   uvicorn main:app --reload
   ```
4. Run front-end:
   ```powershell
   cd ..\front-end
   npm run dev
   ```

## Notes
- Large model file (`back-end/model.keras`, ~66MB) is under GitHub's 100MB hard limit but over the 50MB warning threshold. Consider Git LFS if it grows further.
- Keep `venv/` and `node_modules/` out of git.

## Reinitialization Performed
This README was added during repository reset.
