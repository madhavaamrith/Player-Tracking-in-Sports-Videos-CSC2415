@echo off
REM Assumes you've already created & activated a venv at least once.
REM If not, do:
REM   python -m venv .venv
REM   .\.venv\Scripts\Activate
REM   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
REM   pip install -r requirements.txt

python -m src.pipeline --input_dir data/videos --out_dir outputs --device auto --conf 0.25
pause
