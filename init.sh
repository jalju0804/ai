sudo snap install astral-uv --classic
uv venv --python 3.9
source .venv/bin/activate
uv sync
cd ultralytics/yolo/v8/detect/