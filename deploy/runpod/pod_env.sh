# Sourced (not executed) by bootstrap.sh and serve.sh on the pod.
# Everything that must survive a pod stop/start lives on the /workspace volume:
# the container disk (/, /root, /usr) is wiped on stop, the volume is not.
export FRC_WS="${FRC_WS:-/workspace}"
export FRC_APP="${FRC_APP:-$FRC_WS/frc}"
export FRC_LOGS="$FRC_WS/logs"
export FRC_TOOLS="$FRC_WS/tools"
export FRC_CACHE="$FRC_WS/cache"
export FRC_TUNE="$FRC_WS/tune"

# Model/download caches on the volume so a restart never re-downloads.
export HF_HOME="$FRC_CACHE/hf"
export TORCH_HOME="$FRC_CACHE/torch"
export EASYOCR_MODULE_PATH="$FRC_CACHE/easyocr"
export YOLO_CONFIG_DIR="$FRC_CACHE/ultralytics"
export MPLCONFIGDIR="$FRC_CACHE/matplotlib"
export UV_CACHE_DIR="$FRC_CACHE/uv"
export UV_PYTHON_INSTALL_DIR="$FRC_TOOLS/python"
export UV_LINK_MODE=copy
# Ultralytics must never try to pip-install anything at runtime.
export YOLO_OFFLINE=1

# uv + deno (yt-dlp's JS runtime for 720p/1080p YouTube formats) live on the
# volume; web/app.py also looks in ~/.deno/bin, but PATH works everywhere.
export PATH="$FRC_TOOLS/bin:$FRC_TOOLS/deno/bin:$PATH"
export PYTHONUNBUFFERED=1
