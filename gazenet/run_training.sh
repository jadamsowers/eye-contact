#!/bin/zsh
# Trains (or resumes) the gaze model, then exports Core ML and renders a test clip.
# Runs unattended: nohup gazenet/run_training.sh runs/gazenet_v2 > runs/gazenet_v2.log 2>&1 &
set -euo pipefail
cd "${0:A:h}/.."
RUN=${1:-runs/gazenet_v2}
RESUME=()
[[ -f $RUN/last.pt ]] && RESUME=(--resume $RUN/last.pt)

caffeinate -i -s .venv-train/bin/python gazenet/train.py data/columbia_patches.npz $RUN --steps 15000 $RESUME

.venv-train/bin/python gazenet/export_coreml.py $RUN/best.pt $RUN/GazeWarp.mlpackage
for clip in d3 d6; do
  .venv/bin/python gazenet/apply_video.py $RUN/best.pt LivePortrait/assets/examples/driving/$clip.mp4 $RUN/${clip}_fixed.mov
  ffmpeg -v error -y -i LivePortrait/assets/examples/driving/$clip.mp4 -i $RUN/${clip}_fixed.mov \
    -filter_complex "[0][1]hstack" -c:v libx264 -crf 16 -pix_fmt yuv420p $RUN/${clip}_compare.mp4
done
echo "ALL DONE"
