#!/bin/zsh
# v3: trained on Vision-framed patches (the plug-in's tracker), then exported and tested with
# the Swift pipeline. Unattended: nohup gazenet/run_v3.sh > runs/gazenet_v3.log 2>&1 &
set -euo pipefail
cd "${0:A:h}/.."
RUN=runs/gazenet_v3
RESUME=()
[[ -f $RUN/last.pt ]] && RESUME=(--resume $RUN/last.pt)

caffeinate -i -s .venv-train/bin/python gazenet/train.py data/columbia_vision_patches.npz $RUN --steps 15000 $RESUME
.venv-train/bin/python gazenet/export_coreml.py $RUN/best.pt $RUN/GazeWarp.mlpackage
for clip in d3 d6; do
  EyeContactKit/.build/release/eyecontact LivePortrait/assets/examples/driving/$clip.mp4 $RUN/${clip}_fixed.mov \
    --model $RUN/GazeWarp.mlpackage
  ffmpeg -v error -y -i LivePortrait/assets/examples/driving/$clip.mp4 -i $RUN/${clip}_fixed.mov \
    -filter_complex "[0][1]hstack" -c:v libx264 -crf 16 -pix_fmt yuv420p $RUN/${clip}_compare.mp4
done
echo "ALL DONE"
