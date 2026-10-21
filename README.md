# Eye Contact

Personal eye-contact correction for Final Cut Pro.

## Layout

- `EyeContactKit/`: the Swift engine, shared by the CLI and the FxPlug plug-in
  - `EyeContactCore/EyeTracker.swift`: Vision face landmarks → eye corners, lid curves, pupil (refined by a dark-pixel centroid)
  - `EyeContactCore/EyeGeometry.swift`: gaze offset, correction settings, fades for look-aways, blinks and head turns
  - `EyeContactCore/Smoothing.swift`: One Euro temporal smoothing
  - `EyeContactCore/EyeWarpRenderer.swift`: Metal fragment shader. The iris slides rigidly behind the lids and the sclera stretches or compresses around it
  - `EyeContactCore/TrackingCache.swift`: time-keyed measurement cache with Gaussian smoothing (used by the plug-in)
  - `eyecontact/`: CLI for testing on video files
- `EyeContactFx/`: Final Cut Pro plug-in (FxPlug 4) wrapping the engine
- `LivePortrait/`, `.venv/`: LivePortrait checkout, weights, and Python 3.10 env (for the later learned-model stage)
- `test/`: sample outputs (`compare_*.mp4` are original | corrected)

## CLI

```sh
cd EyeContactKit && swift build -c release
.build/release/eyecontact input.mov output.mov            # ProRes 422 out, audio passed through
.build/release/eyecontact input.mov out.mov --debug       # tracking overlay
.build/release/eyecontact input.mov out.mov --strength 0.7 --target-y 0.03 --log gaze.csv
```

Options: `--strength`, `--target-x/--target-y` (where "looking at camera" sits, in eye widths),
`--max` (look-aways bigger than this are left alone, default 0.18), `--no-smooth`, `--no-refine`, `--frames`.

About 5 ms per frame for tracking plus 0.5 ms for the warp on an M5 (tested on 480–512 px clips).

## Limits of the warp approach

- Works well for small to moderate corrections (up to about 0.18 eye widths). Larger shifts smear, which is why they fade out.
- Doesn't move the eyelids with the gaze (real eyes' upper lids follow vertical gaze).
- Assumes upright footage and the largest face in frame.
- The CLI works in 8-bit BGRA. The plug-in works in FCP's float pipeline.

## FxPlug plugin (`EyeContactFx/`)

The Xcode project is generated from `project.yml` with XcodeGen and mirrors Apple's FxPlug 4 template.

```sh
cd EyeContactFx && xcodegen
xcodebuild -project EyeContactFx.xcodeproj -scheme EyeContact -configuration Release -derivedDataPath build build
ditto "build/Build/Products/Release/Eye Contact.app" "/Applications/Eye Contact.app"
open "/Applications/Eye Contact.app"   # registers the plug-in; restart Final Cut Pro afterwards
```

Or run `EyeContactFx/install.sh`, which builds, installs, and also copies the Motion template.

- `MotionTemplate/Eye Contact/`: the Motion effect template that wraps the FxPlug filter. Final Cut Pro only lists the
  effect through this template, so it must be copied to `~/Movies/Motion Templates.localized/Effects.localized/`

- `Plugin/EyeContactPlugIn.swift`: the FxTileableEffect
  - Requests the full frame, downsamples it to 1280 px BGRA for Vision, and warps FCP's texture directly
  - Raw measurements go into a per-instance `TrackingCache` keyed by frame time and are smoothed on read with a centered Gaussian over whatever neighbors have been measured. FCP renders out of order, so smoothing never depends on render order. On a first, linear playback, smoothing only sees past frames; renders after that see both sides
- Parameters: Strength, Target X/Y, Max Correction, Smoothing, Iris Size, Show Tracking
- Ad-hoc signed for personal use. No developer team is needed
- If FCP doesn't list the effect, check `pluginkit -m -v -p FxPlug` and Console for "Eye Contact"

## License

- Source code: [MIT](LICENSE)
- Trained GazeWarp models (`EyeContactFx/Plugin/*.mlpackage`): [CC BY-NC 4.0](LICENSE-MODELS). They were trained on the
  [Columbia Gaze Data Set](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) (non-commercial research use only),
  so they inherit that restriction. Cite Smith et al., "Gaze Locking", UIST 2013 if you use them.
- Third-party, not included here: [LivePortrait](https://github.com/KwaiVGI/LivePortrait) is MIT, but the InsightFace
  detection models it downloads are non-commercial research only. MediaPipe (used by `gazenet/` for data prep) is Apache 2.0.
