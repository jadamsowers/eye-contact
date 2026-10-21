#!/bin/zsh
# Build the plug-in and package it with its Motion template and installer as dist/EyeContact.zip,
# for installing on another Mac without building there. Pass --no-build to package the existing build.
set -euo pipefail
cd "${0:A:h}"

BUILT="build/Build/Products/Release/Eye Contact.app"
OUT="../dist"
STAGE="$OUT/EyeContact"

if [[ "${1:-}" != "--no-build" ]]; then
  xcodegen -q
  xcodebuild -project EyeContactFx.xcodeproj -scheme EyeContact -configuration Release \
    -derivedDataPath build build -quiet
fi
codesign --verify --deep --strict "$BUILT"

rm -rf "$STAGE" "$OUT/EyeContact.zip"
mkdir -p "$STAGE"
ditto "$BUILT" "$STAGE/Eye Contact.app"
ditto MotionTemplate "$STAGE/MotionTemplate"
cp Package/install.command Package/README.txt "$STAGE/"
chmod +x "$STAGE/install.command"
(cd "$OUT" && ditto -c -k --keepParent EyeContact EyeContact.zip)

echo "Wrote ${OUT:A}/EyeContact.zip"
