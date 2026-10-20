#!/bin/zsh
# Build the plug-in, install it to /Applications, and make sure that's the only registered copy.
set -euo pipefail
cd "${0:A:h}"

LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
BUILT="build/Build/Products/Release/Eye Contact.app"
DEST="/Applications/Eye Contact.app"

xcodegen -q
xcodebuild -project EyeContactFx.xcodeproj -scheme EyeContact -configuration Release \
  -derivedDataPath build build -quiet

# Xcode registers every app it builds; a second copy of the plug-in confuses PlugInKit.
for app in build/Build/Products/*/"Eye Contact.app"; do
  pluginkit -r "$app/Contents/PlugIns/Eye Contact XPC Service.pluginkit" 2>/dev/null || true
  "$LSREGISTER" -u "$app" 2>/dev/null || true
done

pkill -f "Eye Contact XPC Service" 2>/dev/null || true
rm -rf "$DEST"
ditto "$BUILT" "$DEST"
"$LSREGISTER" -f "$DEST"
pluginkit -a "$DEST/Contents/PlugIns/Eye Contact XPC Service.pluginkit"
open "$DEST"

echo "Registered copies:"
pluginkit -m -A -D -v -i local.eyecontact.app.xpc
echo "Restart Final Cut Pro to pick up the change."
