#!/bin/zsh
# Build the plug-in, install it to /Applications, make sure that's the only registered copy,
# and install the Motion template Final Cut Pro uses to list the effect.
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

# Final Cut Pro only lists the effect through its Motion template.
TEMPLATES=~/Movies/"Motion Templates.localized"
if [[ -d ~/Movies/"Motion Templates" && ! -d "$TEMPLATES" ]]; then TEMPLATES=~/Movies/"Motion Templates"; fi
EFFECTS="$TEMPLATES/Effects.localized"
if [[ -d "$TEMPLATES/Effects" && ! -d "$EFFECTS" ]]; then EFFECTS="$TEMPLATES/Effects"; fi
mkdir -p "$EFFECTS"
rm -rf "$EFFECTS/Eye Contact"
ditto "MotionTemplate/Eye Contact" "$EFFECTS/Eye Contact"
mkdir -p "$EFFECTS/Eye Contact/Eye Contact/Media"
xattr -dr com.apple.quarantine "$EFFECTS/Eye Contact" 2>/dev/null || true
echo "Installed Motion template to $EFFECTS/Eye Contact"

echo "Registered copies:"
pluginkit -m -A -D -v -i local.eyecontact.app.xpc
echo "Restart Final Cut Pro to pick up the change."
