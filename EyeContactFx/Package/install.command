#!/bin/zsh
# Install the Eye Contact Final Cut Pro plug-in from this folder into /Applications,
# plus the Motion template Final Cut Pro uses to list the effect.
set -euo pipefail
cd "${0:A:h}"

LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
SRC="Eye Contact.app"
DEST="/Applications/Eye Contact.app"
EXT="Contents/PlugIns/Eye Contact XPC Service.pluginkit"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This build is Apple Silicon (arm64) only." >&2
  exit 1
fi

# The app is ad-hoc signed; a copied/downloaded file carries a quarantine flag that Gatekeeper would block.
xattr -dr com.apple.quarantine "$SRC" 2>/dev/null || true

# Unregister any previous copy so PlugInKit only sees the new one.
if [[ -d "$DEST" ]]; then
  pluginkit -r "$DEST/$EXT" 2>/dev/null || true
  "$LSREGISTER" -u "$DEST" 2>/dev/null || true
fi
pkill -f "Eye Contact XPC Service" 2>/dev/null || true

rm -rf "$DEST"
ditto "$SRC" "$DEST"
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
"$LSREGISTER" -f "$DEST"
pluginkit -a "$DEST/$EXT"
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

echo
echo "Registered copies:"
pluginkit -m -A -D -v -i local.eyecontact.app.xpc || true
echo
echo "Done. Restart Final Cut Pro, then look for Eye Contact in the Effects browser."
