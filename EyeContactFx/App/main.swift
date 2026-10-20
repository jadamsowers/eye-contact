import AppKit

// The wrapper app only exists to carry the FxPlug XPC service; launching it once
// registers the plug-in with Final Cut Pro.
final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let label = NSTextField(wrappingLabelWithString: """
            The Eye Contact effect is installed.

            In Final Cut Pro, open the Effects browser and look for the "Eye Contact" category. \
            Turn on "Show Tracking" to check what it's following.
            """)
        label.frame = NSRect(x: 20, y: 20, width: 380, height: 110)

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 420, height: 150),
                          styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Eye Contact"
        window.contentView?.addSubview(label)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
