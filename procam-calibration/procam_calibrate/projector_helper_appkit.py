#!/usr/bin/env python3
"""Fullscreen projector helper using AppKit NSWindow on a chosen NSScreen index.

Protocol on stdin (one command per line):
  READY is printed on stdout when window is up.
  SHOW <path>
  BLACK
  QUIT
"""

from __future__ import annotations

import sys
import threading

import cv2
import numpy as np


def main() -> int:
    screen_index = int(sys.argv[1])
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyRegular,
        NSBackingStoreBuffered,
        NSColor,
        NSImage,
        NSImageView,
        NSScreen,
        NSWindow,
        NSWindowStyleMaskBorderless,
    )
    from Foundation import NSData, NSObject, NSTimer
    from PyObjCTools import AppHelper

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    screens = NSScreen.screens()
    if screen_index >= len(screens):
        print(f"ERR bad screen index {screen_index}", flush=True)
        return 2
    screen = screens[screen_index]
    frame = screen.frame()

    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame,
        NSWindowStyleMaskBorderless,
        NSBackingStoreBuffered,
        False,
    )
    window.setFrame_display_animate_(frame, True, False)
    window.setLevel_(1000)
    window.setBackgroundColor_(NSColor.blackColor())
    window.setOpaque_(True)
    window.setHidesOnDeactivate_(False)
    image_view = NSImageView.alloc().initWithFrame_(window.contentView().bounds())
    image_view.setImageScaling_(1)  # NSImageScaleAxesIndependently / fill
    image_view.setAutoresizingMask_(18)
    window.contentView().addSubview_(image_view)
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)

    state = {"pending": None, "stop": False}

    def set_bgr(img: np.ndarray) -> None:
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf.tobytes()))
        nsimg = NSImage.alloc().initWithData_(data)
        image_view.setImage_(nsimg)
        window.display()

    # initial black
    set_bgr(np.zeros((height, width, 3), dtype=np.uint8))
    print("READY", flush=True)

    def stdin_thread():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "QUIT":
                state["stop"] = True
                AppHelper.callAfter(AppHelper.stopEventLoop)
                return
            if line == "BLACK":
                state["pending"] = ("BLACK", None)
                continue
            if line.startswith("SHOW "):
                state["pending"] = ("SHOW", line[5:])
                continue
            print("ERR unknown", line, flush=True)

    threading.Thread(target=stdin_thread, daemon=True).start()

    class Ticker(NSObject):
        def tick_(self, _timer):
            if state["stop"]:
                AppHelper.stopEventLoop()
                return
            pending = state["pending"]
            if pending is None:
                return
            state["pending"] = None
            kind, payload = pending
            try:
                if kind == "BLACK":
                    set_bgr(np.zeros((height, width, 3), dtype=np.uint8))
                    print("OK BLACK", flush=True)
                elif kind == "SHOW":
                    img = cv2.imread(payload, cv2.IMREAD_COLOR)
                    if img is None:
                        print("ERR read", payload, flush=True)
                        return
                    set_bgr(img)
                    print("OK SHOW", payload, flush=True)
            except Exception as e:
                print("ERR", e, flush=True)

    ticker = Ticker.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.03, ticker, "tick:", None, True
    )
    AppHelper.runEventLoop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
