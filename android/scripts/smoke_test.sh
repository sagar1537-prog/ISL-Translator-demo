#!/usr/bin/env bash
# Installs the app on the running emulator, opens the site, taps "Start camera",
# rotates the screen and fails if the app crashes or the page throws a JavaScript error.
set -u
PKG=com.ncore.islcounter
APK=app/build/outputs/apk/debug/app-debug.apk
OUT=smoke
mkdir -p "$OUT"
shot() { adb exec-out screencap -p > "$OUT/$1.png"; echo "screenshot $1"; }

# Wake the free Render server so the test measures the app, not a cold start.
for i in $(seq 1 20); do
  curl -fsS -m 25 https://isl-counter.onrender.com/healthz && echo " server awake" && break
  echo "waiting for server ($i)"; sleep 6
done

adb install -r -g "$APK"                       # -g grants the camera permission
adb shell settings put system accelerometer_rotation 0
adb shell settings put system user_rotation 0
adb logcat -c
adb shell am start -W -n "$PKG/.MainActivity"
sleep 30
shot 1-portrait-opened

adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1 && adb pull /sdcard/ui.xml "$OUT/ui-opened.xml" >/dev/null
if grep -q "Sign to the camera\|ISL Counter" "$OUT/ui-opened.xml" 2>/dev/null; then echo "page content visible"; fi

# Tap "Start camera" (found by text; falls back to the centre of the camera area)
XY=$(python3 - "$OUT/ui-opened.xml" <<'PY'
import re, sys
try:
    xml = open(sys.argv[1], encoding="utf-8").read()
    m = re.search(r'text="Start camera"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml) or \
        re.search(r'content-desc="Start camera"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml)
    if m:
        a, b, c, d = map(int, m.groups()); print((a + c) // 2, (b + d) // 2); sys.exit()
except Exception:
    pass
print("")
PY
)
if [ -z "$XY" ]; then
  SIZE=$(adb shell wm size | grep -o '[0-9]*x[0-9]*' | tail -1); W=${SIZE%x*}; H=${SIZE#*x}
  XY="$((W / 2)) $((H * 38 / 100))"; echo "button not in accessibility tree; tapping camera area at $XY"
fi
adb shell input tap $XY
sleep 20
shot 2-portrait-camera

adb shell settings put system user_rotation 1   # landscape
sleep 8
shot 3-landscape
adb shell settings put system user_rotation 0   # back to portrait
sleep 6
shot 4-portrait-again

adb shell input keyevent KEYCODE_HOME          # background ...
sleep 4
adb shell am start -n "$PKG/.MainActivity"     # ... and back: camera must resume
sleep 8
shot 5-resumed

adb logcat -d > "$OUT/logcat.txt"
FAIL=0
if ! adb shell pidof "$PKG" >/dev/null; then echo "FAIL: app is not running"; FAIL=1; fi
if grep -q "FATAL EXCEPTION" "$OUT/logcat.txt"; then echo "FAIL: app crashed"; grep -A20 "FATAL EXCEPTION" "$OUT/logcat.txt"; FAIL=1; fi
if grep -E "Uncaught|pageerror" "$OUT/logcat.txt" | grep -i "chromium\|console" | grep -v "favicon" ; then echo "FAIL: JavaScript error in the page"; FAIL=1; fi
grep -i "chromium.*CONSOLE" "$OUT/logcat.txt" | tail -20 > "$OUT/page-console.txt" || true
[ $FAIL -eq 0 ] && echo "PASS: app opened the site, started the camera, survived rotation and backgrounding"
exit $FAIL
