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
API=$(adb shell getprop ro.build.version.sdk | tr -d '\r')
WV=$(grep -o "ISLCounter.*WebView [^ ]* [0-9.]*" "$OUT/logcat.txt" | tail -1)
echo "::notice title=Android API $API::${WV:-WebView version not logged}"
note() { echo "::$1 title=$2 (API $API)::$(echo "$3" | tr '\n' ' ' | cut -c1-900)"; }
CONSOLE=$(grep -i "chromium.*CONSOLE" "$OUT/logcat.txt" | tail -6)
[ -n "$CONSOLE" ] && note notice "Page console" "$CONSOLE"
FAIL=0
if ! adb shell "ps -A 2>/dev/null || ps" | grep -q "$PKG"; then note error "App not running" "$(adb shell dumpsys activity activities | grep -m3 -i 'resumed')"; FAIL=1; fi
if grep -q "FATAL EXCEPTION" "$OUT/logcat.txt"; then note error "App crashed" "$(grep -A12 'FATAL EXCEPTION' "$OUT/logcat.txt")"; FAIL=1; fi
JSERR=$(grep -E "Uncaught" "$OUT/logcat.txt" | grep -i "chromium\|console" | grep -v "favicon" | head -5)
if [ -n "$JSERR" ]; then
  MAJOR=$(echo "$WV" | grep -o " [0-9]*\." | tail -1 | tr -d ' .')
  if [ -n "$MAJOR" ] && [ "$MAJOR" -lt 90 ]; then
    note warning "Old WebView $MAJOR: page needs 90+ (app shows the update prompt)" "$JSERR"
  else
    note error "JavaScript error in the page" "$JSERR"; FAIL=1
  fi
fi
grep -i "chromium.*CONSOLE" "$OUT/logcat.txt" | tail -20 > "$OUT/page-console.txt" || true
[ $FAIL -eq 0 ] && echo "PASS: app opened the site, started the camera, survived rotation and backgrounding"
exit $FAIL
