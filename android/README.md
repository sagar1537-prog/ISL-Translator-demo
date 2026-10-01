# ISL Counter for Android

A native Android app for https://isl-counter.onrender.com. Recognition runs on the phone, exactly as on the website.

What the app adds over a browser tab:

- **Camera:** Android asks once. If "Don't ask again" was chosen, the app offers to open Settings.
- **Speech:** words are spoken through the phone's own text-to-speech engine, offline and instantly.
- **Rotation:** turning the phone never reloads the page or stops the camera, and the layout switches between portrait and landscape.
- **Screen:** it stays on while you sign. Status bars, gesture bars and notches never cover the page (Android 8 to 15).
- **Loading:** a loading screen covers the free server's wake-up (up to a minute after a quiet spell). Short server errors are retried automatically, and there is an offline screen with a retry button.
- **Background:** going to the background and back restarts the camera automatically.

Requires Android 8.0 or newer (about 97% of devices).

## Get the APK without installing anything (GitHub)

Every push builds the app on GitHub. It then runs the app on Android 10 and Android 14 emulators: it opens the site, taps Start camera, rotates, goes to the background and back, and fails if anything crashes.

1. Open the repository on GitHub, then select **Actions** → **Android app** → the latest run.
2. Download **ISL-Counter-APK** (bottom of the page) and unzip it.
3. Copy `app-release.apk` to the phone, open it, and allow "Install unknown apps" when asked.

The **emulator-api…-screenshots** downloads show the app running on each Android version.

## Build in Android Studio

1. **File → Open** → choose this `android` folder, and wait for the Gradle sync to finish.
2. Plug in a phone with USB debugging on (or start an emulator), then press **Run ▶**.
3. For an APK file: **Build → Build App Bundle(s) / APK(s) → Build APK(s)**.

Command line: `gradlew.bat assembleRelease` (Windows) or `./gradlew assembleRelease`
→ `app/build/outputs/apk/release/app-release.apk`

## Settings

- **Site address:** `SITE_URL` in `app/build.gradle.kts`.
- **App name:** `res/values/strings.xml`.
- **Icon:** `res/drawable/ic_launcher_foreground.xml`.
- **Play Store:** the release APK is signed with the debug key so it installs directly. To publish on the Play Store,
  create an upload key (**Build → Generate Signed App Bundle**) and build an `.aab`.
