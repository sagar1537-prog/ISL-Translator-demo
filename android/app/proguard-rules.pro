# Keep the JavaScript bridge methods if minification is ever enabled.
-keepclassmembers class com.ncore.islcounter.MainActivity$SpeechBridge {
    @android.webkit.JavascriptInterface <methods>;
}
