package com.ncore.islcounter;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Insets;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.speech.tts.TextToSpeech;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.util.Locale;

/**
 * ISL Counter: the website in a full-screen WebView, with what a browser tab can't do well:
 *  - camera permission handled natively (Android prompt -> WebView grant)
 *  - instant offline speech through Android TextToSpeech (window.AndroidTTS)
 *  - rotation and resizing never reload the page or stop the camera
 *  - screen stays on while signing; system bars and notches never cover the page
 *  - a loading screen while the free server wakes up, and an offline screen with retry
 */
public class MainActivity extends Activity {

    private static final String SITE = BuildConfig.SITE_URL;
    private static final String SITE_HOST = Uri.parse(SITE).getHost();
    private static final int REQ_CAMERA = 1001;
    private static final int WALL = 0xFFEEF1F5, INDIGO = 0xFF1D2B64, INK = 0xFF121A33,
            SLATE = 0xFF5B6478, SIGNAL = 0xFFF2B705;

    private final Handler main = new Handler(Looper.getMainLooper());
    private FrameLayout root;
    private WebView web;
    private LinearLayout loadingView, errorView;
    private TextView loadingText;
    private ProgressBar progress;
    private PermissionRequest pendingPermission;
    private TextToSpeech tts;
    private volatile boolean ttsReady;
    private boolean mainFrameFailed;
    private int autoRetries;

    private final Runnable showWakingHint = () -> {
        if (loadingView.getVisibility() == View.VISIBLE) loadingText.setText(R.string.waking);
    };

    // ------------------------------------------------------------------ lifecycle
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        root = new FrameLayout(this);
        root.setBackgroundColor(WALL);
        setContentView(root);
        setupSystemBars();

        web = new WebView(this);
        web.setBackgroundColor(WALL);
        web.setOverScrollMode(View.OVER_SCROLL_NEVER);
        root.addView(web, new FrameLayout.LayoutParams(-1, -1));
        root.addView(buildLoadingView(), new FrameLayout.LayoutParams(-1, -1));
        root.addView(buildErrorView(), new FrameLayout.LayoutParams(-1, -1));

        tts = new TextToSpeech(getApplicationContext(), status -> {
            if (status != TextToSpeech.SUCCESS) return;
            Locale[] wanted = {Locale.forLanguageTag("en-IN"), Locale.UK, Locale.US, Locale.ENGLISH};
            for (Locale l : wanted) {
                int r = tts.setLanguage(l);
                if (r != TextToSpeech.LANG_MISSING_DATA && r != TextToSpeech.LANG_NOT_SUPPORTED) break;
            }
            ttsReady = true;
        });

        setupWebView();
        load();
    }

    @Override
    protected void onResume() {
        super.onResume();
        web.onResume();
        web.resumeTimers();
    }

    @Override
    protected void onPause() {
        if (tts != null) tts.stop();
        web.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        main.removeCallbacksAndMessages(null);
        if (tts != null) tts.shutdown();
        if (web != null) {
            root.removeView(web);
            web.destroy();
        }
        super.onDestroy();
    }

    @SuppressWarnings("deprecation")
    @Override
    public void onBackPressed() {
        if (errorView.getVisibility() != View.VISIBLE && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    // ------------------------------------------------------------------ system bars / notches
    private void setupSystemBars() {
        if (Build.VERSION.SDK_INT >= 35) {
            // Android 15+ always draws behind the bars: pad the content by the bars and the notch.
            root.setOnApplyWindowInsetsListener((v, insets) -> {
                Insets i = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                v.setPadding(i.left, i.top, i.right, i.bottom);
                return WindowInsets.CONSUMED;
            });
            WindowInsetsController c = getWindow().getInsetsController();
            if (c != null) {
                int light = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
                        | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS;
                c.setSystemBarsAppearance(light, light);
            }
        }
        // Older versions: the theme colours the bars and the system keeps content clear of them.
    }

    // ------------------------------------------------------------------ WebView
    @SuppressLint({"SetJavaScriptEnabled", "AddJavascriptInterface"})
    private void setupWebView() {
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);  // camera <video> must play without a tap
        s.setCacheMode(WebSettings.LOAD_DEFAULT);       // uses the site's caching: big files load once
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        s.setSupportZoom(false);
        s.setBuiltInZoomControls(false);
        s.setDisplayZoomControls(false);
        s.setTextZoom(100);                             // layout exactly as designed, whatever the system font size
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setGeolocationEnabled(false);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        s.setUserAgentString(s.getUserAgentString() + " ISLCounterApp/" + BuildConfig.VERSION_NAME);
        if (Build.VERSION.SDK_INT >= 26) s.setSafeBrowsingEnabled(true);
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);

        web.addJavascriptInterface(new SpeechBridge(), "AndroidTTS");

        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(PermissionRequest request) {
                main.post(() -> handlePermission(request));
            }

            @Override
            public void onPermissionRequestCanceled(PermissionRequest request) {
                if (request == pendingPermission) pendingPermission = null;
            }

            @Override
            public void onProgressChanged(WebView view, int p) {
                progress.setProgress(p);
            }
        });

        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                if (SITE_HOST != null && SITE_HOST.equalsIgnoreCase(u.getHost())) return false;
                try {   // other links open in the browser
                    startActivity(new Intent(Intent.ACTION_VIEW, u));
                } catch (ActivityNotFoundException ignored) { }
                return true;
            }

            @Override
            public void onPageStarted(WebView view, String url, android.graphics.Bitmap favicon) {
                mainFrameFailed = false;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                if (!mainFrameFailed) {
                    autoRetries = 0;
                    showPage();
                }
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) {
                    mainFrameFailed = true;
                    showError();
                }
            }

            @Override
            public void onReceivedHttpError(WebView view, WebResourceRequest request, WebResourceResponse response) {
                // A sleeping free server answers 502/503 for a moment while it starts: retry quietly.
                if (request.isForMainFrame() && response.getStatusCode() >= 500) {
                    mainFrameFailed = true;
                    if (autoRetries++ < 20) {
                        showLoading(true);
                        main.postDelayed(() -> web.reload(), 3000);
                    } else showError();
                }
            }

            @Override
            public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
                // The page's renderer was killed (low memory): rebuild instead of crashing.
                recreate();
                return true;
            }
        });
    }

    private void load() {
        showLoading(false);
        web.loadUrl(SITE);
    }

    // ------------------------------------------------------------------ camera permission
    private void handlePermission(PermissionRequest request) {
        boolean wantsCamera = false;
        for (String r : request.getResources()) {
            if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(r)) wantsCamera = true;
        }
        if (!wantsCamera) {
            request.deny();
            return;
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
        } else {
            pendingPermission = request;
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code != REQ_CAMERA) return;
        PermissionRequest req = pendingPermission;
        pendingPermission = null;
        boolean granted = results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED;
        if (req != null) {
            if (granted) req.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
            else req.deny();
        }
        if (!granted && !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            // "Don't ask again" was chosen: the only way back is the settings screen.
            new AlertDialog.Builder(this)
                    .setTitle(R.string.camera_needed_title)
                    .setMessage(R.string.camera_needed_body)
                    .setPositiveButton(R.string.open_settings, (d, w) -> startActivity(
                            new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                    Uri.fromParts("package", getPackageName(), null))))
                    .setNegativeButton(R.string.not_now, null)
                    .show();
        }
    }

    // ------------------------------------------------------------------ speech bridge
    /** Called by the page (static/app.js) as window.AndroidTTS. Runs on a binder thread. */
    final class SpeechBridge {
        @JavascriptInterface
        public boolean isReady() {
            return ttsReady;
        }

        @JavascriptInterface
        public void speak(String text, float rate) {
            if (!ttsReady || text == null || text.trim().isEmpty()) return;
            tts.setSpeechRate(rate > 0.3f && rate < 3f ? rate : 1f);
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "isl-" + System.nanoTime());
        }

        @JavascriptInterface
        public void stop() {
            if (ttsReady) tts.stop();
        }
    }

    // ------------------------------------------------------------------ loading / error screens
    private void showLoading(boolean waking) {
        errorView.setVisibility(View.GONE);
        loadingView.setVisibility(View.VISIBLE);
        loadingText.setText(waking ? R.string.waking : R.string.loading);
        main.removeCallbacks(showWakingHint);
        if (!waking) main.postDelayed(showWakingHint, 6000);
    }

    private void showPage() {
        main.removeCallbacks(showWakingHint);
        loadingView.setVisibility(View.GONE);
        errorView.setVisibility(View.GONE);
    }

    private void showError() {
        main.removeCallbacks(showWakingHint);
        loadingView.setVisibility(View.GONE);
        errorView.setVisibility(View.VISIBLE);
    }

    private int dp(float v) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()));
    }

    private LinearLayout column() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        l.setGravity(Gravity.CENTER);
        l.setPadding(dp(32), dp(32), dp(32), dp(32));
        l.setClickable(true);   // blocks touches to the page underneath
        return l;
    }

    private TextView text(CharSequence s, float sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        t.setGravity(Gravity.CENTER);
        t.setMaxWidth(dp(420));
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private View buildLoadingView() {
        loadingView = column();
        loadingView.setBackgroundColor(INDIGO);

        View mark = new View(this);
        mark.setBackground(getDrawable(R.drawable.ic_launcher_foreground));
        loadingView.addView(mark, new LinearLayout.LayoutParams(dp(96), dp(96)));

        TextView title = text(getString(R.string.app_name), 26, Color.WHITE, true);
        LinearLayout.LayoutParams tp = new LinearLayout.LayoutParams(-2, -2);
        tp.topMargin = dp(8);
        loadingView.addView(title, tp);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100);
        progress.setProgressTintList(android.content.res.ColorStateList.valueOf(SIGNAL));
        progress.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(0x33FFFFFF));
        LinearLayout.LayoutParams pp = new LinearLayout.LayoutParams(dp(220), dp(6));
        pp.topMargin = dp(28);
        loadingView.addView(progress, pp);

        loadingText = text(getString(R.string.loading), 15, 0xFFC9D1EA, false);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2);
        lp.topMargin = dp(16);
        loadingView.addView(loadingText, lp);
        return loadingView;
    }

    private View buildErrorView() {
        errorView = column();
        errorView.setBackgroundColor(WALL);
        errorView.setVisibility(View.GONE);

        errorView.addView(text(getString(R.string.offline_title), 22, INK, true));
        TextView body = text(getString(R.string.offline_body), 16, SLATE, false);
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(-2, -2);
        bp.topMargin = dp(10);
        errorView.addView(body, bp);

        Button retry = new Button(this);
        retry.setText(R.string.retry);
        retry.setAllCaps(false);
        retry.setTextColor(INK);
        retry.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        retry.setTypeface(Typeface.DEFAULT_BOLD);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(SIGNAL);
        bg.setCornerRadius(dp(10));
        retry.setBackground(bg);
        retry.setPadding(dp(28), dp(12), dp(28), dp(12));
        retry.setOnClickListener(v -> {
            autoRetries = 0;
            load();
        });
        LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(-2, ViewGroup.LayoutParams.WRAP_CONTENT);
        rp.topMargin = dp(24);
        errorView.addView(retry, rp);
        return errorView;
    }
}
