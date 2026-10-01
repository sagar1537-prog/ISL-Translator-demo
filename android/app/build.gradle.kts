plugins {
    id("com.android.application")
}

android {
    namespace = "com.ncore.islcounter"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.ncore.islcounter"
        minSdk = 26          // Android 8.0+, about 97% of devices
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        // Change this if the site moves.
        buildConfigField("String", "SITE_URL", "\"https://isl-counter.onrender.com/\"")
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed with the debug key so the release APK installs directly.
            // Use your own key before publishing on the Play Store.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        abortOnError = true
        checkReleaseBuilds = false
    }
}
