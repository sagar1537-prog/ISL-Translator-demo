@rem Gradle wrapper launcher for Windows
@if "%DEBUG%"=="" @echo off
setlocal
set APP_HOME=%~dp0
if "%APP_HOME%"=="" set APP_HOME=.
set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar

if defined JAVA_HOME (
    set JAVA_EXE=%JAVA_HOME%\bin\java.exe
) else (
    set JAVA_EXE=java.exe
)
"%JAVA_EXE%" -version >NUL 2>&1
if errorlevel 1 (
    echo ERROR: Java was not found. Set JAVA_HOME or install a JDK ^(Android Studio includes one^).
    exit /b 1
)
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-Dorg.gradle.appname=gradlew" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
exit /b %ERRORLEVEL%
