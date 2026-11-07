@echo off
rem ============================================================================
rem  Drop-and-go entrypoint.
rem
rem  Two ways to use it:
rem    1. DRAG image files (or folders of them) ONTO this file. They get copied
rem       into data\inbox, and everything in the inbox is processed.
rem    2. DOUBLE-CLICK it. Whatever is already in data\inbox gets processed.
rem
rem  Either way the report opens by itself when it finishes.
rem
rem  NOTE TO ANYONE EDITING THIS FILE:
rem    - Keep it CRLF. cmd.exe seeks by byte offset assuming CRLF; an LF-only
rem      batch file desyncs and silently eats the first characters of lines
rem      (rem -> m, setlocal -> tlocal). It fails in a baffling way.
rem    - Keep it plain ASCII. The console codepage is not UTF-8.
rem    - .cmd not .ps1 on purpose: PowerShell's RemoteSigned policy blocks
rem      unsigned .ps1 files, .cmd is unaffected.
rem ============================================================================
setlocal
cd /d "%~dp0"

if not exist "data\inbox" mkdir "data\inbox"

rem --- copy anything dragged onto this file into the inbox --------------------
:copyloop
if "%~1"=="" goto run
if exist "%~1\" (
    echo Adding folder: %~nx1
    xcopy /y /q "%~1\*.jpg"  "data\inbox\" >nul 2>&1
    xcopy /y /q "%~1\*.jpeg" "data\inbox\" >nul 2>&1
    xcopy /y /q "%~1\*.png"  "data\inbox\" >nul 2>&1
    xcopy /y /q "%~1\*.tif"  "data\inbox\" >nul 2>&1
) else (
    echo Adding file:   %~nx1
    copy /y "%~1" "data\inbox\" >nul
)
shift
goto copyloop

:run
echo.
rem Count IMAGES, not files - README.txt lives in the inbox and must not count.
set COUNT=0
for %%F in (data\inbox\*.jpg data\inbox\*.jpeg data\inbox\*.png data\inbox\*.tif data\inbox\*.tiff data\inbox\*.bmp data\inbox\*.webp) do set /a COUNT+=1
if %COUNT%==0 (
    echo No images in the inbox yet.
    echo.
    echo   Put them in:  %CD%\data\inbox
    echo   ...or just drag them straight onto this file.
    echo.
    pause
    exit /b 1
)
echo Processing %COUNT% image^(s^)...
echo.

".venv\Scripts\python.exe" -m aihtr "data\inbox" --out "out"
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
    if exist "out\report.html" (
        rem AIHTR_NO_OPEN=1 suppresses the browser launch, for testing.
        if "%AIHTR_NO_OPEN%"=="1" (
            echo Report written ^(browser launch suppressed^).
        ) else (
            echo Opening the report...
            start "" "out\report.html"
        )
    )
) else (
    echo Finished with errors ^(exit code %RC%^).
)
echo.
echo Inbox:  %CD%\data\inbox     ^(clear it out when you are done^)
echo Output: %CD%\out
echo.
pause
