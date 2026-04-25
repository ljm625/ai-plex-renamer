@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

rem Replace these placeholders with your real keys before running.
set "TMDB_API_KEY=PASTE_TMDB_API_KEY_HERE"
set "NVIDIA_API_KEY=PASTE_NVIDIA_API_KEY_HERE"
set "NVIDIA_MODEL=meta/llama-3.1-8b-instruct"
set "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1"

echo AI Plex Renamer dry-run
echo.

if not "%~1"=="" (
  set "MEDIA_DIR=%~1"
) else (
  set /p "MEDIA_DIR=Input media folder path: "
)

rem Dragging a folder into an interactive cmd prompt includes quotes.
set "MEDIA_DIR=%MEDIA_DIR:"=%"

if not defined MEDIA_DIR (
  echo No folder path provided.
  goto :fail
)

if not exist "%MEDIA_DIR%\." (
  echo Folder does not exist: "%MEDIA_DIR%"
  goto :fail
)

where ai-plex-renamer >nul 2>nul
if errorlevel 1 (
  echo ai-plex-renamer was not found in PATH.
  echo Install first with: python -m pip install -e .
  goto :fail
)

echo.
echo Running dry-run. No files will be renamed.
echo.

ai-plex-renamer "%MEDIA_DIR%" ^
  --tmdb-api-key "%TMDB_API_KEY%" ^
  --tmdb-include-adult ^
  --nvidia-api-key "%NVIDIA_API_KEY%" ^
  --nvidia-model "%NVIDIA_MODEL%" ^
  --nvidia-base-url "%NVIDIA_BASE_URL%" ^
  --verbose

if errorlevel 1 (
  echo.
  echo ai-plex-renamer exited with an error.
  goto :fail
)

echo.
echo Dry-run finished. Add --apply manually only after checking the output.
goto :done

:fail
echo.
echo Dry-run did not start.
pause
exit /b 1

:done
pause
endlocal
