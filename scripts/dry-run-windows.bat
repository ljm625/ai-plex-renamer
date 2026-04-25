@echo off
setlocal EnableExtensions

rem Replace these placeholders with your real keys before running.
set "TMDB_API_KEY=PASTE_TMDB_API_KEY_HERE"
set "NVIDIA_API_KEY=PASTE_NVIDIA_API_KEY_HERE"
set "NVIDIA_MODEL=meta/llama-3.1-8b-instruct"
set "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1"

echo AI Plex Renamer dry-run
echo.
set /p "MEDIA_DIR=Input media folder path: "

if "%MEDIA_DIR%"=="" (
  echo No folder path provided.
  exit /b 1
)

if not exist "%MEDIA_DIR%" (
  echo Folder does not exist: "%MEDIA_DIR%"
  exit /b 1
)

where ai-plex-renamer >nul 2>nul
if errorlevel 1 (
  echo ai-plex-renamer was not found in PATH.
  echo Install first with: python -m pip install -e .
  exit /b 1
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

echo.
echo Dry-run finished. Add --apply manually only after checking the output.
endlocal
