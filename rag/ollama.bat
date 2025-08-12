@echo off
setlocal

where /Q ollama
if %ERRORLEVEL% EQU 0 (
  ollama %*
  exit /b %ERRORLEVEL%
)

set "CANDIDATE1=%LocalAppData%\Programs\Ollama\ollama.exe"
if exist "%CANDIDATE1%" (
  "%CANDIDATE1%" %*
  exit /b %ERRORLEVEL%
)

set "CANDIDATE2=%ProgramFiles%\Ollama\ollama.exe"
if exist "%CANDIDATE2%" (
  "%CANDIDATE2%" %*
  exit /b %ERRORLEVEL%
)

set "CANDIDATE3=%ProgramFiles(x86)%\Ollama\ollama.exe"
if exist "%CANDIDATE3%" (
  "%CANDIDATE3%" %*
  exit /b %ERRORLEVEL%
)

echo Ollama executable not found. Install Ollama and ensure it is in PATH: https://ollama.com
exit /b 1