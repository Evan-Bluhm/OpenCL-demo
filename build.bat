@echo off
REM Windows command-line build for the two-stream electrostatic PIC demo.
REM
REM Usage:
REM   build.bat              - build bin\two_stream.exe
REM   build.bat run [args]   - build, then run with optional args
REM   build.bat clean        - remove app objects and binary
REM   build.bat distclean    - clean plus remove ImGui/GLFW/GLEW caches
REM
REM Requirements (one-time):
REM   - Visual Studio 2019 or 2022 with the "Desktop development with C++"
REM     workload (free Build Tools edition is fine). The script auto-locates
REM     vcvars64.bat via vswhere; no need to launch a Developer Prompt yourself.
REM   - git on PATH (used to fetch Dear ImGui).
REM   - PowerShell (shipped with Windows; used to download GLFW/GLEW).
REM
REM On first build, GLFW, GLEW and Dear ImGui are auto-fetched into extern\.
REM glfw3.dll and glew32.dll are copied next to the .exe so it can run from
REM Explorer or any cmd window without extra PATH setup.

setlocal EnableDelayedExpansion
pushd "%~dp0"

REM ---- Argument parsing -----------------------------------------------------
set "ACTION=build"
if /I "%~1"=="clean"     set "ACTION=clean"
if /I "%~1"=="distclean" set "ACTION=distclean"
if /I "%~1"=="run"       set "ACTION=run"

if "%ACTION%"=="clean" (
    echo Removing build\ and bin\
    if exist build rmdir /s /q build
    if exist bin   rmdir /s /q bin
    popd & endlocal & exit /b 0
)
if "%ACTION%"=="distclean" (
    echo Removing build\, bin\, and extern\
    if exist build  rmdir /s /q build
    if exist bin    rmdir /s /q bin
    if exist extern rmdir /s /q extern
    popd & endlocal & exit /b 0
)

REM ---- Layout ---------------------------------------------------------------
set "BIN_DIR=bin"
set "OBJ_DIR=build"
set "EXT_DIR=extern"
set "IMGUI_DIR=%EXT_DIR%\imgui"
set "IMGUI_BLD=%IMGUI_DIR%\build"
set "GLFW_DIR=%EXT_DIR%\glfw"
set "GLEW_DIR=%EXT_DIR%\glew"
set "TARGET=%BIN_DIR%\two_stream.exe"

set "IMGUI_TAG=v1.91.0"
set "IMGUI_URL=https://github.com/ocornut/imgui.git"
set "GLFW_VER=3.4"
set "GLFW_URL=https://github.com/glfw/glfw/releases/download/%GLFW_VER%/glfw-%GLFW_VER%.bin.WIN64.zip"
set "GLEW_VER=2.1.0"
set "GLEW_URL=https://github.com/nigels-com/glew/releases/download/glew-%GLEW_VER%/glew-%GLEW_VER%-win32.zip"

REM ---- Locate MSVC ----------------------------------------------------------
REM If cl.exe isn't already on PATH, find vcvars64.bat via vswhere and source it.
REM Use goto rather than a multi-line `if (...)` block so plain %VAR% expansion
REM is reliable (delayed expansion inside parens has tripped users up here).
where cl.exe >nul 2>&1
if not errorlevel 1 goto :have_msvc

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: cl.exe not found and vswhere.exe missing.
    echo Install Visual Studio Build Tools 2019/2022 with the C++ workload, or
    echo run this script from a "x64 Native Tools Command Prompt for VS".
    popd & endlocal & exit /b 1
)
REM Capture via a temp file rather than `for /f`: the expanded VSWHERE path
REM contains "(x86)", whose ')' confuses the for-loop's parenthesized clause.
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\vspath.txt"
set /p VSPATH=<"%TEMP%\vspath.txt"
del "%TEMP%\vspath.txt" 2>nul
if not defined VSPATH (
    echo ERROR: vswhere did not locate a VS install with the C++ x64 toolset.
    popd & endlocal & exit /b 1
)
set "VCVARS=%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%VCVARS%".
    popd & endlocal & exit /b 1
)
echo Activating MSVC environment from "%VSPATH%"
REM vcvars64.bat itself spawns subprocesses (vsdevcmd, vswhere) that may write
REM noise to stderr even on success; suppress both streams and trust errorlevel.
call "%VCVARS%" >nul 2>nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed.
    popd & endlocal & exit /b 1
)
:have_msvc

REM ---- Auto-fetch dependencies ----------------------------------------------
if not exist "%EXT_DIR%" mkdir "%EXT_DIR%"

if not exist "%IMGUI_DIR%\imgui.cpp" (
    echo Fetching Dear ImGui %IMGUI_TAG%...
    git clone --depth 1 --branch %IMGUI_TAG% %IMGUI_URL% "%IMGUI_DIR%"
    if errorlevel 1 (
        echo ERROR: git clone of ImGui failed.
        popd & endlocal & exit /b 1
    )
)

if not exist "%GLFW_DIR%\include\GLFW\glfw3.h" (
    echo Fetching GLFW %GLFW_VER% Win64 binaries...
    call :Fetch "%GLFW_URL%" "%EXT_DIR%\glfw.zip" "%EXT_DIR%" "glfw-%GLFW_VER%.bin.WIN64" "%GLFW_DIR%"
    if errorlevel 1 (
        echo ERROR: failed to fetch/extract GLFW.
        popd & endlocal & exit /b 1
    )
)

if not exist "%GLEW_DIR%\include\GL\glew.h" (
    echo Fetching GLEW %GLEW_VER% Win32/Win64 binaries...
    call :Fetch "%GLEW_URL%" "%EXT_DIR%\glew.zip" "%EXT_DIR%" "glew-%GLEW_VER%" "%GLEW_DIR%"
    if errorlevel 1 (
        echo ERROR: failed to fetch/extract GLEW.
        popd & endlocal & exit /b 1
    )
)

REM ---- Pick a GLFW lib subdir compatible with the active MSVC ---------------
REM GLFW ships per-toolset import libs (lib-vc2019, lib-vc2022, etc).
REM Prefer 2022, fall back to 2019, then any present.
set "GLFW_LIB_DIR="
if exist "%GLFW_DIR%\lib-vc2022\glfw3dll.lib" set "GLFW_LIB_DIR=%GLFW_DIR%\lib-vc2022"
if not defined GLFW_LIB_DIR if exist "%GLFW_DIR%\lib-vc2019\glfw3dll.lib" set "GLFW_LIB_DIR=%GLFW_DIR%\lib-vc2019"
if not defined GLFW_LIB_DIR (
    for /d %%d in ("%GLFW_DIR%\lib-vc*") do (
        if exist "%%d\glfw3dll.lib" set "GLFW_LIB_DIR=%%d"
    )
)
if not defined GLFW_LIB_DIR (
    echo ERROR: no compatible glfw3dll.lib found under "%GLFW_DIR%".
    popd & endlocal & exit /b 1
)

set "GLEW_LIB=%GLEW_DIR%\lib\Release\x64\glew32.lib"
set "GLEW_DLL=%GLEW_DIR%\bin\Release\x64\glew32.dll"
set "GLFW_DLL=%GLFW_LIB_DIR%\glfw3.dll"
set "OPENCL_LIB=src\OpenCL\lib\OpenCL.lib"

if not exist "%GLEW_LIB%" (
    echo ERROR: missing %GLEW_LIB%
    popd & endlocal & exit /b 1
)
if not exist "%OPENCL_LIB%" (
    echo ERROR: missing %OPENCL_LIB%
    popd & endlocal & exit /b 1
)

REM ---- Compile flags --------------------------------------------------------
REM /MD     - link the dynamic CRT (matches the GLFW prebuilts)
REM /EHsc   - standard C++ exceptions
REM /std:c++17, /O2, /utf-8, /nologo, /W3
REM /D_CRT_SECURE_NO_WARNINGS - quiet MSVC's deprecation warnings on POSIX names
REM /DNOMINMAX, /DWIN32_LEAN_AND_MEAN - keep <windows.h> from clobbering identifiers
REM /DGLFW_INCLUDE_NONE - stop GLFW from pulling in the Windows SDK <GL/gl.h>
REM   ahead of <GL/glew.h>; the app explicitly includes GLEW already.
set "CXX_COMMON=/nologo /std:c++17 /EHsc /MD /O2 /utf-8 /W3 /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN /DGLFW_INCLUDE_NONE"
set "APP_INC=/Isrc /Isrc\OpenCL\include /I%IMGUI_DIR% /I%IMGUI_DIR%\backends /I%GLFW_DIR%\include /I%GLEW_DIR%\include"
set "IMGUI_INC=/I%IMGUI_DIR% /I%IMGUI_DIR%\backends /I%GLFW_DIR%\include"

REM ---- Build ImGui static lib (cached) --------------------------------------
set "IMGUI_AR=%IMGUI_BLD%\imgui.lib"
if not exist "%IMGUI_AR%" (
    echo Building Dear ImGui static lib...
    if not exist "%IMGUI_BLD%" mkdir "%IMGUI_BLD%"
    set "IMGUI_OBJS="
    for %%F in (
        "%IMGUI_DIR%\imgui.cpp"
        "%IMGUI_DIR%\imgui_draw.cpp"
        "%IMGUI_DIR%\imgui_tables.cpp"
        "%IMGUI_DIR%\imgui_widgets.cpp"
        "%IMGUI_DIR%\backends\imgui_impl_glfw.cpp"
        "%IMGUI_DIR%\backends\imgui_impl_opengl3.cpp"
    ) do (
        echo   compile %%~nxF
        cl /c %CXX_COMMON% %IMGUI_INC% /Fo"%IMGUI_BLD%\%%~nF.obj" "%%~F" >nul
        if errorlevel 1 (
            echo ERROR: ImGui compile failed for %%~nxF
            popd & endlocal & exit /b 1
        )
        set "IMGUI_OBJS=!IMGUI_OBJS! "%IMGUI_BLD%\%%~nF.obj""
    )
    lib /nologo /OUT:"%IMGUI_AR%" !IMGUI_OBJS!
    if errorlevel 1 (
        echo ERROR: lib.exe failed to build %IMGUI_AR%
        popd & endlocal & exit /b 1
    )
)

REM ---- Build app objects ----------------------------------------------------
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"
if not exist "%OBJ_DIR%\src" mkdir "%OBJ_DIR%\src"

echo Compiling two_stream.cpp
cl /c %CXX_COMMON% %APP_INC% /Fo"%OBJ_DIR%\two_stream.obj" two_stream.cpp
if errorlevel 1 (
    echo ERROR: compile failed for two_stream.cpp
    popd & endlocal & exit /b 1
)

echo Compiling src\kernel.cpp
cl /c %CXX_COMMON% %APP_INC% /Fo"%OBJ_DIR%\src\kernel.obj" src\kernel.cpp
if errorlevel 1 (
    echo ERROR: compile failed for src\kernel.cpp
    popd & endlocal & exit /b 1
)

REM ---- Link -----------------------------------------------------------------
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
echo Linking %TARGET%
link /nologo /OUT:"%TARGET%" ^
    "%OBJ_DIR%\two_stream.obj" "%OBJ_DIR%\src\kernel.obj" ^
    "%IMGUI_AR%" ^
    /LIBPATH:"%GLFW_LIB_DIR%" glfw3dll.lib ^
    "%GLEW_LIB%" ^
    "%OPENCL_LIB%" ^
    opengl32.lib gdi32.lib user32.lib shell32.lib kernel32.lib
if errorlevel 1 (
    echo ERROR: link failed.
    popd & endlocal & exit /b 1
)

REM ---- Stage runtime DLLs next to the exe -----------------------------------
copy /y "%GLFW_DLL%" "%BIN_DIR%\" >nul
copy /y "%GLEW_DLL%" "%BIN_DIR%\" >nul

echo Built %TARGET%

if "%ACTION%"=="run" (
    REM Forward up to 8 args after "run" — the demo only takes an optional device id.
    "%TARGET%" %2 %3 %4 %5 %6 %7 %8 %9
)

popd & endlocal & exit /b 0

REM ---- Helper: download + extract a zip, rename top-level dir ---------------
REM %1 url  %2 zip path  %3 extract-into dir  %4 expected top-level name  %5 desired final dir
:Fetch
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%~1' -OutFile '%~2'; Expand-Archive -Force -Path '%~2' -DestinationPath '%~3'; if (Test-Path '%~3\%~4') { if (Test-Path '%~5') { Remove-Item -Recurse -Force '%~5' }; Rename-Item -Path '%~3\%~4' -NewName (Split-Path '%~5' -Leaf) }; Remove-Item '%~2'"
    exit /b %errorlevel%
