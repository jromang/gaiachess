@echo off
setlocal enabledelayedexpansion

rem ===========================================================
rem Build GaiaChess for Windows with PGO (Profile-Guided Optimization).
rem Toolchain: Rust/MSVC (x86_64-pc-windows-msvc) — native build, no cross-compilation.
rem Produces 7 executables: nnue-avx2, nnue-pext, nnue-avx512, nnue-znver3/4/5, pesto.
rem Everything is packaged into gaiachess-VERSION-w64.zip with the README.
rem
rem Prerequisites: rustup (msvc toolchain), llvm-profdata (in toolchain or PATH)
rem                Visual Studio Build Tools (MSVC linker)
rem
rem Usage: tools\pgo\build-pgo-windows.bat [net.nnue] [extra-features]
rem
rem Examples:
rem   tools\pgo\build-pgo-windows.bat
rem   tools\pgo\build-pgo-windows.bat nets/gen1-sb800.nnue
rem   tools\pgo\build-pgo-windows.bat nets/gen1-sb800.nnue spsa
rem ===========================================================

pushd "%~dp0..\.."
set "ROOT=%CD%"

rem --- Arguments ---
set "NET_ARG=%~1"
set "EXTRA_FEATURES=%~2"

rem --- Parse defaults.conf (eol=# ignores comments) ---
set "DEFAULT_NET="
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("defaults.conf") do (
    set "_k=%%A"
    set "_v=%%~B"
    set "_k=!_k: =!"
    if "!_k!"=="DEFAULT_NET" set "DEFAULT_NET=!_v!"
)

if not defined DEFAULT_NET (
    echo Error: DEFAULT_NET not found in defaults.conf
    popd & exit /b 1
)

if "%NET_ARG%"=="" (set "NET=%DEFAULT_NET%") else (set "NET=%NET_ARG%")

rem --- Features ---
set "FEATURES=nnue,syzygy,nalimov,gaiatb,online-tb"
if not "%EXTRA_FEATURES%"=="" set "FEATURES=nnue,syzygy,nalimov,gaiatb,online-tb,%EXTRA_FEATURES%"

rem --- Version from Cargo.toml (findstr /b = line starting with pattern) ---
set "VERSION="
for /f "tokens=3 delims= " %%V in ('findstr /b "version =" Cargo.toml') do (
    if not defined VERSION (
        set "_v=%%V"
        set "VERSION=!_v:"=!"
    )
)
if not defined VERSION (
    echo Error: version not found in Cargo.toml
    popd & exit /b 1
)

rem --- Directories ---
set "PGO_DIR=%TEMP%\gaiachess-pgo"
set "ZIP_NAME=gaiachess-%VERSION%-w64"
set "ZIP_DIR=%TEMP%\%ZIP_NAME%"
set "BUILD_TARGET=x86_64-pc-windows-msvc"
set "CARGO_EXE=target\%BUILD_TARGET%\release\gaiachess.exe"

rem --- Check network ---
if not exist "%NET%" (
    echo Error: network '%NET%' not found.
    popd & exit /b 1
)

rem --- Find llvm-profdata ---
set "LLVM_PROFDATA="

rem 1) In PATH
for /f "delims=" %%P in ('where llvm-profdata 2^>nul') do (
    if not defined LLVM_PROFDATA set "LLVM_PROFDATA=%%P"
)

rem 2) In the active rustup toolchain
rem    Note: no quotes around the findstr term (parentheses = batch issue)
if not defined LLVM_PROFDATA (
    for /f "delims=" %%H in ('rustup show home') do set "RUSTUP_HOME=%%H"
    set "_TC="
    for /f "tokens=1 delims= " %%T in ('rustup toolchain list') do (
        if not defined _TC set "_TC=%%T"
    )
    if defined _TC (
        set "_LP=!RUSTUP_HOME!\toolchains\!_TC!\lib\rustlib\x86_64-pc-windows-msvc\bin\llvm-profdata.exe"
        if exist "!_LP!" set "LLVM_PROFDATA=!_LP!"
    )
)

if not defined LLVM_PROFDATA (
    echo Error: llvm-profdata not found.
    echo   Solutions:
    echo     winget install LLVM.LLVM     ^(adds llvm-profdata to PATH^)
    echo     or: verify the Rust MSVC toolchain is installed
    popd & exit /b 1
)

rem --- Init ---
if exist "%ZIP_DIR%" rd /s /q "%ZIP_DIR%"
mkdir "%ZIP_DIR%"

echo.
echo === GaiaChess PGO Windows Build ===
echo   Version  : %VERSION%
echo   Target   : x86_64-pc-windows-msvc ^(native^)
echo   Network  : %NET%
echo   Features : %FEATURES%
echo   Variants : nnue-avx2, nnue-pext, nnue-avx512, nnue-znver3/4/5, pesto
echo   LTO      : fat
echo   PGO dir  : %PGO_DIR%
echo   llvm-profdata: %LLVM_PROFDATA%
echo.

rem === NNUE variants ===
call :build "x86-64-v3" "nnue-avx2"   "%FEATURES%" "-C target-feature=-bmi2"
if errorlevel 1 goto :fail

call :build "x86-64-v3" "nnue-pext"   "%FEATURES%" ""
if errorlevel 1 goto :fail

call :build "x86-64-v4" "nnue-avx512" "%FEATURES%" "-C target-feature=+avx512f,+avx512bw,+avx512vl"
if errorlevel 1 goto :fail

call :build "znver3"    "nnue-znver3" "%FEATURES%" ""
if errorlevel 1 goto :fail

call :build "znver4"    "nnue-znver4" "%FEATURES%" "-C target-feature=+avx512f,+avx512bw,+avx512vl"
if errorlevel 1 goto :fail

call :build "znver5"    "nnue-znver5" "%FEATURES%" "-C target-feature=+avx512f,+avx512bw,+avx512vl,+avx512vnni"
if errorlevel 1 goto :fail

rem === PeSTO variant (no NNUE, older platforms) ===
call :build "x86-64" "pesto" "syzygy" ""
if errorlevel 1 goto :fail

rem === Packaging ===
echo.
echo === Packaging ===
copy /y README.md "%ZIP_DIR%\" >nul
if exist "%ZIP_NAME%.zip" del /f "%ZIP_NAME%.zip"
powershell -NoProfile -Command "Compress-Archive -Path '%ZIP_DIR%\*' -DestinationPath '%ZIP_NAME%.zip' -Force"

echo.
echo === Done ===
echo   Archive : %ROOT%\%ZIP_NAME%.zip
powershell -NoProfile -Command ^
    "Add-Type -A System.IO.Compression.FileSystem;" ^
    "$z=[IO.Compression.ZipFile]::OpenRead('%ZIP_NAME%.zip');" ^
    "$z.Entries | ForEach-Object { Write-Host ('  {0,-42} {1,6} KB' -f $_.Name, [math]::Round($_.Length/1KB)) };" ^
    "$z.Dispose()"

popd
exit /b 0

:fail
echo.
echo === FAILED ===
popd
exit /b 1


rem ===========================================================
rem Subroutine: build CPU SUFFIX FEATURES EXTRA_FLAGS
rem   Steps: instrumented -> bench -> merge -> PGO+LTO
rem ===========================================================
:build
setlocal
set "CPU=%~1"
set "SUFFIX=%~2"
set "BUILD_FEATURES=%~3"
set "EXTRA_FLAGS=%~4"

echo.
echo ==========================================
echo === Building %SUFFIX%  ^(cpu=%CPU%^)
echo ==========================================

rem MODEL= only if 'nnue' in features
set "MODEL="
echo %BUILD_FEATURES% | findstr /i "nnue" >nul
if not errorlevel 1 set "MODEL=%NET%"

rem Clean PGO directory
if exist "%PGO_DIR%" rd /s /q "%PGO_DIR%"
mkdir "%PGO_DIR%"

rem --- Step 1/4: Instrumented build ---
rem   CARGO_TARGET_..._RUSTFLAGS prevents build scripts from receiving CPU flags.
echo.
echo --- Step 1/4: Instrumented build (%CPU%) ---
set "RUSTFLAGS="
set "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=-C target-cpu=%CPU%"
if not "%EXTRA_FLAGS%"=="" set "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=!CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS! %EXTRA_FLAGS%"
set "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=!CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS! -Cprofile-generate=%PGO_DIR%"
cargo build --release --target "%BUILD_TARGET%" --features "%BUILD_FEATURES%"
if errorlevel 1 (endlocal & exit /b 1)

rem --- Step 2/4: Collect profiles ---
echo.
echo --- Step 2/4: Collecting profiles (bench) ---
"%CARGO_EXE%" bench
if errorlevel 1 (endlocal & exit /b 1)

rem --- Step 3/4: Merge profiles ---
echo.
echo --- Step 3/4: Merging profiles ---
"%LLVM_PROFDATA%" merge -o "%PGO_DIR%\merged.profdata" "%PGO_DIR%"
if errorlevel 1 (endlocal & exit /b 1)

rem --- Step 4/4: Final PGO + LTO build ---
echo.
echo --- Step 4/4: Final PGO + LTO build (%CPU%) ---
set "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=-C target-cpu=%CPU%"
if not "%EXTRA_FLAGS%"=="" set "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=!CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS! %EXTRA_FLAGS%"
set "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS=!CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS! -Cprofile-use=%PGO_DIR%\merged.profdata"
set "CARGO_PROFILE_RELEASE_LTO=fat"
set "RUSTFLAGS="
cargo build --release --target "%BUILD_TARGET%" --features "%BUILD_FEATURES%"
if errorlevel 1 (endlocal & exit /b 1)

copy /y "%CARGO_EXE%" "%ZIP_DIR%\gaiachess-%SUFFIX%.exe" >nul
echo.
echo   =^> %ZIP_DIR%\gaiachess-%SUFFIX%.exe

endlocal
exit /b 0
