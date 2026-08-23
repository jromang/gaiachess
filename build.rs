fn main() {
    // An empty MODEL is treated as no MODEL: a matrix build that leaves the variable
    // set but blank means "no network", not "read the file called ''".
    if let Ok(model_path) = std::env::var("MODEL").map(|p| p.trim().to_string())
        && !model_path.is_empty()
    {
        let raw = std::fs::read(&model_path)
            .unwrap_or_else(|e| panic!("Failed to read MODEL={model_path}: {e}"));

        let compressed =
            zstd::encode_all(&raw[..], 22).expect("Failed to compress NNUE with zstd");

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_path = format!("{out_dir}/model.nnue.zst");
        std::fs::write(&out_path, &compressed).unwrap();

        println!("cargo:rustc-env=MODEL_ZST={out_path}");
        // Whether a network is baked into the binary is a property of the build, not of
        // the feature set: the engine can be compiled with the NNUE code and no weights,
        // which is what a build that receives them at run time wants.
        println!("cargo:rustc-cfg=net_embedded");
        println!("cargo:rerun-if-changed={model_path}");

        let ratio = compressed.len() as f64 / raw.len() as f64 * 100.0;
        eprintln!(
            "NNUE: {:.2} MB -> {:.2} MB ({ratio:.1}%, zstd-22)",
            raw.len() as f64 / 1024.0 / 1024.0,
            compressed.len() as f64 / 1024.0 / 1024.0,
        );
    }
    println!("cargo:rustc-check-cfg=cfg(net_embedded)");
    // Distribution builds pass `--cfg gaia_dist` in RUSTFLAGS: slider attacks
    // are then elected at runtime (PEXT vs AVX2/magic, AVX-512 setwise) so one
    // binary serves every CPU. Without it, attack selection stays compile-time
    // — local, OpenBench and test builds keep today's branchless codegen.
    println!("cargo:rustc-check-cfg=cfg(gaia_dist)");
    println!("cargo:rerun-if-env-changed=MODEL");

    // GaiaTB (DTM tablebases, 3+4 pieces)
    //
    // Like the network, whether the blob is baked into the binary is a property of the
    // build, not of the feature set: a browser build carries the probing code and
    // receives the blob from its host at run time, because 30 MB inside the module
    // would ship on every code change and defeat the two-module split.
    #[cfg(feature = "gaiatb")]
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("wasm32") {
        let blob_path = std::env::var("GAIATB_BLOB")
            .unwrap_or_else(|_| "tb/tb34.gtpk".to_string());

        // Download from Hugging Face when the blob is not present locally
        if !std::path::Path::new(&blob_path).exists() {
            eprintln!("GaiaTB: {blob_path} not found, downloading from HuggingFace...");
            if let Some(parent) = std::path::Path::new(&blob_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let status = std::process::Command::new("curl")
                .args([
                    "-L", "--fail", "--progress-bar",
                    "-o", &blob_path,
                    // Versioned name, append-only on the dataset side: the file this
                    // code parses can never be updated out from under it. tb34.gtpk
                    // (unsuffixed) is the GTPK v1 blob the pre-calibration engines read.
                    "https://huggingface.co/datasets/jromanghf/gaiatb-tb34/resolve/main/tb34-v2.gtpk?download=true",
                ])
                .status()
                .expect("Failed to run curl — is it installed?");
            if !status.success() {
                panic!("Failed to download GaiaTB blob (curl exit {})", status);
            }
        }

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_path = format!("{out_dir}/tb34.gtpk");
        std::fs::copy(&blob_path, &out_path)
            .unwrap_or_else(|e| panic!("Failed to copy GAIATB_BLOB={blob_path}: {e}"));
        println!("cargo:rustc-env=GAIATB_ZST={out_path}");
        println!("cargo:rustc-cfg=gaiatb_embedded");
        println!("cargo:rerun-if-changed={blob_path}");
        let size = std::fs::metadata(&blob_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("GaiaTB: {:.1} MB embedded", size as f64 / 1024.0 / 1024.0);
    }
    println!("cargo:rustc-check-cfg=cfg(gaiatb_embedded)");
    println!("cargo:rerun-if-env-changed=GAIATB_BLOB");

    // Pyrrhic (Syzygy tablebase probing)
    // tbprobe.c uses <stdatomic.h> (C11 atomics) which MSVC 14.50 doesn't expose
    // cleanly. Use clang — handles C11 atomics natively on all platforms.
    #[cfg(feature = "syzygy")]
    {
        cc::Build::new()
            .compiler("clang")
            .file("src/pyrrhic/tbprobe.c")
            .include("src/pyrrhic")
            .define("_CRT_SECURE_NO_WARNINGS", None)
            .flag("-std=c11")
            .flag("-Wno-deprecated-declarations")
            .flag("-Wno-sign-compare")
            .compile("pyrrhic");
        println!("cargo:rerun-if-changed=src/pyrrhic/tbprobe.c");
        println!("cargo:rerun-if-changed=src/pyrrhic/tbconfig.h");
    }

    // Haiku native shim
    // miniquad has no Haiku backend, so the window, input and audio are a few hundred
    // lines of Be API C++ instead, compiled here and spoken to over a C ABI. Only for a
    // GUI build targeting Haiku; the engine alone stays pure Rust.
    // GAIA_SKIP_SHIM=1 skips the C++ so a cross `cargo check -Zbuild-std` can
    // type-check the Rust side from a machine with no Be headers; check never
    // links, so the missing objects cost nothing.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("haiku")
        && std::env::var("CARGO_FEATURE_GUI").is_ok()
        && std::env::var("GAIA_SKIP_SHIM").as_deref() != Ok("1")
    {
        cc::Build::new()
            .cpp(true)
            .file("src/gui/haiku_shim.cpp")
            .flag("-std=c++17")
            .compile("gaia_haiku_shim");
        // libbe: BApplication, BWindow, BBitmap. libmedia: BSoundPlayer.
        println!("cargo:rustc-link-lib=be");
        println!("cargo:rustc-link-lib=media");
        println!("cargo:rerun-if-changed=src/gui/haiku_shim.cpp");
    }

    // Windows icon
    // Explorer, the taskbar and a shortcut all read the icon off the executable itself,
    // long before anything of ours runs, so the drawings the interface embeds are also
    // linked in as a resource. Only for the MSVC toolchain, which the released Windows
    // binaries are built with: a mingw link would want the same thing through windres.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let icon_path = "src/gui/assets/icon.ico";
        let icon = std::fs::read(icon_path)
            .unwrap_or_else(|e| panic!("Failed to read {icon_path}: {e}"));
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let res_path = format!("{out_dir}/icon.res");
        std::fs::write(&res_path, icon_resource(&icon)).unwrap();
        println!("cargo:rustc-link-arg-bins={res_path}");
        println!("cargo:rerun-if-changed={icon_path}");
    }

    // Rust compiler version
    if let Ok(out) = std::process::Command::new("rustc").arg("--version").output() {
        let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
        println!("cargo:rustc-env=RUSTC_VERSION={ver}");
    }

    // Git commit hash
    //
    // The files git rewrites when the checked-out commit changes are watched, not just
    // read. Without that cargo has no reason to run this script again -- the first
    // `rerun-if` directive above turns off its "any file in the package" default -- so
    // the hash stays baked at whatever it was the first time, and `gaiachess info` goes
    // on naming a commit the binary was not built from. That is the one claim about
    // itself a release must not get wrong. The price is that committing invalidates the
    // build: the next `cargo build` recompiles the crate.
    for path in git_tip_files() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    if let Ok(out) = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output() {
        let hash = String::from_utf8_lossy(&out.stdout).trim().to_string();
        println!("cargo:rustc-env=GIT_HASH={hash}");
    }

    // Build date
    if let Ok(out) = std::process::Command::new("date").arg("+%Y-%m-%d").output() {
        let date = String::from_utf8_lossy(&out.stdout).trim().to_string();
        println!("cargo:rustc-env=BUILD_DATE={date}");
    }
}

/// The files git rewrites when the checked-out commit changes: `HEAD` itself, the ref it
/// names while that ref is still loose, and `packed-refs` for once it has been packed
/// away.
///
/// Their paths are asked of git rather than assumed to sit under `.git/`: this crate is
/// also built inside throwaway worktrees (`tools/sprt/tournament.py`), where `.git` is a
/// file pointing at the real directory. Only paths that exist are returned -- cargo
/// counts a watched file that is not there as changed, which would run this script on
/// every single build instead of on every commit.
fn git_tip_files() -> Vec<std::path::PathBuf> {
    let git_path = |arg: &str| -> Option<std::path::PathBuf> {
        let out = std::process::Command::new("git")
            .args(["rev-parse", "--git-path", arg])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| std::path::PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
    };
    let mut watched: Vec<std::path::PathBuf> =
        ["HEAD", "packed-refs"].iter().filter_map(|arg| git_path(arg)).collect();
    // The branch's own file. A detached HEAD has none: it carries the hash itself, and
    // `HEAD` is then the only file that moves.
    if let Ok(out) = std::process::Command::new("git")
        .args(["symbolic-ref", "--quiet", "HEAD"])
        .output()
        && out.status.success()
    {
        let refname = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if let Some(path) = git_path(&refname) {
            watched.push(path);
        }
    }
    watched.retain(|path| path.exists());
    watched
}

/// Repacks an .ico into the RES file a linker takes resources in.
///
/// Both are a directory in front of a list of images, and the images cross over
/// untouched. Only the directory changes: an .ico points at its images by their offset
/// in the file, a binary points at them by resource number, so each image becomes an
/// icon resource numbered from one and the directory becomes a group naming them.
/// Windows shows the lowest-numbered group as the file's icon, hence group one.
///
/// Written out here rather than handed to rc.exe, which lives inside a Windows SDK, or
/// to a crate: the format has not moved since 1993 and this is the whole of it.
fn icon_resource(ico: &[u8]) -> Vec<u8> {
    /// RT_ICON, one image.
    const ICON: u16 = 3;
    /// RT_GROUP_ICON, the directory naming them.
    const GROUP: u16 = 14;
    /// Entry size in both directories, minus the four bytes they disagree on.
    const SHARED: usize = 12;

    let word = |at: usize| u16::from_le_bytes([ico[at], ico[at + 1]]);
    let dword = |at: usize| u32::from_le_bytes([ico[at], ico[at + 1], ico[at + 2], ico[at + 3]]);
    assert_eq!((word(0), word(2)), (0, 1), "not an icon file");
    let count = word(4) as usize;

    // A RES file opens on an empty resource — no type, no name, no data. That is how a
    // reader tells this format from the 16-bit one it grew out of.
    let mut res = vec![0u8; 32];
    res[4] = 32;
    res[8..12].copy_from_slice(&[0xFF, 0xFF, 0, 0]);
    res[12..16].copy_from_slice(&[0xFF, 0xFF, 0, 0]);

    let mut group = Vec::from(&ico[0..6]);
    for i in 0..count {
        let entry = 6 + (SHARED + 4) * i;
        let size = dword(entry + 8) as usize;
        let offset = dword(entry + 12) as usize;
        let id = i as u16 + 1;
        push_resource(&mut res, ICON, id, 0x1010, &ico[offset..offset + size]);
        // Same entry, with the number that replaces the offset.
        group.extend_from_slice(&ico[entry..entry + SHARED]);
        group.extend_from_slice(&id.to_le_bytes());
    }
    push_resource(&mut res, GROUP, 1, 0x1030, &group);
    res
}

/// Appends one resource: a fixed header, then the data.
///
/// Type and name are numbers rather than strings, which is what the leading `0xFFFF`
/// announces and what keeps the header a round 32 bytes. `memory` is the loading hint
/// 16-bit Windows acted on and every toolchain still writes out of habit.
fn push_resource(res: &mut Vec<u8>, kind: u16, id: u16, memory: u16, data: &[u8]) {
    res.extend_from_slice(&(data.len() as u32).to_le_bytes());
    res.extend_from_slice(&32u32.to_le_bytes());
    for number in [kind, id] {
        res.extend_from_slice(&u16::MAX.to_le_bytes());
        res.extend_from_slice(&number.to_le_bytes());
    }
    res.extend_from_slice(&0u32.to_le_bytes()); // data version
    res.extend_from_slice(&memory.to_le_bytes());
    res.extend_from_slice(&0x0409u16.to_le_bytes()); // language, as every toolchain emits
    res.extend_from_slice(&0u32.to_le_bytes()); // version
    res.extend_from_slice(&0u32.to_le_bytes()); // characteristics
    res.extend_from_slice(data);
    // Every resource starts on a four-byte boundary.
    res.resize(res.len().next_multiple_of(4), 0);
}
