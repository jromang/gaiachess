fn main() {
    if let Ok(model_path) = std::env::var("MODEL") {
        let raw = std::fs::read(&model_path)
            .unwrap_or_else(|e| panic!("Failed to read MODEL={model_path}: {e}"));

        let compressed =
            zstd::encode_all(&raw[..], 22).expect("Failed to compress NNUE with zstd");

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_path = format!("{out_dir}/model.nnue.zst");
        std::fs::write(&out_path, &compressed).unwrap();

        println!("cargo:rustc-env=MODEL_ZST={out_path}");
        println!("cargo:rerun-if-changed={model_path}");

        let ratio = compressed.len() as f64 / raw.len() as f64 * 100.0;
        eprintln!(
            "NNUE: {:.2} MB -> {:.2} MB ({ratio:.1}%, zstd-22)",
            raw.len() as f64 / 1024.0 / 1024.0,
            compressed.len() as f64 / 1024.0 / 1024.0,
        );
    }
    println!("cargo:rerun-if-env-changed=MODEL");

    // GaiaTB (DTM tablebases, 3+4 pieces)
    #[cfg(feature = "gaiatb")]
    {
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
                    "https://huggingface.co/datasets/jromanghf/gaiatb-tb34/resolve/main/tb34.gtpk?download=true",
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
        println!("cargo:rerun-if-changed={blob_path}");
        let size = std::fs::metadata(&blob_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("GaiaTB: {:.1} MB embedded", size as f64 / 1024.0 / 1024.0);
    }
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

    // Rust compiler version
    if let Ok(out) = std::process::Command::new("rustc").arg("--version").output() {
        let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
        println!("cargo:rustc-env=RUSTC_VERSION={ver}");
    }

    // Git commit hash
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
