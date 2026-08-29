//! Append the integrity footer to an existing exported network.
//!
//! Re-publication tool for networks exported before the footer existed
//! (the HuggingFace repo is append-only, so a re-stamped network gets a new
//! name). Refuses a file that is not exactly the bare payload size.
//!
//! Usage:
//!   cargo run --release --bin stamp -- <in.bin> <out.bin>

#[path = "../threats.rs"]
mod threats;
#[path = "../save_format.rs"]
mod save_format;

#[allow(dead_code)] // referenced by the #[path]-included save_format module
const L1_SIZE: usize = 640;
#[allow(dead_code)]
const L2_SIZE: usize = 16;
#[allow(dead_code)]
const L3_SIZE: usize = 32;
#[allow(dead_code)]
const NUM_OUTPUT_BUCKETS: usize = 8;
#[allow(dead_code)]
const EVAL_SCALE: f32 = 287.0;
#[allow(dead_code)]
const TOTAL_THREATS: usize = threats::TOTAL_THREATS;

#[allow(dead_code)]
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    11, 11, 11, 11,
    11, 11, 11, 11,
    11, 11, 11, 11,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: stamp <in.bin> <out.bin>");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", args[1]);
        std::process::exit(1);
    });
    if data.len() != save_format::NNUE_FILE_SIZE {
        eprintln!(
            "{} is {} bytes, expected the bare payload of {} — already stamped or not a GaiaNet-T1 network",
            args[1],
            data.len(),
            save_format::NNUE_FILE_SIZE,
        );
        std::process::exit(1);
    }
    let footer = save_format::footer(&data);
    let mut out = data;
    out.extend_from_slice(&footer);
    std::fs::write(&args[2], &out).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", args[2]);
        std::process::exit(1);
    });
    println!(
        "{}: {} bytes, arch {:#010x}, content {:#018x}",
        args[2],
        out.len(),
        save_format::ARCH_HASH,
        u64::from_le_bytes(footer[4..12].try_into().unwrap()),
    );
}
