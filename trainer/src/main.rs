//! GaiaChess NNUE Trainer v5 — GaiaNet-T1 (filtered threats).
//!
//! Architecture: 51,256 inputs (768 factorised PST + 41,272 threats + 768×12 bucketed PST)
//!   → FT 640 → CReLU pairwise → L1 16 → dual act [x, x²] → L2 32 → L3 (skip) × 8 buckets.
//!
//! Differences vs the previous trainer generation:
//!   - GaiaNet-T1 filtered threats (41,272 instead of the full 79,856), tables identical to the engine
//!   - l0 clipping ±0.49: threat weights are quantized i8(×255) → |w| ≤ 127/255 ≈ 0.498
//!     (lesson learned: ±0.99 saturates 50% of the threat weights at quantization)
//!   - l1 clipping ±0.98 (i8 ×64)
//!
//! process_net converts raw.bin (f32 Bullet) → gaia.bin (engine binary, 38,330,144 bytes).

mod threats;
mod save_format;
mod filter;

use bullet_lib::{
    game::outputs::MaterialCount,
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::{DirectSequentialDataLoader, SfBinpackLoader, ViriBinpackLoader, viribinpack::ViriFilter}},
};

use bullet_lib::game::inputs::SparseInputType;
use threats::GaiaNetT1Inputs;

// Architecture constants
pub const L1_SIZE: usize = 640;
pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 32;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
pub const TOTAL_THREATS: usize = threats::TOTAL_THREATS;

/// King buckets (32 half-board entries, expanded with mirroring) — IDENTICAL to src/nnue/features.rs.
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

/// Default clip (f32 layers: l2, l3) — standard Bullet values.
const ADAMW_PARAMS: AdamWParams = AdamWParams {
    decay: 0.01,
    beta1: 0.9,
    beta2: 0.999,
    min_weight: -0.99,
    max_weight: 0.99,
};

/// Data format detected from file extensions.
#[derive(Clone, Copy, PartialEq)]
enum DataFormat { SfBinpack, ViriFormat, BulletFormat }

macro_rules! run_training {
    ($trainer:expr, $schedule:expr, $settings:expr, $paths:expr, $format:expr, $filter:expr) => {
        match $format {
            DataFormat::SfBinpack => {
                let dataloader = SfBinpackLoader::new_concat_multiple(
                    $paths, 512, $settings.threads, $filter,
                );
                $trainer.run(&$schedule, $settings, &dataloader);
            }
            DataFormat::ViriFormat => {
                let dataloader = ViriBinpackLoader::new_concat_multiple(
                    $paths, 8192, 4,
                    ViriFilter::Custom(|_board, _mv, _eval, _wdl| true),
                );
                $trainer.run(&$schedule, $settings, &dataloader);
            }
            DataFormat::BulletFormat => {
                let dataloader = DirectSequentialDataLoader::new($paths);
                $trainer.run(&$schedule, $settings, &dataloader);
            }
        }
    };
}

fn run_process_net(checkpoint: &str) {
    let raw_path = format!("{checkpoint}/raw.bin");
    let out_path = format!("{checkpoint}/gaia.bin");
    println!("process_net: {raw_path} → {out_path}");
    let data = save_format::process_net(&raw_path).expect("process_net failed");
    std::fs::write(&out_path, &data).expect("Failed to write gaia.bin");
    println!("process_net: wrote {} bytes", data.len());
}

fn run_phase1(
    net_id: &str,
    total_inputs: usize,
    superbatches: usize,
    start_sb: usize,
    initial_lr: f32,
    threads: usize,
    paths: &[&str],
    format: DataFormat,
) -> String {
    let final_lr = initial_lr * 0.3f32.powi(3);
    let sf = save_format::build_save_format();

    println!("=== Phase 1: {net_id} ({superbatches} SB, start={start_sb}) ===");

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(GaiaNetT1Inputs::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&sf)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", total_inputs, L1_SIZE);
            let l1 = builder.new_affine("l1", L1_SIZE, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, NUM_OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE + 2 * L2_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let pairwise_out = stm_subnet.concat(ntm_subnet);

            let l1_out = l1.forward(pairwise_out).select(buckets);
            let l1_out = l1_out.concat(l1_out.abs_pow(2.0));
            let l1_out = l1_out.crelu();
            let l2_out = l2.forward(l1_out).select(buckets).screlu();
            l3.forward(l2_out.concat(l1_out)).select(buckets)
        });

    trainer.optimiser.set_params(ADAMW_PARAMS);

    // l0 (FT): threat weights are quantized i8(×255) → clip ±0.49 is mandatory.
    // PST weights = factorised + bucketed (summed), each ±0.49 → effective range ±0.98,
    // equivalent to the previous global ±0.99 clip on a single tensor.
    let l0_clip = AdamWParams { min_weight: -0.49, max_weight: 0.49, ..ADAMW_PARAMS };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0b", l0_clip);

    // l1: quantized i8(×64) → |w| ≤ 127/64 ≈ 1.98, clip ±0.98 (reference range;
    // lesson learned: without a clip the weights grow to 2× that range)
    let l1_clip = AdamWParams { min_weight: -0.98, max_weight: 0.98, ..ADAMW_PARAMS };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 512,
    };

    if start_sb > 1 {
        let checkpoint = format!("checkpoints/{net_id}-{}", start_sb - 1);
        println!("Resuming from checkpoint: {checkpoint}");
        trainer.load_from_checkpoint(&checkpoint);

        let remaining_sb = superbatches - (start_sb - 1);
        let schedule = TrainingSchedule {
            net_id: net_id.to_string(),
            eval_scale: 287.0,
            steps: TrainingSteps {
                batch_size: 16_384,
                batches_per_superbatch: 6104,
                start_superbatch: start_sb,
                end_superbatch: superbatches,
            },
            wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
            lr_scheduler: lr::Sequence {
                first: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: start_sb - 1 },
                second: lr::Warmup {
                    inner: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: remaining_sb },
                    warmup_batches: 500,
                },
                first_scheduler_final_superbatch: start_sb - 1,
            },
            save_rate: 20,
        };
        run_training!(trainer, schedule, &settings, paths, format, filter::filter_phase1);
    } else {
        let schedule = TrainingSchedule {
            net_id: net_id.to_string(),
            eval_scale: 287.0,
            steps: TrainingSteps {
                batch_size: 16_384,
                batches_per_superbatch: 6104,
                start_superbatch: 1,
                end_superbatch: superbatches,
            },
            wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
            lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
            save_rate: 20,
        };
        run_training!(trainer, schedule, &settings, paths, format, filter::filter_phase1);
    }

    let checkpoint = format!("checkpoints/{net_id}-{superbatches}");
    run_process_net(&checkpoint);
    checkpoint
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut net_id: Option<String> = None;
    let mut superbatches = 1000usize;
    let mut start_sb = 1usize;
    let mut initial_lr = 0.001f32;
    let mut threads = 6usize;
    let mut process_net_path: Option<String> = None;
    let mut dataset_paths: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--process-net" => { i += 1; process_net_path = Some(args[i].clone()); }
            "--net-id" => { i += 1; net_id = Some(args[i].clone()); }
            "--start" => { i += 1; start_sb = args[i].parse().unwrap(); }
            "--lr" => { i += 1; initial_lr = args[i].parse().unwrap(); }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            arg => {
                if let Ok(n) = arg.parse::<usize>() {
                    superbatches = n;
                } else {
                    dataset_paths.push(arg.to_string());
                }
            }
        }
        i += 1;
    }

    if let Some(checkpoint) = process_net_path {
        run_process_net(&checkpoint);
        return;
    }

    let net_id = net_id.expect("--net-id required");
    assert!(!dataset_paths.is_empty(), "At least one dataset path required");

    let format = if dataset_paths.iter().all(|p| p.ends_with(".binpack")) {
        DataFormat::SfBinpack
    } else if dataset_paths.iter().all(|p| p.ends_with(".vf")) {
        DataFormat::ViriFormat
    } else {
        DataFormat::BulletFormat
    };
    let input_type = GaiaNetT1Inputs::new(BUCKET_LAYOUT);
    let total_inputs = input_type.num_inputs();

    let format_name = match format {
        DataFormat::SfBinpack => "binpack (sfbinpack)",
        DataFormat::ViriFormat => "Viriformat",
        DataFormat::BulletFormat => "Bulletformat",
    };
    println!("Net ID: {net_id}");
    println!("Trainer: v5 (GaiaNet-T1, 41,272 filtered threats, batch=16384)");
    println!("Format: {format_name}");
    println!("Architecture: {total_inputs} inputs, eval_scale=287, clip l0 ±0.49 / l1 ±0.98");

    let paths_ref: Vec<&str> = dataset_paths.iter().map(|s| s.as_str()).collect();

    run_phase1(&net_id, total_inputs, superbatches, start_sb, initial_lr, threads, &paths_ref, format);
}

#[cfg(test)]
mod integration_tests {
    use super::save_format;

    #[test]
    fn test_process_net_on_file() {
        let path = "/tmp/test_raw.bin";
        if !std::path::Path::new(path).exists() { return; }
        let data = save_format::process_net(path).unwrap();
        assert_eq!(data.len(), save_format::NNUE_FILE_SIZE);
    }
}
