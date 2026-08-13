//! Trainer ↔ engine forward parity (post-training): evaluates reference FENs
//! with the Bullet graph (f32) loaded from a checkpoint, and prints the evals
//! in centipawns (STM POV).
//!
//! Compare against `position fen X` + `eval` on the engine side (gaia.bin net
//! from the same checkpoint, loaded via EvalFile or embedded via MODEL=).
//!
//! Usage:
//!   cargo run --release --bin parity -- checkpoints/gaianet-t1-1000

#[path = "../threats.rs"]
mod threats;
#[path = "../save_format.rs"]
mod save_format;

use bullet_lib::{
    game::{inputs::SparseInputType, outputs::MaterialCount},
    nn::optimiser::AdamW,
    value::ValueTrainerBuilder,
};
use threats::GaiaNetT1Inputs;

const L1_SIZE: usize = 640;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 8;
const EVAL_SCALE: f32 = 287.0;
#[allow(dead_code)] // referenced by the #[path]-included save_format module
const TOTAL_THREATS: usize = threats::TOTAL_THREATS;

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

/// Coverage FENs: white/black to move, mirroring, varied material buckets.
const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 6 6",
    "1k1r3r/ppq2ppp/2pb1n2/8/3P4/2N1PN2/PP3PPP/1KR2B1R w - - 4 15",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/5pk1/4p1p1/3p3p/3P3P/4P1P1/4KP2/8 w - - 0 40",
    "8/8/8/4k3/8/4P3/4K3/8 w - - 0 60",
    "8/8/3k4/8/8/2Q1K3/8/8 w - - 0 50",
    "7k/8/5q2/8/8/2Q5/8/K7 w - - 0 30",
    "8/8/4k3/8/8/3Q1K2/8/8 b - - 0 50",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: parity <checkpoint_dir>");
        std::process::exit(1);
    }

    let input_type = GaiaNetT1Inputs::new(BUCKET_LAYOUT);
    let total_inputs = input_type.num_inputs();

    // Graph IDENTICAL to main.rs run_phase1 (loading fails if the shapes differ).
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(input_type)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&save_format::build_save_format())
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

    trainer.load_from_checkpoint(&args[1]);

    println!("fen;raw;cp_stm");
    for fen in FENS {
        let raw = trainer.eval(fen);
        let cp = raw * EVAL_SCALE;
        println!("{fen};{raw:.6};{cp:.2}");
    }
}
