//! Bounded headless generation/search and replay/export of the exact search artifacts.
use super::*;
use clap::Subcommand;
use std::{fs, path::PathBuf};

#[derive(Subcommand)]
pub enum AcoCommand {
    /// Generate valid training circuits without a pilot. No PPO training is started.
    Generate {
        #[arg(long, default_value_t = 1)]
        count: usize,
        #[arg(long, default_value_t = 47)]
        seed: u64,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    /// Search using MMAS, optionally against a frozen PPO checkpoint on CPU.
    Search {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 47)]
        seed: u64,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    /// Validate and summarize a saved search; visual replay uses the unified app.
    Replay { file: PathBuf },
    /// Export selected candidates as training TrackDefinition + provenance artifacts.
    Export {
        file: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

pub fn run(command: AcoCommand) -> GenerationResult<()> {
    match command {
        AcoCommand::Generate {
            count,
            seed,
            config,
            output,
            dry_run,
        } => {
            if count == 0 || count > 10_000 {
                return Err("count must be 1..10000".into());
            }
            let config: GeneratorConfig = match config {
                Some(p) => procedural::read_ron(&p)?,
                None => Default::default(),
            };
            config.validate()?;
            if output.exists() {
                return Err(format!("output {} already exists", output.display()));
            }
            if dry_run {
                println!(
                    "DRY RUN: generate {count} training-only tracks, seed={seed}, output={}",
                    output.display()
                );
                return Ok(());
            }
            fs::create_dir_all(&output).map_err(|e| e.to_string())?;
            procedural::write_new_ron(&output.join("generator.ron"), &config)?;
            let mut seen = BTreeSet::new();
            for i in 0..count {
                let generated = procedural::generate(&config, seed.wrapping_add(i as u64))?;
                if !seen.insert(generated.provenance.geometry_hash.clone()) {
                    return Err("duplicate generation geometry; choose another seed/config".into());
                }
                let id = &generated.definition.id;
                generated.save(output.join(format!("{id}.generated.ron")))?;
                generated.export_definition(output.join(format!("{id}.ron")))?;
                println!(
                    "{id}: length={:.1} curvature={:.5} seed={}",
                    generated.geometry.length,
                    generated.geometry.max_curvature,
                    generated.provenance.seed
                );
            }
            Ok(())
        }
        AcoCommand::Search {
            config,
            seed,
            checkpoint,
            output,
            dry_run,
        } => {
            let mut config: AcoConfig = match config {
                Some(p) => procedural::read_ron(&p)?,
                None => Default::default(),
            };
            if let Some(path) = &checkpoint {
                if !path.is_file() {
                    return Err(format!("checkpoint {} does not exist", path.display()));
                }
                config.policy_label = Some(checkpoint_identity(path)?);
            } else if config.policy_label.is_some() {
                return Err("policy_label without --checkpoint would fabricate policy evaluation provenance".into());
            }
            config.validate()?;
            if output.exists() {
                return Err(format!("output {} already exists", output.display()));
            }
            if dry_run {
                println!(
                    "DRY RUN: {:?}, seed={seed}, at most {} candidates/{} decisions, backend=CPU, output={}",
                    config.strategy,
                    config.max_candidates,
                    config.max_evaluation_steps,
                    output.display()
                );
                return Ok(());
            }
            fs::create_dir_all(&output).map_err(|e| e.to_string())?;
            let mut search = SearchArchive::new(config, seed)?;
            if let Some(path) = checkpoint {
                let mut evaluator = crate::rl::curriculum::checkpoint_evaluator(&path)?;
                while !search.is_finished() {
                    search.run_iteration(Some(&mut evaluator))?;
                }
            } else {
                while !search.is_finished() {
                    search.run_iteration(None)?;
                }
            }
            search.save(output.join("search.ron"))?;
            search.export_selected(output.join("tracks"))?;
            print_summary(&search);
            Ok(())
        }
        AcoCommand::Replay { file } => {
            let search = SearchArchive::load(file)?;
            print_summary(&search);
            Ok(())
        }
        AcoCommand::Export { file, output } => {
            let search = SearchArchive::load(file)?;
            let ids = search.export_selected(&output)?;
            println!(
                "exported {} training circuits to {}",
                ids.len(),
                output.display()
            );
            Ok(())
        }
    }
}
fn print_summary(search: &SearchArchive) {
    println!(
        "ACO iterations={} candidates={} evaluation_budget_charged={} selected={} policy={:?}",
        search.state.iteration,
        search.state.candidates,
        search.state.evaluation_steps,
        search.state.selected.len(),
        search.config.policy_label
    );
    for iteration in &search.iterations {
        let rejected = iteration
            .ants
            .iter()
            .filter(|a| a.rejection.is_some())
            .count();
        println!(
            "iteration={} mean_fitness={:.4} diversity={:.4} rejected={rejected} best={:?}",
            iteration.iteration, iteration.mean_fitness, iteration.diversity, iteration.best_global
        );
    }
}

fn checkpoint_identity(path: &std::path::Path) -> GenerationResult<String> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut buffer = [0u8; 65536];
    let mut hash = 0xcbf29ce484222325u64;
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        for byte in &buffer[..count] {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{}#fnv1a64={hash:016x}", path.display()))
}
