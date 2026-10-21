//! Console-only presentation. Persisted metrics keep their original writers/precision.
use super::telemetry::UpdateTelemetry;
use racing_rl::rollout::Rollout;
use std::{collections::HashSet, fmt::Write};

const DIVIDER: &str = "========================================================================";
const FOOTER: &str = "------------------------------------------------------------------------";

fn field(output: &mut String, name: &str, value: impl std::fmt::Display) {
    writeln!(output, "   {:<33} {value}", format!("{name}:")).unwrap();
}

/// Called after the existing update timer stops, never inside the optimizer.
/// update_seconds covers the update and telemetry; persistence is timed separately.
pub(super) fn training_rollout(
    number: u64,
    target: usize,
    rollout: &Rollout,
    update: &UpdateTelemetry,
    update_seconds: f64,
) -> String {
    let mut output = String::new();
    let percent = if target > 0 {
        100. * number as f64 / target as f64
    } else {
        0.
    };
    writeln!(output, "\n{DIVIDER}").unwrap();
    writeln!(
        output,
        " PPO TRAINING | Rollout {number} / {target} ({percent:.1}%)"
    )
    .unwrap();
    writeln!(output, "{DIVIDER}\n").unwrap();

    let observed = rollout
        .transitions
        .iter()
        .map(|s| (s.env_id, s.episode_id))
        .collect::<HashSet<_>>()
        .len();
    let [completed, collisions, timeouts] = super::counts(rollout);
    writeln!(output, " Episodes").unwrap();
    field(&mut output, "Observed", observed);
    field(&mut output, "Terminated", completed + collisions);
    field(&mut output, "Truncated", timeouts);
    field(&mut output, "Completed", completed);
    field(&mut output, "Collisions", collisions);
    field(&mut output, "Timeouts", timeouts);

    let mut parts = [0.; 3];
    let mut total = 0.;
    for step in &rollout.transitions {
        parts[0] += f64::from(step.reward.progress);
        parts[1] += f64::from(step.reward.completion);
        parts[2] += f64::from(step.reward.collision);
        total += f64::from(step.reward.total());
    }
    writeln!(output, "\n Rewards").unwrap();
    for (name, value) in ["Total", "Progress", "Completion", "Collision"]
        .into_iter()
        .zip([total, parts[0], parts[1], parts[2]])
    {
        field(&mut output, name, format_args!("{value:+.4}"));
    }

    writeln!(output, "\n PPO Update").unwrap();
    for (name, value) in update.console_lines() {
        field(&mut output, name, value);
    }
    writeln!(
        output,
        "   KL/clip/loss stats include the KL-rejected minibatch."
    )
    .unwrap();
    writeln!(
        output,
        "   Gradient norms are pre-clip, for applied updates only."
    )
    .unwrap();

    writeln!(output, "\n Simulation").unwrap();
    field(&mut output, "Transitions", rollout.transitions.len());
    field(&mut output, "Decisions", rollout.transitions.len());
    field(
        &mut output,
        "Physical ticks",
        rollout
            .transitions
            .iter()
            .map(|s| s.physical_ticks)
            .sum::<u64>(),
    );
    field(&mut output, "Rendered frames", 0);
    let timings = &rollout.timings;
    if timings.total_seconds > 0. {
        field(
            &mut output,
            "Collection throughput",
            format_args!(
                "{:.2} transitions/s",
                rollout.transitions.len() as f64 / timings.total_seconds
            ),
        );
    } else {
        field(&mut output, "Collection throughput", "N/A transitions/s");
    }

    writeln!(output, "\n Timings").unwrap();
    for (name, seconds) in [
        ("Observation", timings.observations_seconds),
        ("Upload", timings.upload_seconds),
        ("Inference", timings.inference_seconds),
        ("Download", timings.download_seconds),
        ("Simulation", timings.simulation_seconds),
        ("Preparation", timings.preparation_seconds),
        ("Collection total", timings.total_seconds),
        ("PPO update total", update_seconds),
        (
            "Overall (collection + update)",
            timings.total_seconds + update_seconds,
        ),
    ] {
        field(&mut output, name, format_args!("{seconds:.4}s"));
    }
    writeln!(
        output,
        "   Update total includes telemetry and loss-file writes."
    )
    .unwrap();
    writeln!(output, "{FOOTER}").unwrap();
    output
}

#[cfg(test)]
mod tests;
