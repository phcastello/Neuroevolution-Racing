//! Small per-rollout summaries of the existing trainer diagnostics.
//! No tensors, gradients, rollout samples, or training state are retained here.
use std::io::{self, Write};

use racing_rl::{Result, config::Config, ensure, trainer::UpdateMetrics};

use crate::simulation::rl_environment::TrackMetrics;

#[derive(Debug, Default)]
struct Statistics {
    count: usize,
    sum: f64,
    first: Option<f64>,
    last: Option<f64>,
    maximum: Option<f64>,
}

impl Statistics {
    fn push(&mut self, value: f32) -> Result<()> {
        ensure(value.is_finite(), "non-finite update telemetry")?;
        let value = f64::from(value);
        self.count += 1;
        self.sum += value;
        self.first.get_or_insert(value);
        self.last = Some(value);
        self.maximum = Some(self.maximum.map_or(value, |previous| previous.max(value)));
        Ok(())
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }
}

/// Loss/KL/clip summaries include the rejected minibatch when early stopping occurs.
/// Gradient summaries include only minibatches that actually applied an Adam update.
#[derive(Debug)]
pub(super) struct UpdateTelemetry {
    updates_applied: u64,
    updates_possible: usize,
    minibatches_per_epoch: usize,
    optimizer_step_before: u64,
    optimizer_step: u64,
    early_stop: Option<(usize, usize)>,
    kl: Statistics,
    kl_after_first_update: Option<f64>,
    clip_fraction: Statistics,
    gradient_norm: Statistics,
    gradients_clipped: usize,
    policy_loss: Statistics,
    value_loss: Statistics,
    total_loss: Statistics,
    latent_entropy: Statistics,
    log_std: [f32; 2],
    latent_std: [f32; 2],
}

impl UpdateTelemetry {
    pub(super) fn from_metrics(
        metrics: &[UpdateMetrics],
        config: &Config,
        optimizer_steps_before: u64,
        optimizer_steps_after: u64,
        log_std: [f32; 2],
    ) -> Result<Self> {
        ensure(
            config.minibatch_size > 0 && config.update_epochs > 0 && config.rollout_size() > 0,
            "invalid update telemetry budget",
        )?;
        let minibatches_per_epoch = config.rollout_size().div_ceil(config.minibatch_size);
        let updates_possible = minibatches_per_epoch
            .checked_mul(config.update_epochs)
            .ok_or_else(|| racing_rl::Error::Invalid("update telemetry budget overflow".into()))?;
        ensure(
            metrics.len() <= updates_possible,
            "update telemetry exceeds budget",
        )?;
        let updates_applied = optimizer_steps_after
            .checked_sub(optimizer_steps_before)
            .ok_or_else(|| racing_rl::Error::Invalid("optimizer step moved backwards".into()))?;
        let latent_std = log_std.map(f32::exp);
        ensure(
            log_std.into_iter().chain(latent_std).all(f32::is_finite),
            "non-finite policy standard deviation telemetry",
        )?;
        let mut summary = Self {
            updates_applied,
            updates_possible,
            minibatches_per_epoch,
            optimizer_step_before: optimizer_steps_before,
            optimizer_step: optimizer_steps_after,
            early_stop: None,
            kl: Statistics::default(),
            // Evaluated on the next shuffled minibatch, before the second update.
            kl_after_first_update: metrics
                .first()
                .filter(|first| !first.stopped_by_kl)
                .and_then(|_| metrics.get(1))
                .map(|second| f64::from(second.loss.approximate_kl)),
            clip_fraction: Statistics::default(),
            gradient_norm: Statistics::default(),
            gradients_clipped: 0,
            policy_loss: Statistics::default(),
            value_loss: Statistics::default(),
            total_loss: Statistics::default(),
            latent_entropy: Statistics::default(),
            log_std,
            latent_std,
        };
        let mut expected_step = optimizer_steps_before;
        for (index, metric) in metrics.iter().enumerate() {
            summary.kl.push(metric.loss.approximate_kl)?;
            summary.clip_fraction.push(metric.loss.clip_fraction)?;
            summary.policy_loss.push(metric.loss.policy_loss)?;
            summary.value_loss.push(metric.loss.value_loss)?;
            summary.total_loss.push(metric.loss.total_loss)?;
            summary.latent_entropy.push(metric.loss.latent_entropy)?;
            if metric.stopped_by_kl {
                ensure(
                    index + 1 == metrics.len() && metric.gradient_norm_before_clip.is_none(),
                    "invalid early-stop update telemetry",
                )?;
                summary.early_stop = Some((
                    index / minibatches_per_epoch + 1,
                    index % minibatches_per_epoch + 1,
                ));
            } else {
                let norm = metric.gradient_norm_before_clip.ok_or_else(|| {
                    racing_rl::Error::Invalid("applied update has no gradient norm".into())
                })?;
                ensure(norm >= 0., "negative gradient norm telemetry")?;
                summary.gradient_norm.push(norm)?;
                summary.gradients_clipped += usize::from(norm > config.max_gradient_norm);
                expected_step = expected_step
                    .checked_add(1)
                    .ok_or_else(|| racing_rl::Error::Invalid("optimizer step overflow".into()))?;
            }
            ensure(
                metric.optimizer_step == expected_step,
                "update telemetry disagrees with recorded optimizer step",
            )?;
        }
        ensure(
            expected_step == optimizer_steps_after,
            "update telemetry disagrees with optimizer step delta",
        )?;
        Ok(summary)
    }

    /// Read-only, human-readable views of the same summaries written to CSV.
    /// Missing observations stay distinct from an observed zero.
    pub(super) fn console_lines(&self) -> Vec<(&'static str, String)> {
        let number = |value: Option<f64>, precision: usize| {
            value.map_or_else(|| "N/A".into(), |value| format!("{value:.precision$}"))
        };
        let percentage = |value: Option<f64>| {
            value.map_or_else(|| "N/A".into(), |value| format!("{:.1}%", value * 100.))
        };
        vec![
            (
                "Adam steps",
                format!("{} / {}", self.updates_applied, self.updates_possible),
            ),
            (
                "Update budget used",
                percentage(Some(
                    self.updates_applied as f64 / self.updates_possible as f64,
                )),
            ),
            (
                "Minibatches per epoch",
                self.minibatches_per_epoch.to_string(),
            ),
            (
                "Optimizer step before",
                self.optimizer_step_before.to_string(),
            ),
            ("Optimizer step after", self.optimizer_step.to_string()),
            (
                "Early stop (KL)",
                if self.early_stop.is_some() {
                    "Yes"
                } else {
                    "No"
                }
                .into(),
            ),
            (
                "Stop epoch",
                self.early_stop
                    .map_or_else(|| "N/A".into(), |(epoch, _)| epoch.to_string()),
            ),
            (
                "Stop minibatch",
                self.early_stop
                    .map_or_else(|| "N/A".into(), |(_, minibatch)| minibatch.to_string()),
            ),
            ("Loss evaluations", self.kl.count.to_string()),
            ("Approximate KL initial", number(self.kl.first, 6)),
            (
                "KL after first update",
                number(self.kl_after_first_update, 6),
            ),
            ("Approximate KL mean", number(self.kl.mean(), 6)),
            ("Approximate KL maximum", number(self.kl.maximum, 6)),
            ("Approximate KL last", number(self.kl.last, 6)),
            (
                "Clip fraction initial",
                percentage(self.clip_fraction.first),
            ),
            ("Clip fraction mean", percentage(self.clip_fraction.mean())),
            (
                "Clip fraction maximum",
                percentage(self.clip_fraction.maximum),
            ),
            ("Clip fraction last", percentage(self.clip_fraction.last)),
            ("Gradient count", self.gradient_norm.count.to_string()),
            ("Gradient norm mean", number(self.gradient_norm.mean(), 4)),
            (
                "Gradient norm maximum",
                number(self.gradient_norm.maximum, 4),
            ),
            ("Gradient norm last", number(self.gradient_norm.last, 4)),
            (
                "Gradients clipped",
                percentage(
                    (self.gradient_norm.count > 0)
                        .then(|| self.gradients_clipped as f64 / self.gradient_norm.count as f64),
                ),
            ),
            ("Log std steering", format!("{:.4}", self.log_std[0])),
            ("Log std acceleration", format!("{:.4}", self.log_std[1])),
            ("Latent std steering", format!("{:.4}", self.latent_std[0])),
            (
                "Latent std acceleration",
                format!("{:.4}", self.latent_std[1]),
            ),
            ("Policy loss mean", number(self.policy_loss.mean(), 4)),
            ("Value loss mean", number(self.value_loss.mean(), 4)),
            ("Total loss mean", number(self.total_loss.mean(), 4)),
            ("Latent entropy mean", number(self.latent_entropy.mean(), 4)),
        ]
    }

    pub(super) fn write_header(writer: &mut impl Write) -> io::Result<()> {
        writeln!(
            writer,
            "rollout,updates_applied,updates_possible,update_budget_fraction,minibatches_per_epoch,optimizer_step_before,optimizer_step,early_stopped,stop_epoch,stop_minibatch,loss_evaluations,kl_initial,kl_after_first_update,kl_mean,kl_max,kl_last,clip_fraction_initial,clip_fraction_mean,clip_fraction_max,clip_fraction_last,gradient_count,gradient_norm_mean,gradient_norm_max,gradient_norm_last,gradient_clipped_fraction,log_std_steering,log_std_acceleration,latent_std_steering,latent_std_acceleration,policy_loss_mean,value_loss_mean,total_loss_mean,latent_entropy_mean"
        )
    }

    pub(super) fn write_row(&self, rollout: u64, writer: &mut impl Write) -> io::Result<()> {
        let (stop_epoch, stop_minibatch) = self.early_stop.map_or_else(
            || (String::new(), String::new()),
            |(epoch, minibatch)| (epoch.to_string(), minibatch.to_string()),
        );
        write!(
            writer,
            "{rollout},{},{},{},{},{},{},{},{stop_epoch},{stop_minibatch},{}",
            self.updates_applied,
            self.updates_possible,
            self.updates_applied as f64 / self.updates_possible as f64,
            self.minibatches_per_epoch,
            self.optimizer_step_before,
            self.optimizer_step,
            self.early_stop.is_some(),
            self.kl.count,
        )?;
        for statistic in [
            self.kl.first,
            self.kl_after_first_update,
            self.kl.mean(),
            self.kl.maximum,
            self.kl.last,
            self.clip_fraction.first,
            self.clip_fraction.mean(),
            self.clip_fraction.maximum,
            self.clip_fraction.last,
        ] {
            write_optional(writer, statistic)?;
        }
        write!(writer, ",{}", self.gradient_norm.count)?;
        for statistic in [
            self.gradient_norm.mean(),
            self.gradient_norm.maximum,
            self.gradient_norm.last,
            (self.gradient_norm.count > 0)
                .then(|| self.gradients_clipped as f64 / self.gradient_norm.count as f64),
        ] {
            write_optional(writer, statistic)?;
        }
        for value in self.log_std.into_iter().chain(self.latent_std) {
            write!(writer, ",{value}")?;
        }
        for statistic in [
            self.policy_loss.mean(),
            self.value_loss.mean(),
            self.total_loss.mean(),
            self.latent_entropy.mean(),
        ] {
            write_optional(writer, statistic)?;
        }
        writeln!(writer)
    }
}

fn write_optional(writer: &mut impl Write, value: Option<f64>) -> io::Result<()> {
    match value {
        Some(value) => write!(writer, ",{value}"),
        None => write!(writer, ","),
    }
}

pub(super) fn write_track_header(writer: &mut impl Write) -> io::Result<()> {
    writeln!(
        writer,
        "rollout,track,decisions,physical_ticks,reward_sum,reward_per_decision,progress_reward_sum,progress_gain,progress_gain_per_decision,progress_fraction_mean_decision,progress_fraction_max,completed,collision,timeout,ended_episode_progress_mean,translation_collisions,rotation_collisions,rotation_only_collisions"
    )
}

/// Episode ending counts and final progress include only episodes ending in this
/// rollout; decision averages also include episodes that continue into the next.
pub(super) fn write_track_row(
    rollout: u64,
    track: &str,
    metrics: &TrackMetrics,
    writer: &mut impl Write,
) -> io::Result<()> {
    if ![
        metrics.reward_sum,
        metrics.progress_reward_sum,
        metrics.progress_gain,
        metrics.progress_fraction_sum,
        f64::from(metrics.progress_fraction_max),
        metrics.ended_episode_progress_sum,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-finite track telemetry",
        ));
    }
    let per_decision = |sum: f64| (metrics.decisions > 0).then(|| sum / metrics.decisions as f64);
    let ended = metrics.episodes_completed + metrics.episodes_collision + metrics.episodes_timeout;
    write!(
        writer,
        "{rollout},\"{}\",{},{},{}",
        track.replace('"', "\"\""),
        metrics.decisions,
        metrics.physical_ticks,
        metrics.reward_sum,
    )?;
    write_optional(writer, per_decision(metrics.reward_sum))?;
    write!(
        writer,
        ",{},{}",
        metrics.progress_reward_sum, metrics.progress_gain
    )?;
    write_optional(writer, per_decision(metrics.progress_gain))?;
    write_optional(writer, per_decision(metrics.progress_fraction_sum))?;
    write!(
        writer,
        ",{},{},{},{}",
        metrics.progress_fraction_max,
        metrics.episodes_completed,
        metrics.episodes_collision,
        metrics.episodes_timeout,
    )?;
    write_optional(
        writer,
        (ended > 0).then(|| metrics.ended_episode_progress_sum / ended as f64),
    )?;
    writeln!(
        writer,
        ",{},{},{}",
        metrics.collision_events.translation,
        metrics.collision_events.rotation,
        metrics.collision_events.rotation_only,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use racing_rl::ppo::LossMetrics;

    fn config() -> Config {
        Config {
            num_envs: 32,
            rollout_length: 256,
            minibatch_size: 512,
            update_epochs: 4,
            ..Default::default()
        }
    }

    fn metrics(applied: usize, stop: bool, initial_step: u64) -> Vec<UpdateMetrics> {
        (0..applied + usize::from(stop))
            .map(|index| {
                let stopped = index == applied;
                UpdateMetrics {
                    loss: LossMetrics {
                        policy_loss: -0.1,
                        value_loss: 0.2,
                        latent_entropy: 1.,
                        total_loss: 0.,
                        mean_ratio: 1.,
                        clip_fraction: index as f32 / 100.,
                        approximate_kl: index as f32 / 1000.,
                    },
                    gradient_norm_before_clip: (!stopped).then_some(1.),
                    optimizer_step: initial_step + (index + usize::from(!stopped)) as u64,
                    stopped_by_kl: stopped,
                }
            })
            .collect()
    }

    fn row(summary: &UpdateTelemetry) -> std::collections::HashMap<String, String> {
        let mut output = Vec::new();
        UpdateTelemetry::write_header(&mut output).unwrap();
        summary.write_row(7, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        let mut lines = output.lines();
        let header: Vec<_> = lines.next().unwrap().split(',').collect();
        let values: Vec<_> = lines.next().unwrap().split(',').collect();
        assert_eq!(header.len(), values.len());
        header
            .into_iter()
            .zip(values)
            .map(|(name, value)| (name.into(), value.into()))
            .collect()
    }

    #[test]
    fn console_formats_existing_diagnostics_with_short_english_labels() {
        let summary =
            UpdateTelemetry::from_metrics(&metrics(3, true, 100), &config(), 100, 103, [-1., -2.])
                .unwrap();
        let lines = summary.console_lines();
        assert_eq!(lines.len(), 31);
        assert!(lines.iter().all(|(label, value)| {
            label.is_ascii() && label.len() <= 32 && value.is_ascii() && value.len() <= 40
        }));
        let values: std::collections::HashMap<_, _> = lines.into_iter().collect();
        assert_eq!(values["Adam steps"], "3 / 64");
        assert_eq!(values["Update budget used"], "4.7%");
        assert_eq!(values["Minibatches per epoch"], "16");
        assert_eq!(values["Optimizer step before"], "100");
        assert_eq!(values["Optimizer step after"], "103");
        assert_eq!(values["Early stop (KL)"], "Yes");
        assert_eq!(values["Stop epoch"], "1");
        assert_eq!(values["Stop minibatch"], "4");
        assert_eq!(values["Loss evaluations"], "4");
        assert_eq!(values["Approximate KL initial"], "0.000000");
        assert_eq!(values["KL after first update"], "0.001000");
        assert_eq!(values["Approximate KL mean"], "0.001500");
        assert_eq!(values["Approximate KL maximum"], "0.003000");
        assert_eq!(values["Approximate KL last"], "0.003000");
        assert_eq!(values["Clip fraction initial"], "0.0%");
        assert_eq!(values["Clip fraction mean"], "1.5%");
        assert_eq!(values["Clip fraction maximum"], "3.0%");
        assert_eq!(values["Clip fraction last"], "3.0%");
        assert_eq!(values["Gradient count"], "3");
        assert_eq!(values["Gradient norm mean"], "1.0000");
        assert_eq!(values["Gradient norm maximum"], "1.0000");
        assert_eq!(values["Gradient norm last"], "1.0000");
        assert_eq!(values["Gradients clipped"], "100.0%");
        assert_eq!(values["Log std steering"], "-1.0000");
        assert_eq!(values["Log std acceleration"], "-2.0000");
        assert_eq!(values["Latent std steering"], "0.3679");
        assert_eq!(values["Latent std acceleration"], "0.1353");
        assert_eq!(values["Policy loss mean"], "-0.1000");
        assert_eq!(values["Value loss mean"], "0.2000");
        assert_eq!(values["Total loss mean"], "0.0000");
        assert_eq!(values["Latent entropy mean"], "1.0000");
    }

    #[test]
    fn console_distinguishes_missing_statistics_from_observed_zero() {
        let summary = UpdateTelemetry::from_metrics(&[], &config(), 5, 5, [0.; 2]).unwrap();
        let values: std::collections::HashMap<_, _> = summary.console_lines().into_iter().collect();
        assert_eq!(values["Early stop (KL)"], "No");
        assert_eq!(values["Loss evaluations"], "0");
        assert_eq!(values["Gradient count"], "0");
        assert_eq!(values["Update budget used"], "0.0%");
        for label in [
            "Stop epoch",
            "Stop minibatch",
            "Approximate KL initial",
            "KL after first update",
            "Approximate KL mean",
            "Approximate KL maximum",
            "Approximate KL last",
            "Clip fraction initial",
            "Clip fraction mean",
            "Clip fraction maximum",
            "Clip fraction last",
            "Gradient norm mean",
            "Gradient norm maximum",
            "Gradient norm last",
            "Gradients clipped",
            "Policy loss mean",
            "Value loss mean",
            "Total loss mean",
            "Latent entropy mean",
        ] {
            assert_eq!(values[label], "N/A", "{label}");
        }
    }

    #[test]
    fn console_presentation_preserves_csv_and_run_log_diagnostics() {
        let summary =
            UpdateTelemetry::from_metrics(&metrics(1, true, 5), &config(), 5, 6, [0.; 2]).unwrap();
        let csv = |summary: &UpdateTelemetry| {
            let mut output = Vec::new();
            UpdateTelemetry::write_header(&mut output).unwrap();
            summary.write_row(7, &mut output).unwrap();
            output
        };
        let csv_before = csv(&summary);
        let debug_before = format!("{summary:?}");
        assert!(!summary.console_lines().is_empty());
        assert_eq!(csv(&summary), csv_before);
        assert_eq!(format!("{summary:?}"), debug_before);
    }

    #[test]
    fn complete_budget_has_exact_optimizer_delta_and_no_stop_position() {
        let metrics = metrics(64, false, 100);
        let summary =
            UpdateTelemetry::from_metrics(&metrics, &config(), 100, 164, [-0.5; 2]).unwrap();
        let row = row(&summary);
        assert_eq!(row["updates_applied"], "64");
        assert_eq!(row["updates_possible"], "64");
        assert_eq!(row["update_budget_fraction"], "1");
        assert_eq!(row["early_stopped"], "false");
        assert_eq!(row["stop_epoch"], "");
        assert_eq!(row["stop_minibatch"], "");
        assert_eq!(row["gradient_count"], "64");
        assert_eq!(row["gradient_clipped_fraction"], "1");
        assert_eq!(row["kl_initial"], "0");
        assert_eq!(row["clip_fraction_initial"], "0");
        assert!((row["kl_after_first_update"].parse::<f64>().unwrap() - 0.001).abs() < 1e-9);
        assert_eq!(metrics[0].optimizer_step, 101);
        assert_eq!(metrics.len(), 64);
    }

    #[test]
    fn rejected_minibatch_is_counted_as_evaluation_not_update() {
        for (applied, expected_epoch, expected_minibatch) in [(1, 1, 2), (16, 2, 1), (63, 4, 16)] {
            let metrics = metrics(applied, true, 100);
            let summary = UpdateTelemetry::from_metrics(
                &metrics,
                &config(),
                100,
                100 + applied as u64,
                [-1., -2.],
            )
            .unwrap();
            let row = row(&summary);
            assert_eq!(row["updates_applied"], applied.to_string());
            assert_eq!(row["loss_evaluations"], (applied + 1).to_string());
            assert_eq!(row["gradient_count"], applied.to_string());
            assert_eq!(row["early_stopped"], "true");
            assert_eq!(row["stop_epoch"], expected_epoch.to_string());
            assert_eq!(row["stop_minibatch"], expected_minibatch.to_string());
            assert_eq!(row["log_std_steering"], "-1");
            assert!(
                (row["latent_std_acceleration"].parse::<f32>().unwrap() - (-2_f32).exp()).abs()
                    < 1e-7
            );
            assert_eq!(metrics.len(), applied + 1);
        }
    }

    #[test]
    fn first_minibatch_stop_has_blank_gradient_statistics() {
        let summary =
            UpdateTelemetry::from_metrics(&metrics(0, true, 5), &config(), 5, 5, [0.; 2]).unwrap();
        let row = row(&summary);
        assert_eq!(row["updates_applied"], "0");
        assert_eq!(row["stop_epoch"], "1");
        assert_eq!(row["stop_minibatch"], "1");
        assert_eq!(row["gradient_count"], "0");
        assert_eq!(row["kl_after_first_update"], "");
        for name in [
            "gradient_norm_mean",
            "gradient_norm_max",
            "gradient_norm_last",
            "gradient_clipped_fraction",
        ] {
            assert_eq!(row[name], "");
        }
        assert!(row.values().all(|value| value != "NaN" && value != "inf"));
    }

    #[test]
    fn optimizer_inconsistencies_and_nonfinite_values_are_rejected() {
        let mut recorded = metrics(1, true, 5);
        assert!(UpdateTelemetry::from_metrics(&recorded, &config(), 5, 7, [0.; 2]).is_err());
        recorded[0].optimizer_step = 99;
        assert!(UpdateTelemetry::from_metrics(&recorded, &config(), 5, 6, [0.; 2]).is_err());
        assert!(UpdateTelemetry::from_metrics(&[], &config(), 5, 5, [f32::NAN; 2]).is_err());
    }

    #[test]
    fn budget_tracks_architecture_independent_minibatch_count() {
        let mut config = config();
        config.num_envs = 1;
        config.rollout_length = 5;
        config.minibatch_size = 2;
        config.update_epochs = 2;
        let summary =
            UpdateTelemetry::from_metrics(&metrics(6, false, 0), &config, 0, 6, [0.; 2]).unwrap();
        assert_eq!(summary.updates_possible, 6);
        assert_eq!(summary.minibatches_per_epoch, 3);
    }

    #[test]
    fn track_rows_report_decision_and_episode_denominators() {
        let metrics = TrackMetrics {
            decisions: 4,
            physical_ticks: 8,
            reward_sum: 2.,
            progress_reward_sum: 1.5,
            progress_gain: 6.,
            progress_fraction_sum: 2.,
            progress_fraction_max: 0.75,
            episodes_completed: 1,
            episodes_collision: 1,
            episodes_timeout: 0,
            ended_episode_progress_sum: 1.25,
            collision_events: crate::simulation::rl_environment::CollisionEvents {
                physical_ticks: 8,
                translation: 1,
                rotation: 3,
                rotation_only: 2,
            },
        };
        let mut output = Vec::new();
        write_track_header(&mut output).unwrap();
        write_track_row(3, "interlagos", &metrics, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        let mut lines = output.lines();
        let header: Vec<_> = lines.next().unwrap().split(',').collect();
        let values: Vec<_> = lines.next().unwrap().split(',').collect();
        assert_eq!(header.len(), values.len());
        assert_eq!(
            values.join(","),
            "3,\"interlagos\",4,8,2,0.5,1.5,6,1.5,0.5,0.75,1,1,0,0.625,1,3,2"
        );
    }

    #[test]
    fn track_csv_escapes_names_and_leaves_missing_averages_blank() {
        let mut output = Vec::new();
        write_track_row(1, "Track, \"A\"", &TrackMetrics::default(), &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(
            output,
            "1,\"Track, \"\"A\"\"\",0,0,0,,0,0,,,0,0,0,0,,0,0,0\n"
        );
        assert!(!output.contains("NaN"));
        assert!(!output.contains("inf"));
    }
}
