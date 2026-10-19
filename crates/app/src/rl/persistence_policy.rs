//! Optional artifact sampling. This policy never controls collection or updates.
use std::num::NonZeroU64;

use clap::Args;

#[derive(Args, Clone, Debug, Default)]
pub struct PersistenceOptions {
    /// Save heavy artifacts every N completed rollouts, plus the first/last windows.
    #[arg(long, value_name = "N")]
    pub(super) save_every: Option<NonZeroU64>,
    /// Save this many initial rollouts (default: 250 when --save-every is set).
    #[arg(long, value_name = "N", requires = "save_every")]
    pub(super) save_first: Option<u64>,
    /// Save this many final rollouts (default: 250 when --save-every is set).
    #[arg(long, value_name = "N", requires = "save_every")]
    pub(super) save_last: Option<u64>,
}

impl PersistenceOptions {
    pub(super) fn should_save(&self, completed: u64, total: usize) -> bool {
        let total = total as u64;
        if completed == 0 || completed > total {
            return false;
        }
        let Some(every) = self.save_every else {
            return true;
        };
        completed == total
            || completed <= self.save_first.unwrap_or(250)
            || completed > total.saturating_sub(self.save_last.unwrap_or(250))
            || completed % every.get() == 0
    }

    /// Sampling aligns all artifacts with the completed rollout. Preserve legacy
    /// zero-based loss filenames when the optional sampling policy is absent.
    pub(super) fn losses_index(&self, completed: u64) -> u64 {
        if self.save_every.is_some() {
            completed
        } else {
            completed.saturating_sub(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn sampled() -> PersistenceOptions {
        PersistenceOptions {
            save_every: NonZeroU64::new(250),
            ..Default::default()
        }
    }

    #[test]
    fn twenty_thousand_rollout_boundaries_and_resumed_counts() {
        let policy = sampled();
        for (completed, expected) in [
            (250, true),
            (251, false),
            (500, true),
            (19_749, false),
            (19_750, true),
            (19_751, true),
            (20_000, true),
        ] {
            assert_eq!(
                policy.should_save(completed, 20_000),
                expected,
                "{completed}"
            );
        }
        assert_eq!(
            (1..=20_000)
                .filter(|n| policy.should_save(*n, 20_000))
                .count(),
            578
        );
        assert_eq!(
            (251..=20_000)
                .filter(|n| policy.should_save(*n, 20_000))
                .count(),
            328
        );
    }

    #[test]
    fn short_targets_overlap_without_underflow_or_missing_final_rollouts() {
        let policy = sampled();
        for total in [1, 10, 249, 250, 300, 499, 500] {
            assert!((1..=total as u64).all(|n| policy.should_save(n, total)));
        }
        assert!(!policy.should_save(0, 0));
        assert!(!policy.should_save(0, 20_000));
        assert!(!policy.should_save(20_001, 20_000));
    }

    #[test]
    fn final_checkpoint_is_saved_even_without_windows_or_matching_interval() {
        let policy = PersistenceOptions {
            save_first: Some(0),
            save_last: Some(0),
            ..sampled()
        };
        assert!(!policy.should_save(1, 1_001));
        assert!(policy.should_save(1_000, 1_001));
        assert!(policy.should_save(1_001, 1_001));
        assert!(policy.should_save(1, 1));
    }

    #[test]
    fn windows_and_interval_are_configurable() {
        let policy = PersistenceOptions {
            save_every: NonZeroU64::new(3),
            save_first: Some(2),
            save_last: Some(2),
        };
        let selected: Vec<_> = (1..=11).filter(|n| policy.should_save(*n, 11)).collect();
        assert_eq!(selected, [1, 2, 3, 6, 9, 10, 11]);
    }

    #[test]
    fn absent_sampling_preserves_all_artifacts_and_legacy_loss_indices() {
        let policy = PersistenceOptions::default();
        assert!((1..=20_000).all(|n| policy.should_save(n, 20_000)));
        for completed in [1, 250, 251, 500, 20_000] {
            assert_eq!(policy.losses_index(completed), completed - 1);
            assert_eq!(sampled().losses_index(completed), completed);
        }
    }

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        persistence: PersistenceOptions,
    }

    #[test]
    fn cli_rejects_zero_interval_and_windows_without_sampling() {
        for arguments in [
            vec!["test", "--save-every", "0"],
            vec!["test", "--save-first", "250"],
            vec!["test", "--save-last", "250"],
        ] {
            assert!(TestCli::try_parse_from(arguments).is_err());
        }
        assert!(TestCli::try_parse_from(["test"]).is_ok());
        let cli = TestCli::try_parse_from([
            "test",
            "--save-every",
            "250",
            "--save-first",
            "0",
            "--save-last",
            "0",
        ])
        .unwrap();
        assert_eq!(cli.persistence.save_every.unwrap().get(), 250);
        assert_eq!(cli.persistence.save_first, Some(0));
        assert_eq!(cli.persistence.save_last, Some(0));
    }

    #[test]
    fn complete_resume_command_parses_without_starting_training() {
        let cli = crate::Cli::try_parse_from([
            "neuroevolution-racing",
            "rl",
            "train",
            "--config",
            "configs/rl/large_200k_20k.ron",
            "--resume",
            "results_ppo/ppo-1792357418403561300-31472/checkpoint-000250.ron",
            "--save-every",
            "250",
            "--save-first",
            "250",
            "--save-last",
            "250",
        ])
        .unwrap();
        let Some(crate::CommandLine::Rl {
            command:
                crate::rl::RlCommand::Train {
                    config,
                    resume,
                    persistence,
                },
        }) = cli.command
        else {
            panic!("Expected the PPO resume command")
        };
        assert_eq!(
            config,
            std::path::Path::new("configs/rl/large_200k_20k.ron")
        );
        assert_eq!(
            resume.unwrap(),
            std::path::Path::new("results_ppo/ppo-1792357418403561300-31472/checkpoint-000250.ron")
        );
        assert!(!persistence.should_save(251, 20_000));
        assert!(persistence.should_save(500, 20_000));
        assert!(persistence.should_save(20_000, 20_000));
    }
}
