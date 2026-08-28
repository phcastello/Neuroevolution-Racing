use super::controller::{MLP_INPUT_SIZE, MLP_OUTPUT_SIZE};
use bevy::prelude::Resource;
use neuroevolution::{
    genetic::{Config, Population},
    neural::{Activation, Architecture},
};
use rand::{SeedableRng, rngs::StdRng};

const DEFAULT_EVALUATION_DURATION: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
}

#[derive(Resource)]
pub struct TrainingState {
    architecture: Architecture,
    config: Config,
    population: Population,
    rng: StdRng,
    evaluation_elapsed: f32,
    evaluation_duration: f32,
    history: Vec<GenerationStats>,
    champion_genome: Option<Vec<f32>>,
}

impl TrainingState {
    pub fn new(population_size: usize) -> Result<Self, String> {
        let layer_sizes = vec![MLP_INPUT_SIZE, 8, MLP_OUTPUT_SIZE];
        let activations = vec![Activation::Tanh, Activation::Tanh];
        let architecture = Architecture::new(layer_sizes, activations)?;

        let config = Config {
            population_size,
            genome_length: architecture.parameter_count(),
            ..Config::default()
        };

        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng)?;

        Ok(Self {
            architecture,
            config,
            population,
            rng,
            evaluation_elapsed: 0.0,
            evaluation_duration: DEFAULT_EVALUATION_DURATION,
            history: Vec::new(),
            champion_genome: None,
        })
    }

    pub fn evolve_generation(&mut self) -> Result<GenerationStats, &'static str> {
        let best = self
            .population
            .best_individual()?
            .ok_or("a populacao nao pode estar vazia")?;
        let best_fitness = best.fitness().unwrap();
        let champion_genome = best.genome().genes().to_vec();
        let average_fitness = self
            .population
            .individuals()
            .iter()
            .map(|individual| individual.fitness().unwrap())
            .sum::<f32>()
            / self.population.len() as f32;
        let stats = GenerationStats {
            generation: self.population.generation(),
            best_fitness,
            average_fitness,
        };
        let next_population = self.population.evolve(&self.config, &mut self.rng)?;

        self.population = next_population;
        self.champion_genome = Some(champion_genome);
        self.history.push(stats);
        self.reset_evaluation_time();

        Ok(stats)
    }

    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn population_mut(&mut self) -> &mut Population {
        &mut self.population
    }

    pub fn generation(&self) -> usize {
        self.population.generation()
    }

    pub fn history(&self) -> &[GenerationStats] {
        &self.history
    }

    pub fn champion_genome(&self) -> Option<&[f32]> {
        self.champion_genome.as_deref()
    }

    pub fn evaluation_elapsed(&self) -> f32 {
        self.evaluation_elapsed
    }

    pub fn evaluation_duration(&self) -> f32 {
        self.evaluation_duration
    }

    pub fn evaluation_remaining(&self) -> f32 {
        (self.evaluation_duration - self.evaluation_elapsed).max(0.0)
    }

    pub fn evaluation_progress(&self) -> f32 {
        (self.evaluation_elapsed / self.evaluation_duration).clamp(0.0, 1.0)
    }

    pub fn advance_evaluation(&mut self, delta_seconds: f32) -> Result<bool, &'static str> {
        if !delta_seconds.is_finite() || delta_seconds < 0.0 {
            return Err("delta_seconds deve ser finito e nao negativo");
        }

        self.evaluation_elapsed =
            (self.evaluation_elapsed + delta_seconds).min(self.evaluation_duration);

        Ok(self.evaluation_finished())
    }

    pub fn reset_evaluation_time(&mut self) {
        self.evaluation_elapsed = 0.0;
    }

    pub fn set_evaluation_duration(&mut self, duration_seconds: f32) -> Result<(), &'static str> {
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err("a duracao da avaliacao deve ser finita e maior que zero");
        }

        self.evaluation_duration = duration_seconds;
        self.reset_evaluation_time();
        Ok(())
    }

    pub fn evaluation_finished(&self) -> bool {
        self.evaluation_elapsed >= self.evaluation_duration
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_EVALUATION_DURATION, TrainingState};

    fn training_state() -> TrainingState {
        TrainingState::new(2).expect("training state should be valid")
    }

    #[test]
    fn evaluation_timer_starts_at_zero() {
        let state = training_state();

        assert_eq!(state.evaluation_elapsed(), 0.0);
        assert_eq!(state.evaluation_duration(), DEFAULT_EVALUATION_DURATION);
        assert_eq!(state.evaluation_remaining(), DEFAULT_EVALUATION_DURATION);
        assert_eq!(state.evaluation_progress(), 0.0);
        assert!(!state.evaluation_finished());
    }

    #[test]
    fn advancing_timer_stops_at_duration_and_reports_completion() {
        let mut state = training_state();

        assert!(!state.advance_evaluation(7.5).unwrap());
        assert_eq!(state.evaluation_elapsed(), 7.5);
        assert_eq!(state.evaluation_remaining(), 12.5);

        assert!(state.advance_evaluation(20.0).unwrap());
        assert_eq!(state.evaluation_elapsed(), DEFAULT_EVALUATION_DURATION);
        assert_eq!(state.evaluation_remaining(), 0.0);
        assert_eq!(state.evaluation_progress(), 1.0);
    }

    #[test]
    fn resetting_timer_starts_a_new_evaluation() {
        let mut state = training_state();
        state
            .advance_evaluation(DEFAULT_EVALUATION_DURATION)
            .unwrap();

        state.reset_evaluation_time();

        assert_eq!(state.evaluation_elapsed(), 0.0);
        assert!(!state.evaluation_finished());
    }

    #[test]
    fn changing_duration_resets_timer() {
        let mut state = training_state();
        state.advance_evaluation(5.0).unwrap();

        state.set_evaluation_duration(10.0).unwrap();

        assert_eq!(state.evaluation_duration(), 10.0);
        assert_eq!(state.evaluation_elapsed(), 0.0);
    }

    #[test]
    fn timer_rejects_invalid_values() {
        let mut state = training_state();

        assert!(state.advance_evaluation(-1.0).is_err());
        assert!(state.advance_evaluation(f32::NAN).is_err());
        assert!(state.set_evaluation_duration(0.0).is_err());
        assert!(state.set_evaluation_duration(f32::INFINITY).is_err());
        assert_eq!(state.evaluation_elapsed(), 0.0);
        assert_eq!(state.evaluation_duration(), DEFAULT_EVALUATION_DURATION);
    }

    #[test]
    fn constructor_uses_requested_population_size() {
        let state = TrainingState::new(3).unwrap();

        assert_eq!(state.population().len(), 3);
    }

    #[test]
    fn evolution_records_generation_stats_and_champion() {
        let mut state = training_state();
        state.population_mut().individuals_mut()[0].set_fitness(10.0);
        state.population_mut().individuals_mut()[1].set_fitness(30.0);
        let expected_champion = state.population().individuals()[1]
            .genome()
            .genes()
            .to_vec();

        let stats = state.evolve_generation().unwrap();

        assert_eq!(stats.generation, 0);
        assert_eq!(stats.best_fitness, 30.0);
        assert_eq!(stats.average_fitness, 20.0);
        assert_eq!(state.generation(), 1);
        assert_eq!(state.history(), &[stats]);
        assert_eq!(state.champion_genome(), Some(expected_champion.as_slice()));
    }
}
