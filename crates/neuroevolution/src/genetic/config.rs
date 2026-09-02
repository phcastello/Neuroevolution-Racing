#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub population_size: usize,
    pub genome_length: usize,

    pub elite_fraction: f32,
    pub tournament_size: usize,

    pub crossover_probability: f32,

    pub mutation_probability: f32,
    pub mutation_sigma: f32,

    pub initial_gene_min: f32,
    pub initial_gene_max: f32,

    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 500,
            genome_length: 5,
            elite_fraction: 0.05,
            tournament_size: 3,
            crossover_probability: 0.80,
            mutation_probability: 0.10,
            mutation_sigma: 0.15,
            initial_gene_min: -1.0,
            initial_gene_max: 1.0,
            seed: 42,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.population_size == 0 {
            return Err("population_size deve ser maior que zero".into());
        }

        if self.genome_length == 0 {
            return Err("genome_length deve ser maior que zero".into());
        }

        if !(0.0..=1.0).contains(&self.elite_fraction) {
            return Err("elite_fraction deve estar entre 0.0 e 1.0".into());
        }

        if self.tournament_size == 0 {
            return Err("tournament_size deve ser maior que zero".into());
        }

        if !(0.0..=1.0).contains(&self.crossover_probability) {
            return Err("crossover_probability deve estar entre 0.0 e 1.0".into());
        }

        if !(0.0..=1.0).contains(&self.mutation_probability) {
            return Err("mutation_probability deve estar entre 0.0 e 1.0".into());
        }

        if !self.mutation_sigma.is_finite() || self.mutation_sigma < 0.0 {
            return Err("mutation_sigma deve ser um número finito não negativo".into());
        }

        if !self.initial_gene_min.is_finite() || !self.initial_gene_max.is_finite() {
            return Err("os limites dos genes devem ser números finitos".into());
        }

        if self.initial_gene_min >= self.initial_gene_max {
            return Err("initial_gene_min deve ser menor que initial_gene_max".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::genetic::Config;

    #[test]
    fn population_size_eq_zero_results_error() {
        let mut config = Config::default();
        config.population_size = 0;

        assert!(config.validate().is_err());
    }

    #[test]
    fn probabilities_outside_unit_interval_are_rejected() {
        for invalid in [-0.01, 1.01, f32::NAN] {
            let mut crossover = Config::default();
            crossover.crossover_probability = invalid;
            assert!(crossover.validate().is_err());

            let mut mutation = Config::default();
            mutation.mutation_probability = invalid;
            assert!(mutation.validate().is_err());
        }
    }
}
