use rand::RngExt;

use crate::genetic::{
    Config,
    genome::Genome,
    operators::{mutate, uniform_crossover},
};

use super::individual::Individual;

pub struct Population {
    individuals: Vec<Individual>,
    generation: usize,
}

impl Population {
    pub fn new<R: rand::Rng>(config: &Config, rng: &mut R) -> Result<Self, String> {
        config.validate()?;
        let mut individuals = Vec::with_capacity(config.population_size);

        for _ in 0..config.population_size {
            let genome = Genome::random(config, rng);
            let individual = Individual::new(genome);
            individuals.push(individual);
        }

        Ok(Self {
            individuals,
            generation: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn individuals_mut(&mut self) -> &mut [Individual] {
        self.individuals.as_mut_slice()
    }

    pub fn individuals(&self) -> &[Individual] {
        self.individuals.as_slice()
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.individuals().iter().all(|individual| {
            individual
                .fitness()
                .is_some_and(|fitness| fitness.is_finite())
        }) {
            Ok(())
        } else {
            Err("Todos os individuos devem possuir fitness finito")
        }
    }

    pub fn best_individual(&self) -> Result<Option<&Individual>, &'static str> {
        self.validate()?;

        let mut best_individual: Option<&Individual> = None;
        for individual in self.individuals() {
            match best_individual {
                None => {
                    best_individual = Some(individual);
                }
                Some(current_best) => {
                    if individual.fitness().unwrap() > current_best.fitness().unwrap() {
                        best_individual = Some(individual);
                    }
                }
            }
        }

        Ok(best_individual)
    }

    pub fn tournament_selection<R: rand::Rng>(
        &self,
        config: &Config,
        rng: &mut R,
    ) -> Result<usize, &'static str> {
        config.validate()?;
        self.validate()?;

        let mut best_index: Option<usize> = None;
        for _ in 0..config.tournament_size {
            let index = rng.random_range(0..self.len());
            match best_index {
                None => {
                    best_index = Some(index);
                }
                Some(current_index) => {
                    if self.individuals[index].fitness().unwrap()
                        > self.individuals[current_index].fitness().unwrap()
                    {
                        best_index = Some(index);
                    }
                }
            }
        }

        Ok(best_index.unwrap())
    }

    pub fn sort_by_fitness_descending(&mut self) -> Result<(), &'static str> {
        self.validate()?;
        self.individuals
            .sort_by(|a, b| b.fitness().unwrap().total_cmp(&a.fitness().unwrap()));

        Ok(())
    }

    pub fn calculate_elite_amount(&self, config: &Config) -> usize {
        if config.elite_fraction == 0.0 {
            return 0;
        } else {
            let elite_amount = self.individuals().len() as f32 * config.elite_fraction;
            elite_amount.round().max(1.0) as usize
        }
    }

    pub fn elites(&mut self, config: &Config) -> Result<Vec<Individual>, &'static str> {
        config.validate()?;
        self.validate()?;
        let _ = self.sort_by_fitness_descending();
        let elite_count = self.calculate_elite_amount(config);
        let mut elites: Vec<Individual> = Vec::with_capacity(elite_count);

        for elite_individual_index in 0..elite_count {
            elites.push(self.individuals[elite_individual_index].clone());
        }
        Ok(elites)
    }

    pub fn create_child<R: rand::Rng>(
        &self,
        config: &Config,
        rng: &mut R,
    ) -> Result<Individual, &'static str> {
        // seleciona pais por tournment
        let parent_a_index = self.tournament_selection(config, rng)?;
        let parent_b_index = self.tournament_selection(config, rng)?;

        let genome_parent_a = self.individuals()[parent_a_index].genome();
        let genome_parent_b = self.individuals()[parent_b_index].genome();

        // sorteia se haverá crossover
        let mut child_genome = if rng.random::<f32>() < config.crossover_probability {
            uniform_crossover(rng, genome_parent_a, genome_parent_b)?
        } else if rng.random_bool(0.5) {
            genome_parent_a.clone()
        } else {
            genome_parent_b.clone()
        };

        mutate(rng, &mut child_genome, config)?;

        Ok(Individual::new(child_genome))
    }

    pub fn evolve<R: rand::Rng>(
        &mut self,
        config: &Config,
        rng: &mut R,
    ) -> Result<Population, &'static str> {
        config.validate()?;
        self.validate()?;

        let mut next_individuals = self.elites(config)?;

        while next_individuals.len() < config.population_size {
            next_individuals.push(self.create_child(config, rng)?);
        }

        let new_population = Population {
            individuals: next_individuals,
            generation: self.generation + 1,
        };

        Ok(new_population)
    }
}

#[cfg(test)]
mod test {
    use rand::{RngExt, rngs::StdRng};

    use super::{Config, Genome, Individual};
    use crate::genetic::population::Population;
    use rand::SeedableRng;

    #[test]
    fn population_has_config_population_size() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert_eq!(population.len(), config.population_size);
    }

    #[test]
    fn individuals_start_without_fitness() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert!(
            population
                .individuals
                .iter()
                .all(|individual| individual.fitness().is_none())
        )
    }

    #[test]
    fn population_starts_in_gen_zero() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert_eq!(population.generation, 0)
    }

    #[test]
    fn individuals_mut_exposes_mutable_individuals() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut population = Population::new(&config, &mut rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        assert!(
            population
                .individuals
                .iter()
                .all(|individual| individual.fitness() == Some(1.0))
        );
    }

    #[test]
    fn return_best_individual_from_population() {
        let mut population = Population {
            individuals: vec![
                Individual::new(Genome::new(vec![1.0])),
                Individual::new(Genome::new(vec![2.0])),
                Individual::new(Genome::new(vec![3.0])),
            ],
            generation: 0,
        };

        population.individuals_mut()[0].set_fitness(-3.0);
        population.individuals_mut()[1].set_fitness(-1.0);
        population.individuals_mut()[2].set_fitness(-2.0);

        let best = population.best_individual().unwrap().unwrap();
        assert_eq!(best.fitness(), Some(-1.0));
    }

    #[test]
    fn sorts_individuals_by_fitness_descending() {
        let mut population = Population {
            individuals: vec![
                Individual::new(Genome::new(vec![-1.0])),
                Individual::new(Genome::new(vec![-2.0])),
                Individual::new(Genome::new(vec![-3.0])),
            ],
            generation: 0,
        };

        population.individuals_mut()[0].set_fitness(-1.0);
        population.individuals_mut()[1].set_fitness(-3.0);
        population.individuals_mut()[2].set_fitness(-2.0);

        population.sort_by_fitness_descending().unwrap();

        let fitnesses: Vec<f32> = population
            .individuals()
            .iter()
            .map(|individual| individual.fitness().unwrap())
            .collect();

        assert_eq!(fitnesses, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn error_if_individual_without_fitness() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert!(population.validate().is_err());
    }

    #[test]
    fn tournament_selection_errors_when_population_has_no_fitness() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert!(population.tournament_selection(&config, &mut rng).is_err());
    }

    #[test]
    fn tournament_selection_errors_when_tournament_size_is_zero() {
        let mut config = Config::default();
        config.tournament_size = 0;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&Config::default(), &mut rng).unwrap();

        assert!(population.tournament_selection(&config, &mut rng).is_err());
    }

    #[test]
    fn tournament_selection_returns_an_index_from_the_population() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut population = Population::new(&config, &mut rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let selected_index = population.tournament_selection(&config, &mut rng).unwrap();

        assert!(selected_index < population.len());
    }

    #[test]
    fn tournament_selection_chooses_best_fitness_among_sampled() {
        let config = Config::default();
        let mut selection_rng = StdRng::seed_from_u64(config.seed);
        let mut expected_rng = StdRng::seed_from_u64(config.seed);

        let mut selection_population = Population::new(&config, &mut selection_rng).unwrap();
        let mut expected_population = Population::new(&config, &mut expected_rng).unwrap();

        for individual in selection_population.individuals_mut() {
            individual.set_fitness(
                selection_rng.random_range(config.initial_gene_min..config.initial_gene_max),
            );
        }

        for individual in expected_population.individuals_mut() {
            individual.set_fitness(
                expected_rng.random_range(config.initial_gene_min..config.initial_gene_max),
            );
        }

        let best_individual_selection_population_index = selection_population
            .tournament_selection(&config, &mut selection_rng)
            .unwrap();

        let mut best_individual_expected_population_index: Option<usize> = None;
        let mut selected_indexes: Vec<usize> = Vec::with_capacity(config.tournament_size);
        for _ in 0..config.tournament_size {
            let index = expected_rng.random_range(0..expected_population.len());
            selected_indexes.push(index);
        }

        for i in selected_indexes {
            match best_individual_expected_population_index {
                None => {
                    best_individual_expected_population_index = Some(i);
                }
                Some(current_best_index) => {
                    if expected_population.individuals[i].fitness().unwrap()
                        > expected_population.individuals[current_best_index]
                            .fitness()
                            .unwrap()
                    {
                        best_individual_expected_population_index = Some(i);
                    }
                }
            }
        }

        assert_eq!(
            best_individual_expected_population_index.unwrap(),
            best_individual_selection_population_index
        )
    }

    #[test]
    fn calculate_elite_amount_is_zero_when_fraction_is_zero() {
        let mut config = Config::default();
        config.elite_fraction = 0.0;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert_eq!(population.calculate_elite_amount(&config), 0);
    }

    #[test]
    fn calculate_elite_amount_returns_fraction_of_population() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert_eq!(population.calculate_elite_amount(&config), 25);
    }

    #[test]
    fn calculate_elite_amount_keeps_one_elite_for_small_positive_fraction() {
        let mut config = Config::default();
        config.population_size = 2;
        config.elite_fraction = 0.05;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert_eq!(population.calculate_elite_amount(&config), 1);
    }

    #[test]
    fn calculate_elite_amount_rounds_to_nearest_integer() {
        let mut config = Config::default();
        config.population_size = 37;
        config.elite_fraction = 0.05;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let population = Population::new(&config, &mut rng).unwrap();

        assert_eq!(population.calculate_elite_amount(&config), 2);
    }

    #[test]
    fn select_elites_returns_top_two_independent_clones() {
        let mut config = Config::default();
        config.elite_fraction = 0.5;

        let mut population = Population {
            individuals: vec![
                Individual::new(Genome::new(vec![-1.0])),
                Individual::new(Genome::new(vec![-2.0])),
                Individual::new(Genome::new(vec![-3.0])),
                Individual::new(Genome::new(vec![-0.1])),
            ],
            generation: 0,
        };

        population.individuals_mut()[0].set_fitness(-1.0);
        population.individuals_mut()[1].set_fitness(-3.0);
        population.individuals_mut()[2].set_fitness(-2.0);
        population.individuals_mut()[3].set_fitness(-0.2);

        let mut elites = population.elites(&config).unwrap();

        assert_eq!(elites.len(), 2);
        assert_eq!(elites[0].fitness(), Some(-0.2));
        assert_eq!(elites[1].fitness(), Some(-1.0));

        assert_eq!(elites[0].fitness(), population.individuals()[0].fitness());
        elites[0].set_fitness(-2.0);
        assert_ne!(elites[0].fitness(), population.individuals()[0].fitness());
    }

    #[test]
    fn create_child_starts_without_fitness() {
        let config = Config::default();
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut child_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let child = population.create_child(&config, &mut child_rng).unwrap();

        assert_eq!(child.fitness(), None);
    }

    #[test]
    fn create_child_has_the_same_genome_length_as_its_parents() {
        let config = Config::default();
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut child_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let child = population.create_child(&config, &mut child_rng).unwrap();

        assert_eq!(child.genome().len(), config.genome_length);
    }

    #[test]
    fn create_child_without_crossover_or_mutation_copies_a_selectable_parent_genome() {
        let mut config = Config::default();
        config.crossover_probability = 0.0;
        config.mutation_probability = 0.0;
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut child_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let child = population.create_child(&config, &mut child_rng).unwrap();

        assert!(
            population
                .individuals()
                .iter()
                .any(|parent| child.genome() == parent.genome())
        );
    }

    #[test]
    fn create_child_is_deterministic_for_equal_population_and_rng_state() {
        let config = Config::default();
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let mut rng_a = StdRng::seed_from_u64(config.seed + 1);
        let mut rng_b = StdRng::seed_from_u64(config.seed + 1);
        let child_a = population.create_child(&config, &mut rng_a).unwrap();
        let child_b = population.create_child(&config, &mut rng_b).unwrap();

        assert_eq!(child_a.genome(), child_b.genome());
        assert_eq!(child_a.fitness(), child_b.fitness());
    }

    #[test]
    fn create_child_errors_when_population_has_not_been_evaluated() {
        let config = Config::default();
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut child_rng = StdRng::seed_from_u64(config.seed + 1);
        let population = Population::new(&config, &mut population_rng).unwrap();

        assert!(population.create_child(&config, &mut child_rng).is_err());
    }

    #[test]
    fn evolve_returns_a_population_with_the_configured_size() {
        let mut config = Config::default();
        config.population_size = 8;
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut evolution_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let next_population = population.evolve(&config, &mut evolution_rng).unwrap();

        assert_eq!(next_population.len(), config.population_size);
    }

    #[test]
    fn evolve_increments_the_generation_from_zero_and_a_later_generation() {
        let mut config = Config::default();
        config.population_size = 2;
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut evolution_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for individual in population.individuals_mut() {
            individual.set_fitness(1.0);
        }

        let next_population = population.evolve(&config, &mut evolution_rng).unwrap();
        assert_eq!(next_population.generation, 1);

        let mut later_population = next_population;
        for individual in later_population.individuals_mut() {
            individual.set_fitness(1.0);
        }
        later_population.generation = 7;

        let following_population = later_population
            .evolve(&config, &mut evolution_rng)
            .unwrap();
        assert_eq!(following_population.generation, 8);
    }

    #[test]
    fn evolve_preserves_the_best_elites_unchanged() {
        let mut config = Config::default();
        config.population_size = 8;
        config.genome_length = 1;
        config.elite_fraction = 0.25;
        let mut evolution_rng = StdRng::seed_from_u64(config.seed);
        let mut population = Population {
            individuals: (0..8)
                .map(|gene| Individual::new(Genome::new(vec![gene as f32])))
                .collect(),
            generation: 0,
        };

        let fitnesses = [3.0, 8.0, 1.0, 6.0, 2.0, 7.0, 4.0, 5.0];
        for (individual, fitness) in population.individuals_mut().iter_mut().zip(fitnesses) {
            individual.set_fitness(fitness);
        }

        let next_population = population.evolve(&config, &mut evolution_rng).unwrap();

        assert_eq!(
            next_population.individuals()[0].genome().genes(),
            &vec![1.0]
        );
        assert_eq!(next_population.individuals()[0].fitness(), Some(8.0));
        assert_eq!(
            next_population.individuals()[1].genome().genes(),
            &vec![5.0]
        );
        assert_eq!(next_population.individuals()[1].fitness(), Some(7.0));
    }

    #[test]
    fn evolve_keeps_fitness_only_for_elites() {
        let mut config = Config::default();
        config.population_size = 8;
        config.elite_fraction = 0.25;
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut evolution_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        for (index, individual) in population.individuals_mut().iter_mut().enumerate() {
            individual.set_fitness(index as f32);
        }

        let next_population = population.evolve(&config, &mut evolution_rng).unwrap();
        let elite_amount = (config.population_size as f32 * config.elite_fraction).round() as usize;

        assert!(
            next_population.individuals()[..elite_amount]
                .iter()
                .all(|individual| individual.fitness().is_some())
        );
        assert!(
            next_population.individuals()[elite_amount..]
                .iter()
                .all(|individual| individual.fitness().is_none())
        );
    }

    #[test]
    fn evolve_errors_when_population_has_not_been_evaluated() {
        let config = Config::default();
        let mut population_rng = StdRng::seed_from_u64(config.seed);
        let mut evolution_rng = StdRng::seed_from_u64(config.seed + 1);
        let mut population = Population::new(&config, &mut population_rng).unwrap();

        assert!(population.evolve(&config, &mut evolution_rng).is_err());
    }
}
