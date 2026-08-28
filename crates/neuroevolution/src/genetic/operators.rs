use rand::RngExt;

use crate::genetic::{Config, genome::Genome};
use rand_distr::{Distribution, Normal};

pub fn uniform_crossover<R: rand::Rng>(
    rng: &mut R,
    genome_a: &Genome,
    genome_b: &Genome,
) -> Result<Genome, &'static str> {
    if genome_a.len() != genome_b.len() {
        return Err("Os genomas devem ter o mesmo tamanho");
    }

    let mut child_genome = Vec::with_capacity(genome_a.len());
    for i in 0..genome_a.len() {
        let factor = rng.random_range(0..=1);
        if factor == 0 {
            child_genome.push(genome_a.genes()[i]);
        } else if factor == 1 {
            child_genome.push(genome_b.genes()[i]);
        }
    }
    Ok(Genome::new(child_genome))
}

pub fn mutate<R: rand::Rng>(
    rng: &mut R,
    genome: &mut Genome,
    config: &Config,
) -> Result<(), &'static str> {
    config.validate()?;
    let normal =
        Normal::<f32>::new(0.0, config.mutation_sigma).map_err(|_| "mutation_sigma inválido")?;

    for gene in genome.genes_mut() {
        let happens = rng.random::<f32>() < config.mutation_probability;
        if happens {
            *gene += normal.sample(rng);
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use rand::{SeedableRng, rngs::StdRng};

    use crate::genetic::{
        Config,
        genome::Genome,
        operators::{mutate, uniform_crossover},
    };

    #[test]
    fn child_has_same_length_as_parents() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let genome_a = Genome::new(vec![1.0, 1.0, 1.0, 1.0]);
        let genome_b = Genome::new(vec![2.0, 2.0, 2.0, 2.0]);

        let child = uniform_crossover(&mut rng, &genome_a, &genome_b).unwrap();
        assert_eq!(child.len(), genome_a.len());
    }

    #[test]
    fn child_genes_come_from_either_parent() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let genome_a = Genome::new(vec![1.0, 1.0, 1.0, 1.0]);
        let genome_b = Genome::new(vec![2.0, 2.0, 2.0, 2.0]);

        let child = uniform_crossover(&mut rng, &genome_a, &genome_b).unwrap();
        assert!(child.genes().iter().all(|&gene| gene == 1.0 || gene == 2.0));
    }

    #[test]
    fn error_when_genomes_has_different_len() {
        let config = Config::default();
        let mut rng = StdRng::seed_from_u64(config.seed);
        let genome_a = Genome::new(vec![1.0, 1.0, 1.0, 1.0]);
        let genome_b = Genome::new(vec![2.0, 2.0, 2.0]);

        let child = uniform_crossover(&mut rng, &genome_a, &genome_b);
        assert!(child.is_err());
    }

    #[test]
    fn mutation_with_zero_probability_keeps_genome_unchanged() {
        let mut config = Config::default();
        config.mutation_probability = 0.0;
        let original = Genome::new(vec![-1.0, 0.0, 1.0]);
        let mut genome = original.clone();
        let mut rng = StdRng::seed_from_u64(config.seed);

        mutate(&mut rng, &mut genome, &config).unwrap();

        assert_eq!(genome, original);
    }

    #[test]
    fn mutation_with_invalid_config_returns_error() {
        let mut config = Config::default();
        config.mutation_probability = 1.1;
        let mut genome = Genome::new(vec![1.0, 2.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(config.seed);

        assert!(mutate(&mut rng, &mut genome, &config).is_err());
    }

    #[test]
    fn mutation_with_same_seed_produces_same_genomes() {
        let mut config = Config::default();
        config.mutation_probability = 1.0;
        let mut genome_a = Genome::new(vec![-1.0, 0.0, 1.0]);
        let mut genome_b = genome_a.clone();
        let mut rng_a = StdRng::seed_from_u64(config.seed);
        let mut rng_b = StdRng::seed_from_u64(config.seed);

        mutate(&mut rng_a, &mut genome_a, &config).unwrap();
        mutate(&mut rng_b, &mut genome_b, &config).unwrap();

        assert_eq!(genome_a, genome_b);
    }

    #[test]
    fn mutation_does_not_change_genome_length() {
        let mut config = Config::default();
        config.mutation_probability = 1.0;
        let mut genome = Genome::new(vec![-1.0, 0.0, 1.0]);
        let original_length = genome.len();
        let mut rng = StdRng::seed_from_u64(config.seed);

        mutate(&mut rng, &mut genome, &config).unwrap();

        assert_eq!(genome.len(), original_length);
    }

    #[test]
    fn mutation_with_probability_one_changes_genome() {
        let mut config = Config::default();
        config.mutation_probability = 1.0;
        let original = Genome::new(vec![-1.0, 0.0, 1.0]);
        let mut genome = original.clone();
        let mut rng = StdRng::seed_from_u64(config.seed);

        mutate(&mut rng, &mut genome, &config).unwrap();

        assert_ne!(genome, original);
    }

    #[test]
    fn mutation_with_zero_sigma_keeps_genome_unchanged() {
        let mut config = Config::default();
        config.mutation_probability = 1.0;
        config.mutation_sigma = 0.0;
        let original = Genome::new(vec![-1.0, 0.0, 1.0]);
        let mut genome = original.clone();
        let mut rng = StdRng::seed_from_u64(config.seed);

        mutate(&mut rng, &mut genome, &config).unwrap();

        assert_eq!(genome, original);
    }
}
