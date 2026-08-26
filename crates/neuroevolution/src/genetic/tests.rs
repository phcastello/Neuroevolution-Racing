use rand::{rngs::StdRng, SeedableRng};

use crate::genetic::{Config, genome::Genome, population::Population};
use crate::neural::{Activation, Architecture, Mlp};

fn calculate_fitness(target: &Genome, genome: &Genome) -> f32 {
    let mut fitness: f32 = 0.0;

    for i in 0..genome.len() {
        let gene = genome.genes()[i];
        let target_gene = target.genes()[i];
        let diff = gene - target_gene;

        fitness -= diff.powi(2);
    }

    fitness
}

fn evaluate_unevaluated_population_against_target(population: &mut Population, target: &Genome) {
    for individual in population.individuals_mut() {
        if individual.fitness().is_none() {
            let fitness = calculate_fitness(target, individual.genome());
            individual.set_fitness(fitness);
        }
    }
}

#[test]
fn random_genome_builds_a_valid_mlp_with_the_architecture_parameter_count() {
    let architecture = Architecture::new(vec![6, 8, 2], vec![Activation::Tanh, Activation::Tanh])
        .unwrap();
    assert_eq!(architecture.parameter_count(), 74);

    let mut config = Config::default();
    config.genome_length = architecture.parameter_count();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let genome = Genome::random(&config, &mut rng);

    assert_eq!(genome.len(), 74);

    let mlp = Mlp::from_parameters(&architecture, genome.genes()).unwrap();
    let output = mlp.forward(&[0.0; 6]).unwrap();

    assert_eq!(output.len(), 2);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn target_fitness_is_assigned_to_all_individuals(){
    let config = Config::default();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut population = Population::new(&config, &mut rng).unwrap();

    let target = Genome::new(vec![0.5,-0.3,-0.9,0.1,0.0]);

    for individual in population.individuals_mut(){
        let fitness = calculate_fitness(&target, individual.genome());
        individual.set_fitness(fitness);
    }
    
    assert!(
        population
            .individuals()
            .iter()
            .all(|individual| individual.fitness().is_some())
    );

}

#[test]
fn fitness_calculation_works_properly(){
    let target = Genome::new(vec![1.0, 0.0]);
    let genome = Genome::new(vec![0.5, 1.0]);

    let fitness = calculate_fitness(&target, &genome);

    // -(0.5 - 1.0)^2 - (1.0 - 0.0)^2 = -0.25 - 1.0 = -1.25
    assert_eq!(fitness, -1.25);
}

#[test]
fn best_final_fitness_is_greater_than_initial(){
    let config = Config::default();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut population = Population::new(&config, &mut rng).unwrap();

    let target = Genome::new(vec![0.5,-0.3,-0.9,0.1,0.0]);


    let target_generation = 200;
    let mut best_fitness_by_gen: Vec<f32> = Vec::with_capacity(target_generation+1);
    let mut best_genome_by_gen: Vec<Genome> = Vec::with_capacity(target_generation+1);

    while population.generation() <= target_generation{
        evaluate_unevaluated_population_against_target(&mut population, &target);

        // coleta o fitness do melhor individuo de cada geração.
        // cada indice representa exatamente a geração.
        let best_individual = population
            .best_individual()
            .unwrap()
            .unwrap();
        best_fitness_by_gen.push(best_individual.fitness().unwrap());
        best_genome_by_gen.push(best_individual.genome().clone());

        if population.generation() == target_generation{
            break;
        }
        population = population.evolve(&config, &mut rng).unwrap()
    }

    println!("\n========== GENOMA-ALVO ==========");
    println!("{:?}", target.genes());
    println!("==================================");

    for generation in [0, 1, 100, target_generation] {
        println!("\n======= GERAÇÃO {generation} =======");
        println!("fitness: {}", best_fitness_by_gen[generation]);
        println!("genoma: {:?}", best_genome_by_gen[generation].genes());
        println!("=========================");
    }

    assert!(
        best_fitness_by_gen[0] < best_fitness_by_gen[target_generation],
        "fitness gen zero: {}, fitness gen 1: {}, fitness gen 100: {}, fitness gen {}: {}",
        best_fitness_by_gen[0],
        best_fitness_by_gen[1],
        best_fitness_by_gen[100],
        target_generation,
        best_fitness_by_gen[target_generation]
    );
    assert!(
        best_fitness_by_gen
            .windows(2)
            .all(|pair| pair[1] >= pair[0]),
        "O melhor fitness diminuiu em alguma geração: {:?}",
        best_fitness_by_gen
    );
}
