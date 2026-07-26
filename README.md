# Evolite/Neural

A small neural-network implementation trained using a Genetic Algorithm (GA). Instead of backpropagation, populations of networks are evolved using fitness evaluation, crossover, selection, and mutation.

## Features

- Feed-forward network built from configurable layer sizes
- Multiple activation functions (sigmoid, relu, tanh, identity, step, softmax)
- Training via a genetic algorithm:

  - Fitness minimization (based on a loss function)
  - Tournament selection
  - Crossover (layer/weight-wise mixing)
  - Mutations (weights, biases, activations, node swapping, and random resets)

- Optional penalty for large weights

## How it works

### Forward Pass (`activate`)

Inputs are propagated layer-by-layer:

1. Each `Layer` computes:
   - Weighted sum of inputs
   - Plus bias
2. The layer applies its activation function to produce outputs.
3. The final layer output is returned.

### Training (`train`)

Training creates/evolves a population of candidate networks.

Each generation:

1. Fitness is computed by running all training samples and summing loss.
2. Optionally adds a penalty proportional to squared weights/biases.
3. The GA performs:
   - Tournament selection
   - Crossover: produces a child network by mixing weights and biases from two parents
   - Mutation: randomly applies one mutation operator to create a new child

At the end, the best evolved network is imported back into this.

## Loss Functions

Implemented loss functions include:

- MSE (Mean Squared Error)
- CrossEntropyLoss

## Mutations

Mutations in `Mutations` are defined as functions that modify a network:

- Modify Weights (`MOD_WEIGHT`)

  - Adds a small random delta to individual weights with some probability

- Reset Weights (`RES_WEIGHT`)

  - Replaces weights with random values within a range

- Modify Biases (`MOD_BIAS`)

  - Adds a small random delta to biases

- Reset Biases (`RES_BIAS`)

  - Replaces biases with random values

- Mutate Activation Functions (`MOD_ACTIVATION`)

  - Randomly changes a layer’s activation function from the allowed set

- Swap Nodes (`SWAP_NODES`)

  - Swaps biases and activation functions between randomly selected layers/nodes

Preset mutation sets:

- `FFW` (default if mutations is not specified): `[MOD_WEIGHT, MOD_BIAS, MOD_ACTIVATION, SWAP_NODES, RES_WEIGHT, RES_BIAS]`
- `ALL` includes the same set in this version (kept for extensibility)

## Usage Example

```ts
import { Network } from "@evolite/neural";

const net = new Network([2, 4, 1]);

const dataSet = [
  { inputs: [0, 0], outputs: [0] },
  { inputs: [0, 1], outputs: [1] },
  { inputs: [1, 0], outputs: [1] },
  { inputs: [1, 1], outputs: [0] },
];

await net.train(dataSet, { logging: true });
const out = net.activate([0, 1]);
console.log(out);
```

## Training Options

TrainingOptions includes:

- `epochs` (default: Infinity)
- `lossFunction` (default: MSELoss)
- `penalizeBigWeights` (default: true)
- `penalizeBigWeightsPenalty` (default: 0.00001)
- `mutations` (default: Mutations.FFW)
- GA options from `@evolite/core` you can see them at [here](https://evoliteorg.github.io/Core/)

### Note

If you want to stop the epochs at a determined loss change `fitnessObjective` to the desired loss

The default value is `0.001`

### Roadmap

- [ ] Implement tests for every class
- [ ] Add support for recurrent neurons
- [ ] Implement web workers / worker threads for faster training
- [ ] Implement more loss functions
- [ ] Implement more activation functions
- [ ] Make the networks architecture-free
