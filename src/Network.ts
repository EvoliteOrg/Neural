/* eslint-disable security/detect-object-injection */

import Activations from "./ActivationFunctions";
import { Layer } from "./Layer";
import {
  GeneticAlgorithm,
  Optimize,
  tournamentSelection,
  type geneticAlgorithmOptions,
} from "@evolite/core";
import { MSELoss, type LossFunction } from "./LossFunctions";
import type { Mutation } from "./Mutations";
import { Mutations } from "./Mutations";
export type DataSet = { inputs: number[]; outputs: number[] }[];
export type TrainingOptions = {
  epochs?: number;
  lossFunction?: LossFunction;
  penalizeBigWeights?: boolean;
  penalizeBigWeightsPenalty?: number;
  mutations?: Mutation[];
} & Partial<geneticAlgorithmOptions<Network>>;

export class Network {
  public layers: Layer[] = [];
  private inputs: number;
  private outputs: number;
  public fitness = 0;
  private layerSizes: number[] = [];
  public geneticAlgorithmClass = GeneticAlgorithm; //This allows for dependency injection of the GeneticAlgorithm class,making it easier to change the implementation of the genetic algorithm if needed.
  constructor(layerSizes: number[]) {
    if (layerSizes.length < 2)
      throw new Error(
        "Network must have at least 2 layers (input and output)."
      );
    this.layerSizes = layerSizes;
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(
        new Layer(layerSizes[i]!, layerSizes[i + 1]!, Activations.sigmoid)
      );
    }
    this.inputs = layerSizes[0]!;
    this.outputs = layerSizes[layerSizes.length - 1]!;
  }
  activate(inputs: number[]): number[] {
    if (inputs.length !== this.inputs)
      throw new Error(
        `Input size must be ${this.inputs}, but got ${inputs.length}.`
      );
    let currentInputs = inputs;
    for (let i = 0; i < this.layers.length; i++) {
      currentInputs = this.layers[i]!.calculate(currentInputs);
    }
    if (currentInputs.length !== this.outputs)
      throw new Error(
        `Output size must be ${this.outputs}, but got ${currentInputs.length}.`
      );
    return currentInputs;
  }
  clone(): Network {
    const clone = new Network(this.layerSizes);
    for (let i = 0; i < this.layers.length; i++) {
      clone.layers[i] = this.layers[i]!.clone();
    }
    return clone;
  }
  import(network: Network): void {
    this.layers = [];
    for (const layer of network.layers) {
      this.layers.push(layer.clone());
    }
    this.inputs = network.inputs;
    this.outputs = network.outputs;
  }
  resetState(value = 0): void {
    for (const layer of this.layers) {
      layer.resetState(value);
    }
  }
  async train(
    dataSet: DataSet,
    options: TrainingOptions = {
      initialPopulation: [this.clone(), this.clone()],
    }
  ) {
    if (!options.initialPopulation || options.initialPopulation.length < 2)
      options.initialPopulation = [this.clone(), this.clone()];
    if (!options.maxPopulationSize || options.maxPopulationSize < 2)
      options.maxPopulationSize = 100;
    if (!options.mutations || options.mutations.length === 0)
      options.mutations = Mutations.FFW;
    if (options.penalizeBigWeights === undefined)
      options.penalizeBigWeights = true;
    if (options.penalizeBigWeightsPenalty === undefined)
      options.penalizeBigWeightsPenalty = 0.00001;
    if (options.fitnessObjective === undefined)
      options.fitnessObjective = 0.001;
    if (options.mutationRate === undefined) options.mutationRate = 1;
    const objective = options.penalizeBigWeights
      ? options.fitnessObjective + options.penalizeBigWeightsPenalty
      : options.fitnessObjective;
    const ga = new this.geneticAlgorithmClass({
      ...options,
      logging: false,
      initialPopulation: options.initialPopulation,
      optimization: Optimize.Minimize,
      fitnessObjective: objective,
    });
    ga.setCrossoverMethod(async (parent1, parent2) => {
      const child = parent1.clone();

      const minLayers = Math.min(parent1.layers.length, parent2.layers.length);

      for (let i = 0; i < minLayers; i++) {
        const cLayer = child.layers[i];
        const p2Layer = parent2.layers[i];

        if (!cLayer?.weights || !p2Layer?.weights) continue;

        const cW = cLayer.weights;
        const p2W = p2Layer.weights;
        if (cW.length !== p2W.length) continue;

        for (let j = 0; j < cW.length; j++) {
          if (!p2W[j] || !cW[j] || cW[j]!.length !== p2W[j]!.length) continue;

          for (let k = 0; k < cW[j]!.length; k++) {
            if (Math.random() < 0.5) cW[j]![k] = p2W[j]![k]!;
          }
        }
        if (
          cLayer.biases &&
          p2Layer.biases &&
          cLayer.biases.length === p2Layer.biases.length
        ) {
          for (let b = 0; b < cLayer.biases.length; b++) {
            if (Math.random() < 0.5) cLayer.biases[b] = p2Layer.biases[b]!;
          }
        }
      }

      return child;
    });
    const lossFunction = options.lossFunction ?? MSELoss;
    ga.setFitnessFunction(async (network) => {
      network.resetState();
      let totalError = 0;
      for (let i = 0; i < dataSet.length; i++) {
        const input = dataSet[i]!.inputs;
        const expectedOutput = dataSet[i]!.outputs;
        const actualOutput = network.activate(input);
        totalError += lossFunction(expectedOutput, actualOutput);
      }
      if (options.penalizeBigWeights)
        totalError +=
          this.layers.map((layer) => layer.loss()).reduce((a, b) => a + b, 0) *
          options.penalizeBigWeightsPenalty!;
      return totalError;
    });
    ga.setSelectionMethod(tournamentSelection);
    ga.setMutationMethod(async (network) => {
      const child = network.clone();
      if (options.mutations === undefined || options.mutations.length === 0)
        throw new Error(
          "No mutations provided. Please provide at least one mutation in the training options."
        );
      const mutation =
        options.mutations[Math.floor(Math.random() * options.mutations.length)];
      mutation?.mutate(child);

      return child;
    });
    const fittest = await ga.evolve(
      options.epochs ?? Infinity,
      async (generation, _population, fittest) => {
        if (
          options.logging &&
          generation % (options.loggingInterval ?? 1) === 0
        ) {
          console.log(`epoch: ${generation}, loss: ${-fittest.fitness}`);
        }
      }
    );

    if (options.logging)
      console.log(`Epoch: ${ga.generation}, Loss: ${-fittest.fitness}`);
    this.import(fittest);
  }
}
