import { Layer } from "./Layer";
import { GeneticAlgorithm, type geneticAlgorithmOptions } from "@evolite/core";
import { type LossFunction } from "./LossFunctions";
import type { Mutation } from "./Mutations";
export type DataSet = {
    inputs: number[];
    outputs: number[];
}[];
export type TrainingOptions = {
    epochs?: number;
    lossFunction?: LossFunction;
    penalizeBigWeights?: boolean;
    penalizeBigWeightsPenalty?: number;
    mutations?: Mutation[];
} & Partial<geneticAlgorithmOptions<Network>>;
export declare class Network {
    layers: Layer[];
    private inputs;
    private outputs;
    fitness: number;
    private layerSizes;
    geneticAlgorithmClass: typeof GeneticAlgorithm;
    constructor(layerSizes: number[]);
    activate(inputs: number[]): number[];
    clone(): Network;
    import(network: Network): void;
    resetState(value?: number): void;
    train(dataSet: DataSet, options?: TrainingOptions): Promise<void>;
}
