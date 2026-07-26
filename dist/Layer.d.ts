import type { ActivationFunction } from "./ActivationFunctions";
export declare class Layer {
    weights: number[][];
    biases: number[];
    activationFunction: ActivationFunction;
    hiddenWeights: number[][];
    hiddenState: number[];
    constructor(inputSize: number, outputSize: number, activationFunction: ActivationFunction);
    calculate(inputs: number[]): number[];
    resetState(value?: number): void;
    clone(): Layer;
    loss(): number;
}
