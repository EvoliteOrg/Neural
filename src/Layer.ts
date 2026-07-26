/* eslint-disable security/detect-object-injection */
import type { ActivationFunction } from "./ActivationFunctions";

export class Layer {
  public weights: number[][];
  public biases: number[];
  public activationFunction: ActivationFunction;
  public hiddenWeights: number[][] = [];
  public hiddenState: number[] = [];
  constructor(
    inputSize: number,
    outputSize: number,
    activationFunction: ActivationFunction
  ) {
    this.weights = new Array(outputSize);
    this.activationFunction = activationFunction;
    for (let i = 0; i < outputSize; i++) {
      this.weights[i] = new Array(inputSize);
      for (let j = 0; j < inputSize; j++) {
        this.weights[i]![j] = Math.random() * 2 - 1;
      }
    }
    this.biases = new Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
      this.biases[i] = Math.random() * 2 - 1;
    }
    this.hiddenWeights = new Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
      this.hiddenWeights[i] = new Array(outputSize);
      for (let j = 0; j < outputSize; j++) {
        this.hiddenWeights[i]![j] = 0;
      }
    }
    this.hiddenState = new Array(outputSize).fill(0);
  }
  calculate(inputs: number[]): number[] {
    const outputs: number[] = new Array(this.weights.length);
    for (let i = 0; i < this.weights.length; i++) {
      let sum = 0;
      for (let j = 0; j < inputs.length; j++) {
        sum += this.weights[i]![j]! * inputs[j]!;
      }
      for (let j = 0; j < this.hiddenWeights.length; j++) {
        sum += this.hiddenWeights[i]![j]! * this.hiddenState[j]!;
      }

      outputs[i] = sum + this.biases[i]!;
    }
    const activatedOutputs = this.activationFunction(outputs);
    this.hiddenState = activatedOutputs;
    return activatedOutputs;
  }
  resetState(value = 0) {
    this.hiddenState?.fill(value);
  }
  clone(): Layer {
    const clone = new Layer(
      this.weights[0]!.length,
      this.weights.length,
      this.activationFunction
    );
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i]!.length; j++) {
        clone.weights[i]![j]! = this.weights[i]![j]!;
      }
    }
    for (let i = 0; i < this.biases.length; i++) {
      clone.biases[i]! = this.biases[i]!;
    }
    for (let i = 0; i < this.hiddenWeights.length; i++) {
      for (let j = 0; j < this.hiddenWeights[i]!.length; j++) {
        clone.hiddenWeights[i]![j]! = this.hiddenWeights[i]![j]!;
      }
    }
    return clone;
  }
  loss() {
    let loss = 0;
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i]!.length; j++) {
        const value = this.weights[i]![j]!;

        const biasValue = this.biases[i]!;
        loss += value * value;
        loss += biasValue * biasValue;
      }
    }
    for (let i = 0; i < this.hiddenWeights.length; i++) {
      for (let j = 0; j < this.hiddenWeights[i]!.length; j++) {
        const hiddenValue = this.hiddenWeights[i]![j]!;
        loss += hiddenValue * hiddenValue;
      }
    }
    return loss;
  }
}
