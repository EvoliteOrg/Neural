/* eslint-disable security/detect-object-injection */
import type { Network } from "./Network";
import Activations, { type ActivationFunction } from "./ActivationFunctions";

export interface Mutation {
  mutate(network: Network): void;
}

export interface RangedMutation extends Readonly<Mutation> {
  min: number;
  max: number;
}
export interface RatedMutation extends Readonly<Mutation> {
  rate: number;
}

const MOD_WEIGHT: RangedMutation & RatedMutation = {
  min: -0.01,
  max: 0.01,
  rate: 0.1,
  mutate(network: Network): void {
    for (let i = 0; i < network.layers.length; i++) {
      for (let j = 0; j < network.layers[i]!.weights.length; j++) {
        for (let k = 0; k < network.layers[i]!.weights[j]!.length; k++) {
          const random = Math.random();
          if (random < this.rate) {
            network.layers[i]!.weights[j]![k]! +=
              Math.random() * (this.max - this.min) + this.min;
          }
        }
      }
    }
  },
};

const RES_WEIGHT: RangedMutation & RatedMutation = {
  min: -1,
  max: 1,
  rate: 0.01,
  mutate(network: Network): void {
    for (let i = 0; i < network.layers.length; i++) {
      for (let j = 0; j < network.layers[i]!.weights.length; j++) {
        for (let k = 0; k < network.layers[i]!.weights[j]!.length; k++) {
          const random = Math.random();
          if (random < this.rate) {
            network.layers[i]!.weights[j]![k]! =
              Math.random() * (this.max - this.min) + this.min;
          }
        }
      }
    }
  },
};

const MOD_BIAS: RangedMutation & RatedMutation = {
  min: -0.01,
  max: 0.01,
  rate: 0.1,
  mutate(network: Network): void {
    for (let i = 0; i < network.layers.length; i++) {
      for (let j = 0; j < network.layers[i]!.biases.length; j++) {
        const random = Math.random();
        if (random < this.rate) {
          network.layers[i]!.biases[j]! +=
            Math.random() * (this.max - this.min) + this.min;
        }
      }
    }
  },
};

const RES_BIAS: RangedMutation & RatedMutation = {
  min: -1,
  max: 1,
  rate: 0.01,
  mutate(network: Network): void {
    for (let i = 0; i < network.layers.length; i++) {
      for (let j = 0; j < network.layers[i]!.biases.length; j++) {
        const random = Math.random();
        if (random < this.rate) {
          network.layers[i]!.biases[j]! =
            Math.random() * (this.max - this.min) + this.min;
        }
      }
    }
  },
};

const MOD_ACTIVATION: RatedMutation & { allowed: ActivationFunction[] } = {
  allowed: [...Object.values(Activations)],
  rate: 0.1,
  mutate(network: Network): void {
    for (let i = 0; i < network.layers.length; i++) {
      const random = Math.random();
      if (random < this.rate) {
        const randomActivation =
          this.allowed[Math.floor(Math.random() * this.allowed.length)]!;
        network.layers[i]!.activationFunction = randomActivation;
      }
    }
  },
};

const SWAP_NODES: RatedMutation = {
  rate: 0.1,
  mutate(network: Network): void {
    const layerIndex1 = Math.floor(Math.random() * network.layers.length);
    const layerIndex2 = Math.floor(Math.random() * network.layers.length);
    const nodeIndex1 = Math.floor(
      Math.random() * network.layers[layerIndex1]!.weights.length
    );
    const nodeIndex2 = Math.floor(
      Math.random() * network.layers[layerIndex2]!.weights.length
    );

    const tempBias = network.layers[layerIndex1]!.biases[nodeIndex1]!;
    network.layers[layerIndex1]!.biases[nodeIndex1]! =
      network.layers[layerIndex2]!.biases[nodeIndex2]!;
    network.layers[layerIndex2]!.biases[nodeIndex2]! = tempBias;

    const tempActivation = network.layers[layerIndex1]!.activationFunction;
    network.layers[layerIndex1]!.activationFunction =
      network.layers[layerIndex2]!.activationFunction;
    network.layers[layerIndex2]!.activationFunction = tempActivation;
  },
};

export const Mutations = {
  ALL: [MOD_WEIGHT, MOD_BIAS, MOD_ACTIVATION, SWAP_NODES, RES_WEIGHT, RES_BIAS],
  FFW: [MOD_WEIGHT, MOD_BIAS, MOD_ACTIVATION, SWAP_NODES, RES_WEIGHT, RES_BIAS],
  MOD_WEIGHT: [MOD_WEIGHT],
  MOD_BIAS: [MOD_BIAS],
  MOD_ACTIVATION: [MOD_ACTIVATION],
  SWAP_NODES: [SWAP_NODES],
  RES_WEIGHT: [RES_WEIGHT],
  RES_BIAS: [RES_BIAS],
};
