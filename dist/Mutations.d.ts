import type { Network } from "./Network";
import { type ActivationFunction } from "./ActivationFunctions";
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
export declare const Mutations: {
    ALL: RatedMutation[];
    FFW: RatedMutation[];
    MOD_WEIGHT: (RangedMutation & RatedMutation)[];
    MOD_BIAS: (RangedMutation & RatedMutation)[];
    MOD_ACTIVATION: (RatedMutation & {
        allowed: ActivationFunction[];
    })[];
    SWAP_NODES: RatedMutation[];
    RES_WEIGHT: (RangedMutation & RatedMutation)[];
    RES_BIAS: (RangedMutation & RatedMutation)[];
};
