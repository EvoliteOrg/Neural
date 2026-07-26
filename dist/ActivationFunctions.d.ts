export type ActivationFunction = (x: number[]) => number[];
export declare function sigmoid(x: number[]): number[];
export declare function relu(x: number[]): number[];
export declare function tanh(x: number[]): number[];
export declare function identity(x: number[]): number[];
export declare function step(x: number[]): number[];
export declare function softmax(xs: number[]): number[];
declare const _default: {
    sigmoid: typeof sigmoid;
    relu: typeof relu;
    tanh: typeof tanh;
    identity: typeof identity;
    step: typeof step;
    softmax: typeof softmax;
};
export default _default;
