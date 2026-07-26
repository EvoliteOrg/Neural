export type LossFunction = (predictions: number[], targets: number[]) => number;
export declare const MSELoss: LossFunction;
export declare const CrossEntropyLoss: LossFunction;
export declare const MAELoss: LossFunction;
