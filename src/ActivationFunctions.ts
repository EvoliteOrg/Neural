export type ActivationFunction = (x: number[]) => number[];

export function sigmoid(x: number[]): number[] {
  return x.map((val) => 1 / (1 + Math.exp(-val)));
}
export function relu(x: number[]): number[] {
  return x.map((val) => Math.max(0, val));
}
export function tanh(x: number[]): number[] {
  return x.map((val) => Math.tanh(val));
}
export function identity(x: number[]): number[] {
  return x;
}
export function step(x: number[]): number[] {
  return x.map((val) => (val >= 0 ? 1 : 0));
}
export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const denom = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / denom);
}
export default {
  sigmoid,
  relu,
  tanh,
  identity,
  step,
  softmax,
};
