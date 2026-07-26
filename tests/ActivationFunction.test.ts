/* eslint-disable security/detect-object-injection */
import { describe, expect, test } from "bun:test";
import acts, {
  sigmoid,
  relu,
  tanh,
  identity,
  step,
  softmax,
} from "../src/ActivationFunctions";
describe("activation functions - elementwise ops", () => {
  test("identity returns the same values (by equality)", () => {
    const x = [1, -2, 0.5];
    expect(identity(x)).toEqual([1, -2, 0.5]);
  });

  test("relu clamps negatives to 0", () => {
    const x = [-3, -0.1, 0, 2.5];
    expect(relu(x)).toEqual([0, 0, 0, 2.5]);
  });

  test("step returns 1 for val >= 0 else 0", () => {
    const x = [-0.01, 0, 0.01];
    expect(step(x)).toEqual([0, 1, 1]);
  });

  test("tanh matches Math.tanh elementwise", () => {
    const x = [-2, -1, 0, 1, 2];
    const out = tanh(x);
    for (let i = 0; i < x.length; i++) {
      expect(out[i]).toBeCloseTo(Math.tanh(x[i]!), 10);
    }
  });
});

describe("sigmoid", () => {
  test("sigmoid of 0 is 0.5", () => {
    expect(sigmoid([0])).toEqual([0.5]);
  });

  test("sigmoid matches formula elementwise", () => {
    const x = [-10, -1, 0, 1, 10];
    const out = sigmoid(x);
    for (let i = 0; i < x.length; i++) {
      const expected = 1 / (1 + Math.exp(-x[i]!));
      expect(out[i]).toBeCloseTo(expected, 12);
    }
  });

  test("sigmoid outputs are in (0, 1)", () => {
    const x = [-100, -2, 0, 2, 100];
    const out = sigmoid(x);
    for (const v of out) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("softmax", () => {
  test("softmax sums to 1 (within tolerance)", () => {
    const x = [1, 2, 3];
    const out = softmax(x);
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 12);
  });

  test("softmax outputs are all positive", () => {
    const x = [-10, 0, 10];
    const out = softmax(x);
    for (const v of out) {
      expect(v).toBeGreaterThan(0);
    }
  });

  test("softmax prefers larger inputs", () => {
    const x = [0, 0, 10];
    const out = softmax(x);
    expect(out[2]).toBeGreaterThan(out[0]!);
    expect(out[2]).toBeGreaterThan(out[1]!);
  });

  test("softmax handles large values without overflow (numerical stability)", () => {
    const x = [1000, 1001, 1002];
    const out = softmax(x);
    expect(out.every((v) => !Number.isNaN(v))).toBe(true);
    expect(out[2]).toBeGreaterThan(out[0]!);
  });

  test("softmax of a single element is 1", () => {
    expect(softmax([42])).toEqual([1]);
  });

  test("softmax is invariant to adding a constant offset", () => {
    const x = [1.2, -0.3, 0.7];
    const out1 = softmax(x);
    const out2 = softmax(x.map((v) => v + 5)); // shift by constant
    for (let i = 0; i < out1.length; i++) {
      expect(out2[i]).toBeCloseTo(out1[i]!, 12);
    }
  });

  test("softmax of equal inputs is uniform", () => {
    const x = [2, 2, 2, 2];
    const out = softmax(x);
    const expected = 1 / x.length;
    for (const v of out) {
      expect(v).toBeCloseTo(expected, 12);
    }
  });

  test("softmax handles two-element input", () => {
    const x = [0, 0]; // should be [0.5, 0.5]
    expect(softmax(x)).toEqual([0.5, 0.5]);
  });
});

describe("exports", () => {
  test("default export contains all functions", () => {
    expect(acts).toHaveProperty("sigmoid");
    expect(acts).toHaveProperty("relu");
    expect(acts).toHaveProperty("tanh");
    expect(acts).toHaveProperty("identity");
    expect(acts).toHaveProperty("step");
    expect(acts).toHaveProperty("softmax");
  });
});
