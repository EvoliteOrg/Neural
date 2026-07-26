export type LossFunction = (predictions: number[], targets: number[]) => number;
export const MSELoss: LossFunction = (predictions, targets) => {
  if (predictions.length !== targets.length)
    throw new Error("Predictions and targets must have the same length.");
  const n = predictions.length;
  const sumSquaredError = predictions.reduce((sum, pred, i) => {
    const error = pred - targets.at(i)!;
    return sum + error * error;
  }, 0);
  return sumSquaredError / n;
};
export const CrossEntropyLoss: LossFunction = (predictions, targets) => {
  if (predictions.length !== targets.length)
    throw new Error("Predictions and targets must have the same length.");
  const n = predictions.length;
  const epsilon = 1e-15;
  const loss = predictions.reduce((sum, pred, i) => {
    const target = targets.at(i)!;
    return (
      sum -
      (target * Math.log(pred + epsilon) +
        (1 - target) * Math.log(1 - pred + epsilon))
    );
  }, 0);
  return loss / n;
};
export const MAELoss: LossFunction = (predictions, targets) => {
  if (predictions.length !== targets.length)
    throw new Error("Predictions and targets must have the same length.");
  const n = predictions.length;
  const sumAbsoluteError = predictions.reduce((sum, pred, i) => {
    const error = Math.abs(pred - targets.at(i)!);
    return sum + error;
  }, 0);
  return sumAbsoluteError / n;
};
