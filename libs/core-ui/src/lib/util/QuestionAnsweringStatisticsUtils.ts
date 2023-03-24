// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { localization } from "@responsible-ai/localization";

import {
  ILabeledStatistic,
  TotalCohortSamples
} from "../Interfaces/IStatistic";

import { JointDataset } from "./JointDataset";

export enum QuestionAnsweringMetrics {
  ExactMatchRatio = "exactMatchRatio",
  F1Score = "f1Score"
}

function getf1Score(actual: string[], predicted: string[]): number {
  const truePositives = actual.filter((value) =>
    predicted.includes(value)
  ).length;
  const falsePositives = predicted.filter(
    (value) => !actual.includes(value)
  ).length;
  const falseNegatives = actual.filter(
    (value) => !predicted.includes(value)
  ).length;

  const precision = truePositives / (truePositives + falsePositives);
  const recall = truePositives / (truePositives + falseNegatives);

  return 2 * ((precision * recall) / (precision + recall));
}

export const generateQuestionAnsweringStats: (
  jointDataset: JointDataset,
  selectionIndexes: number[][]
) => ILabeledStatistic[][] = (
  jointDataset: JointDataset,
  selectionIndexes: number[][]
): ILabeledStatistic[][] => {
  const numLabels = jointDataset.numLabels;
  return selectionIndexes.map((selectionArray) => {
    const matchingLabels = [];
    const count = selectionArray.length;
    for (let i = 0; i < numLabels; i++) {
      const trueYs = jointDataset.unwrap(JointDataset.TrueYLabel + i);
      const predYs = jointDataset.unwrap(JointDataset.PredictedYLabel + i);

      const trueYSubset = selectionArray.map((i) => trueYs[i]);
      const predYSubset = selectionArray.map((i) => predYs[i]);
      matchingLabels.push(
        trueYSubset.filter((trueY, index) => trueY === predYSubset[index])
          .length
      );
    }
    const sum = matchingLabels.reduce((prev, curr) => prev + curr, 0);
    const exactMatchRatio = sum / (numLabels * selectionArray.length);

    const f1Score = getf1Score(
      jointDataset.unwrap(JointDataset.TrueYLabel),
      jointDataset.unwrap(JointDataset.PredictedYLabel)
    );

    return [
      {
        key: TotalCohortSamples,
        label: localization.Interpret.Statistics.samples,
        stat: count
      },
      {
        key: QuestionAnsweringMetrics.ExactMatchRatio,
        label: localization.Interpret.Statistics.exactMatchRatio,
        stat: exactMatchRatio
      },
      {
        key: QuestionAnsweringMetrics.F1Score,
        label: localization.Interpret.Statistics.f1Score,
        stat: f1Score
      }
    ];
  });
};
