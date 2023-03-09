// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { localization } from "@responsible-ai/localization";

import {
  ILabeledStatistic,
  TotalCohortSamples
} from "../Interfaces/IStatistic";

import { JointDataset } from "./JointDataset";

export enum QuestionAnsweringMetrics {
  ExactMatchRatio = "exactMatchRatio"
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
      }
    ];
  });
};
