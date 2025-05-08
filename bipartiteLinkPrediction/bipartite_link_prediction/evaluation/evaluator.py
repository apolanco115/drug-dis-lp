import csv

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Tuple, Iterable


class Evaluator:
    @staticmethod
    def calculate_roc(y_true: Iterable[int], y_scores: Iterable[float]) -> Tuple[Iterable[float], Iterable[float], Iterable[float]]:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        """
        Calculate the ROC curve.
        
        Args:
        y_true (Iterable[int]): The true labels.
        y_scores (Iterable[float]): The predicted scores.
        
        Returns:
        Tuple[Iterable[float], Iterable[float], Iterable[float]]: The false positive rate, true positive rate, and thresholds values.
        """


        return fpr, tpr, thresholds

    @staticmethod
    def calculate_aucroc(fpr: Iterable[float], tpr: Iterable[float]) -> float:
        """
        Calculate the area under the ROC curve.
        Args:
        fpr (Iterable[float]): The false positive rate values.
        tpr (Iterable[float]): The true positive rate values.

        Returns:
        float: The area under the ROC curve.
        """
        return auc(fpr, tpr)

    @staticmethod
    def calculate_pr_curve(y_true: Iterable[int], y_scores: Iterable[float]) -> Tuple[Iterable[float], Iterable[float], Iterable[float]]:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        """
        Calculate the precision-recall curve.
        
        Args:
        y_true (Iterable[int]): The true labels.
        y_scores (Iterable[float]): The predicted scores.
        
        Returns:
        Tuple[Iterable[float], Iterable[float], Iterable[float]]: The precision, recall, and thresholds values
        """


        return precision[::-1], recall[::-1], thresholds

    @staticmethod
    def calculate_aucpr(precision: Iterable[float], recall: Iterable[float]) -> float:
        """
        Calculate the area under the precision-recall curve.
        Args:
        precision (Iterable[float]): The precision values.
        recall (Iterable[float]): The recall values.

        Returns:
        float: The area under the precision-recall curve.
        """
        return auc(recall, precision)

    @staticmethod
    def calculate_average_precision(y_true: Iterable[int], y_scores: Iterable[float]) -> float:
        """
        Calculate the average precision score.
        Args:
        y_true (Iterable[int]): The true labels.
        y_scores (Iterable[float]): The predicted scores.

        Returns:
        float: The average precision score.
        """
        return average_precision_score(y_true, y_scores)

    @staticmethod
    def calculate_average(values: Iterable[float]) -> float:
        """
        Calculate the average of a list of values.
        Args:
        values (Iterable[float]): The values to calculate the average for.

        Returns:
        float: The average of the values.
        """
        return np.mean(values)

    @staticmethod
    def top_k_precision(y_true, y_score, k=100):
        """
        Calculate the top-k precision. Code copied from: https://gist.github.com/mblondel/7337391

        Args:
        y_true (np.ndarray): The true labels.
        y_score (np.ndarray): The predicted scores.
        k (int): The number of top predictions to consider.

        Returns:
        float: The top-k precision.
        """
        unique_y = np.unique(y_true)

        if len(unique_y) > 2:
            raise ValueError("Only supported for two relevance levels.")

        pos_label = unique_y[1]
        n_pos = np.sum(y_true == pos_label)

        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        n_relevant = np.sum(y_true == pos_label)

        return float(n_relevant) / min(n_pos, k)

    @staticmethod
    def calculate_standard_error(values: Iterable[float]) -> float:
        """
        Calculate the standard error of the mean.
        Args:
        values (Iterable[float]): The values to calculate the standard error for.

        Returns:
        float: The standard error of the mean.
        """
        return stats.sem(values)

    @staticmethod
    def interpolate_curve(x_values: Iterable[float], y_values: Iterable[float], num_points: int) -> np.ndarray:
        """
        Interpolate the y values of a curve at the given base_x points.
        Args:
        x_values (Iterable[float]): The x values of the curve.
        y_values (Iterable[float]): The y values of the curve.
        num_points (int): The number of points to interpolate.

        Returns:
        np.ndarray: The interpolated y values.
        """
        base_x = np.linspace(0, 1, num_points)
        y_interp = np.interp(base_x, x_values, y_values)

        return base_x, y_interp

    @staticmethod
    def calculate_average_curve(
            curve_list: Iterable[Tuple[Iterable[float], Iterable[float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the average curve from a list of curves with identical x values.

        Args:
        curve_list (List[Tuple[np.ndarray, np.ndarray]]): A list of tuples, where each tuple
            contains a numpy array of x values and a numpy array of y values for a curve.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the common x values and the average y values.
        """
        base_x = curve_list[0][0]
        average_y = np.zeros_like(base_x)
        for _, y in curve_list:
            average_y+=y
        average_y /= len(curve_list)

        return base_x, average_y
    @staticmethod
    def write_curve(x: Iterable[float], y: Iterable[float], filename: str = "data.csv", x_header: str = 'x', y_header: str = 'y') -> None:
        """
        Save a pair of related data lists to a CSV file.

        Args:
        x (Iterable[float]): The x values.
        y (Iterable[float]): The y values.
        filename (str): The name of the file to save the data to.
        x_header (str): The header for the x values.
        y_header (str): The header for the y values.

        Returns:
        None
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x_header, y_header])
            for fpr_value, tpr_value in zip(x, y):
                writer.writerow([fpr_value, tpr_value])
