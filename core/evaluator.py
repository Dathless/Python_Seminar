from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):

    accuracy = round(
        accuracy_score(y_true, y_pred) * 100,
        2
    )

    precision = round(
        precision_score(y_true, y_pred) * 100,
        2
    )

    recall = round(
        recall_score(y_true, y_pred) * 100,
        2
    )

    f1 = round(
        f1_score(y_true, y_pred) * 100,
        2
    )

    matrix = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = matrix.ravel()

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    for label in ['0', '1']:
        report[label]['f1score'] = round(
            report[label]['f1-score'],
            2
        )

        report[label]['precision'] = round(
            report[label]['precision'],
            2
        )

        report[label]['recall'] = round(
            report[label]['recall'],
            2
        )

        report[label]['support'] = int(
            report[label]['support']
        )

        report['weighted avg']['precision'] = round(
            report['weighted avg']['precision'],
            2
        )

        report['weighted avg']['recall'] = round(
            report['weighted avg']['recall'],
            2
        )

        report['weighted avg']['f1score'] = round(
            report['weighted avg']['f1-score'],
            2
        )

        report['weighted avg']['support'] = int(
            report['weighted avg']['support']
        )
        
        report['weighted_avg'] = report['weighted avg']

        report['macro_avg'] = report['macro avg']

        report['macro_avg']['f1score'] = round(
            report['macro avg']['f1-score'],
            2
        )

        report['weighted_avg']['f1score'] = round(
            report['weighted avg']['f1-score'],
            2
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "report": report
    }

import pandas as pd


def get_feature_importance(data):

    correlations = data.corr(numeric_only=True)["Outcome"]

    correlations = correlations.drop("Outcome")

    importance = []

    for feature, score in correlations.items():

        importance.append({
            "feature": feature,
            "score": round(abs(score), 2)
        })

    importance = sorted(
        importance,
        key=lambda x: x["score"],
        reverse=True
    )

    return importance