import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from uqlm.utils import math_postprocessor
from uqlm.utils.tuner import Tuner


def extract_scores_from_df(df):
    scores_names = [
        "semantic_negentropy",
        "noncontradiction",
        "exact_match",
        "bert_score",
        "cosine_sim",
        "normalized_probability",
        "min_probability",
        "judge_1",
        "judge_2",
        "judge_3",
    ]

    score_lists = []
    for col in scores_names:
        if col in df.columns:
            score_lists.append(df[col].values.tolist())
    return score_lists


def grade_responses(name, df):
    n = len(df)
    answers = df["answer"].tolist()
    responses = df["response"].tolist()
    if name in ["gsm8k", "svamp"]:
        return [math_postprocessor(responses[i]) == answers[i] for i in range(n)]
    if name in ["ai2_arc", "csqa"]:
        return [responses[i] == answers[i] for i in range(n)]
    if name in ["nq_open", "popqa"]:
        return [_check_entailment(responses[i], answers[i]) for i in range(n)]


def compute_metric_values(correct_indicators, df):
    ignore_columns = [
        "prompt",
        "response",
        "sampled_responses",
        "answer",
        "confidence_score",
        "predicted_as_correct",
        "response_correct",
        "max_prob",
        "logprobs",
        "multiple_responses",
        "noncontradiction",
        "samples_valid",
        "self_reflection_cont",
        "judge_cont",
        "ensemble_score",
    ]

    tmp = dict()
    for col in df.columns:
        if col not in ignore_columns:
            scores = Scores()
            f1, p, r = scores.compute_fpr(y_true=correct_indicators, y_score=df[col])
            rocauc = scores.compute_rocauc(y_true=correct_indicators, y_score=df[col])
            tmp[col] = {"f1-score": f1, "precision": p, "recall": r, "rocauc": rocauc}
    return tmp


def kfold_crossvalidation(_score, _correct_indicators, n_params, n_t, k=5):
    kf = KFold(n_splits=k)
    f1_k, precision_k, recall_k, rocauc_k = [0] * k, [0] * k, [0] * k, [0] * k
    for i, (train_index, test_index) in enumerate(kf.split(_correct_indicators)):
        # 1. Split input data to train & test dataset
        train_score = [np.array(_s)[train_index].tolist() for _s in _score]
        test_score = [np.array(_s)[test_index].tolist() for _s in _score]
        train_correct = np.array(_correct_indicators)[train_index].tolist()
        test_correct = np.array(_correct_indicators)[test_index].tolist()

        # Different weights objective are used for UQEnsemble tunining for RocAUC and F1score computation
        for weights_objective in ["fbeta_score", "roc_auc"]:
            # 2. Predict ensemble scores on test dataset
            # 2a. Compute optimizaed weights
            tuner_object = Tuner()
            new_params = tuner_object.tune_params(
                score_lists=train_score,
                weights_objective=weights_objective,
                correct_indicators=train_correct,
                n_trials=n_t,
            )
            # 2b Compute predicted values
            test_score_pred = _compute_ensemble_scores(
                scores=test_score, weights=new_params["weights"]
            )

            # 3. Compute all scores on hold-out test datset
            if weights_objective == "fbeta_score":
                f1_k[i], precision_k[i], recall_k[i] = Scores().compute_fpr(
                    y_true=test_correct, y_score=test_score_pred
                )
            else:
                rocauc_k[i] = Scores().compute_rocauc(
                    y_true=test_correct, y_score=test_score_pred
                )

    return {
        "f1-score": np.mean(f1_k),
        "precision": np.mean(precision_k),
        "recall": np.mean(recall_k),
        "rocauc": np.mean(rocauc_k),
    }


class Scores:
    def __init__(self, threshold: float = 0):
        self.threshold = threshold

    def compute_fpr(self, y_true, y_score):
        f1 = self.tune_and_compute_f1score(y_true, y_score)
        p = self.compute_precision(y_true, y_score)
        r = self.compute_recall(y_true, y_score)
        return f1, p, r

    def tune_and_compute_f1score(self, y_true, y_score):
        best_score = 0
        for threshold in np.arange(0, 1, 0.01):
            y_pred = [1 if score > threshold else 0 for score in y_score]
            score = f1_score(y_true=y_true, y_pred=y_pred)
            if score > best_score:
                best_score = score
                self.threshold = threshold
        return best_score

    def compute_precision(self, y_true, y_score):
        y_pred = [1 if score > self.threshold else 0 for score in y_score]
        return precision_score(y_true=y_true, y_pred=y_pred)

    def compute_recall(self, y_true, y_score):
        y_pred = [1 if score > self.threshold else 0 for score in y_score]
        return recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def compute_rocauc(y_true, y_score):
        return roc_auc_score(y_true=y_true, y_score=y_score)


def _compute_ensemble_scores(scores, weights):
    """Helper function to compute dot product for getting ensemble scores"""
    return [
        sum(si * w for si, w in zip(scores_i, weights)) for scores_i in zip(*scores)
    ]


def _check_entailment(response: str, possible_answers: list[str]) -> bool:
    """Check entailment of possible answers in a response"""
    for s in possible_answers:
        if s.lower() in response.lower():
            return True
    return False
