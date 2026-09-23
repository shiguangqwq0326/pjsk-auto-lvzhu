"""Fit a conservative size-based correction for anime-hand detection scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent / "calibration"
BASE_CUTOFF = 0.35


def area_fraction(candidate: dict) -> float:
    left, top, right, bottom = candidate["box_xyxy"]
    return (max(0, right - left) * max(0, bottom - top) /
            (candidate["width"] * candidate["height"]))


def border_margin(candidate: dict) -> int:
    left, top, right, bottom = candidate["box_xyxy"]
    return min(left, top, candidate["width"] - right,
               candidate["height"] - bottom)


def corrected_score(candidate: dict, size_limit: float, power: float,
                    border_limit: float | None = None,
                    border_scale: float = 1.0) -> float:
    area = area_fraction(candidate)
    score = candidate["raw_confidence"] * min(
        1.0, (size_limit / max(area, 1e-9)) ** power)
    if border_limit is not None and border_margin(candidate) <= border_limit:
        score *= border_scale
    return score


def measure(ids: list[int], positives: set[int], candidates: list[dict],
            size_limit: float | None = None, power: float = 2.0,
            border_limit: float | None = None,
            border_scale: float = 1.0) -> dict:
    tp = fp = fn = tn = 0
    for i in ids:
        candidate = candidates[i]
        score = (candidate["raw_confidence"] if size_limit is None else
                 corrected_score(candidate, size_limit, power,
                                 border_limit, border_scale))
        predicted = score >= BASE_CUTOFF
        actual = i in positives
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": tn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def main() -> None:
    candidates = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "reviewed_labels.json").read_text(encoding="utf-8"))
    positives = set(labels["positive_ids"])
    negatives = set(labels["negative_ids"])
    trusted = set(labels["confirmed_by_user"])
    trusted_negatives = set(labels.get("confirmed_negative_ids", []))
    assert (not positives & negatives and trusted <= positives and
            trusted_negatives <= negatives)
    ids = sorted(positives | negatives)
    groups = [candidates[i]["image"] for i in ids]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=17)
    train_positions, test_positions = next(splitter.split(ids, groups=groups))
    trusted_groups = {candidates[i]["image"] for i in trusted | trusted_negatives}
    test_ids = [ids[j] for j in test_positions if groups[j] not in trusted_groups]
    test_set = set(test_ids)
    train_ids = [i for i in ids if i not in test_set]
    assert {candidates[i]["image"] for i in train_ids}.isdisjoint(
        {candidates[i]["image"] for i in test_ids})

    train_max_positive_area = max(area_fraction(candidates[i])
                                  for i in train_ids if i in positives)
    # Keep a 2.5 percentage-point safety margin above every reviewed hand.
    size_limit = math.ceil((train_max_positive_area + 0.025) * 100) / 100
    train_baseline = measure(train_ids, positives, candidates)
    choices = []
    for power in (1.0, 2.0, 3.0):
        result = measure(train_ids, positives, candidates, size_limit, power)
        if result["false_negative"] == train_baseline["false_negative"]:
            choices.append((result["f1"], -power, power))
    if not choices:
        raise RuntimeError("没有找到能保持已核对手部召回率的校准参数")
    power = max(choices)[2]
    held_out_baseline = measure(test_ids, positives, candidates)
    held_out_corrected = measure(test_ids, positives, candidates, size_limit, power)
    if (held_out_corrected["false_negative"] > held_out_baseline["false_negative"]
            or held_out_corrected["false_positive"] > held_out_baseline["false_positive"]):
        raise RuntimeError("校准在留出页面上变差，未发布该修正")

    def learn_border_rule(positive_ids: set[int], size: float) -> tuple[float, float]:
        if not trusted_negatives:
            return -1.0, 1.0
        nearest_positive = min(border_margin(candidates[i]) for i in positive_ids)
        farthest_negative = max(border_margin(candidates[i]) for i in trusted_negatives)
        if farthest_negative >= nearest_positive:
            raise RuntimeError("边缘误检与已核对手部距离重叠，无法安全学习边缘规则")
        limit = (farthest_negative + nearest_positive) / 2
        highest_negative_score = max(corrected_score(candidates[i], size, power)
                                     for i in trusted_negatives)
        scale = min(1.0, BASE_CUTOFF * 0.95 / highest_negative_score)
        return limit, scale

    train_border_limit, train_border_scale = learn_border_rule(
        positives & set(train_ids), size_limit)
    held_out_border = measure(test_ids, positives, candidates, size_limit, power,
                              train_border_limit, train_border_scale)
    if (held_out_border["false_negative"] > held_out_corrected["false_negative"] or
            held_out_border["false_positive"] > held_out_corrected["false_positive"]):
        raise RuntimeError("边缘校准在留出页面上变差，未发布该修正")

    # After evaluation, refit the size limit on every reviewed positive example.
    final_limit = math.ceil((max(area_fraction(candidates[i]) for i in positives)
                             + 0.025) * 100) / 100
    final_border_limit, final_border_scale = learn_border_rule(positives, final_limit)
    model = {"type": "size_border_score_correction", "size_limit": final_limit,
             "power": power, "cutoff": BASE_CUTOFF,
             "border_margin_limit_px": final_border_limit,
             "border_score_scale": round(final_border_scale, 6),
             "formula": "raw_score * min(1, (size_limit / area_fraction) ** power) * (border_score_scale if min_border_margin_px <= border_margin_limit_px else 1)",
             "image_count": 68, "reviewed_positive_count": len(positives),
             "reviewed_negative_count": len(negatives),
             "trusted_positive_count": len(trusted),
             "trusted_negative_count": len(trusted_negatives)}
    report = {"model": model, "training_pages": len({candidates[i]["image"] for i in train_ids}),
              "held_out_pages": len({candidates[i]["image"] for i in test_ids}),
              "training_baseline": train_baseline,
              "training_corrected": measure(train_ids, positives, candidates,
                                            size_limit, power),
              "held_out_baseline": held_out_baseline,
              "held_out_corrected": held_out_corrected,
              "held_out_border_corrected": held_out_border,
              "all_reviewed_baseline": measure(ids, positives, candidates),
              "all_reviewed_corrected": measure(ids, positives, candidates,
                                                final_limit, power),
              "all_reviewed_border_corrected": measure(
                  ids, positives, candidates, final_limit, power,
                  final_border_limit, final_border_scale),
              "unreviewed_candidates": len(candidates) - len(ids),
              "note": "仅评估逐框核对的候选，未测未检出的手；校正分数不是概率。"}
    (ROOT / "confidence_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
