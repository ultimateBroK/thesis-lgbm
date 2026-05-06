"""Model Performance section renderer."""

from __future__ import annotations

import streamlit as st

from thesis.charts import (
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_prediction_distribution_chart,
)
from thesis.dashboard.cards import render_metric_card
from thesis.dashboard.shared import render_chart


def render_model_section(data: dict, session_dir: str = "") -> None:
    """Render model performance metrics and model-analysis charts."""
    st.markdown("> 🏠 Dashboard > **Model Performance**")
    st.header("Model Performance")

    preds = data.get("predictions")
    fi = data.get("feature_importance", {})

    if preds is not None and len(preds) > 0:
        required_cols = {"true_label", "pred_label"}
        if not required_cols.issubset(set(preds.columns)):
            st.warning(
                f"Predictions missing columns: {required_cols - set(preds.columns)}"
            )
            return

        y_true = preds["true_label"].to_numpy()
        y_pred = preds["pred_label"].to_numpy()
        total = len(y_true)

        exact_acc = float((y_true == y_pred).mean())

        non_hold_mask = (y_true != 0) & (y_pred != 0)
        if non_hold_mask.sum() > 0:
            dir_correct = y_true[non_hold_mask] == y_pred[non_hold_mask]
            dir_acc = float(dir_correct.mean())
            dir_baseline = 0.5
        else:
            dir_acc = 0.0
            dir_baseline = 0.5

        per_class: dict[str, dict[str, float | int]] = {}
        for cls, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            true_mask = y_true == cls
            pred_mask = y_pred == cls
            recall = (
                float((y_pred[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
            )
            precision = (
                float((y_true[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[name] = {
                "true_count": int(true_mask.sum()),
                "pred_count": int(pred_mask.sum()),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }

        with st.container(border=True):
            st.subheader("Accuracy Metrics")
            st.caption("Model prediction accuracy against test set labels")

            acc_cols = st.columns(4, gap="small")
            render_metric_card(
                acc_cols[0],
                "Directional Accuracy",
                f"{dir_acc:.1%}",
                f"+{(dir_acc - dir_baseline) * 100:.1f}pp vs random",
                "#3b82f6",
            )
            render_metric_card(
                acc_cols[1],
                "Exact-Match Accuracy",
                f"{exact_acc:.1%}",
                None,
                "#8b5cf6",
            )
            render_metric_card(
                acc_cols[2],
                "Directional Baseline",
                f"{dir_baseline:.1%}",
                "Random guess baseline",
                "#6b7280",
            )
            render_metric_card(
                acc_cols[3],
                "Test Samples",
                f"{total:,}",
                None,
                "#10b981",
            )

        st.subheader("Per-Class Performance")
        cls_col1, cls_col2, cls_col3 = st.columns(3)
        for idx, (name, cls_metrics) in enumerate(per_class.items()):
            col = [cls_col1, cls_col2, cls_col3][idx]
            with col:
                st.markdown(f"**{name}**")
                st.caption(
                    f"True: {cls_metrics['true_count']:,}"
                    f" | Predicted: {cls_metrics['pred_count']:,}"
                )
                st.progress(
                    cls_metrics["recall"], text=f"Recall: {cls_metrics['recall']:.1%}"
                )
                st.progress(
                    cls_metrics["precision"],
                    text=f"Precision: {cls_metrics['precision']:.1%}",
                )
                st.progress(cls_metrics["f1"], text=f"F1: {cls_metrics['f1']:.2f}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            chart = build_confusion_matrix_chart(y_true, y_pred)
            render_chart(chart, height="500px")
        with col2:
            st.subheader("Confidence Distribution")
            chart = build_confidence_distribution_chart(preds)
            render_chart(chart, height="500px")

        st.subheader("Prediction Distribution")
        chart = build_prediction_distribution_chart(y_true, y_pred)
        render_chart(chart, height="400px")
    else:
        st.info("No predictions data available.")

    if fi:
        st.subheader("Feature Importance (Hybrid)")
        chart = build_feature_importance_chart(fi)
        render_chart(chart, height="600px")
    else:
        st.info("No feature importance data available.")
