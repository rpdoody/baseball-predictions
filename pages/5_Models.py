import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


from page_utils import (
    READABLE_COLS,
    _load_precomputed,
    add_betting_oracle_footer,
    init_session_state,
    render_sidebar,
)
from retrosheet import MODERN_START, load_gameinfo, season_standings
from src.evaluation.calibration import calibration_plot_data


min_year, max_year = render_sidebar()

_pre = _load_precomputed()

_pre_gameinfo_max_year = int(_pre["gameinfo"]["season"].max())

_live_gameinfo = load_gameinfo(
    min_year=MODERN_START,
    max_year=datetime.date.today().year,
)

_GAMEINFO_MAX_YEAR = max(
    _pre_gameinfo_max_year,
    int(_live_gameinfo["season"].max()) if not _live_gameinfo.empty else _pre_gameinfo_max_year,
)

features_df = _pre["model_features"][
    _pre["model_features"]["season"].between(min_year, max_year)
].copy()
init_session_state(features_df=features_df)


def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """Calculate the optimal height for a Streamlit dataframe based on rows."""
    calculated_height = (len(df) * row_height) + header_height + padding
    return min(calculated_height, max_height) if max_height is not None else calculated_height


def _build_feature_matrix(gi: pd.DataFrame, ts_yr: pd.DataFrame) -> pd.DataFrame:
    """Build season features with vectorized home/visitor standings joins."""
    gi_valid = gi[gi["hometeam"].isin(ts_yr.index) & gi["visteam"].isin(ts_yr.index)].copy()
    if gi_valid.empty:
        return pd.DataFrame()

    stat_cols = ["WPct", "RS_per_G", "RA_per_G", "RD_per_G", "PythWPct"]
    home_stats = ts_yr[stat_cols].add_prefix("home_")
    vis_stats = ts_yr[stat_cols].add_prefix("vis_")

    merged = gi_valid.merge(home_stats, left_on="hometeam", right_index=True).merge(
        vis_stats, left_on="visteam", right_index=True
    )

    merged["WPct_diff"] = merged["home_WPct"] - merged["vis_WPct"]
    merged["PythWPct_diff"] = merged["home_PythWPct"] - merged["vis_PythWPct"]
    merged["RS_advantage"] = merged["home_RS_per_G"] - merged["vis_RS_per_G"]
    merged["RA_advantage"] = merged["vis_RA_per_G"] - merged["home_RA_per_G"]
    merged["home_win"] = (merged["wteam"] == merged["hometeam"]).astype(int)

    merged = merged.rename(
        columns={
            "visteam": "visitor",
            "hometeam": "home_team",
            "home_RS_per_G": "home_RS_G",
            "home_RA_per_G": "home_RA_G",
            "home_RD_per_G": "home_RD_G",
            "vis_RS_per_G": "vis_RS_G",
            "vis_RA_per_G": "vis_RA_G",
            "vis_RD_per_G": "vis_RD_G",
        }
    )

    keep_cols = [
        "date",
        "visitor",
        "home_team",
        "home_WPct",
        "vis_WPct",
        "home_RS_G",
        "home_RA_G",
        "vis_RS_G",
        "vis_RA_G",
        "home_RD_G",
        "vis_RD_G",
        "home_PythWPct",
        "vis_PythWPct",
        "WPct_diff",
        "PythWPct_diff",
        "RS_advantage",
        "RA_advantage",
        "daynight",
        "attendance",
        "temp",
        "windspeed",
        "home_win",
        "total_runs",
    ]
    return merged[[col for col in keep_cols if col in merged.columns]].reset_index(drop=True)


tab_feat, tab_models, tab_eval, tab_savant = st.tabs(
    ["Betting Features", "ML Models", "Model Evaluation", "Savant Research"]
)


# ── Betting Features ──────────────────────────────────────────────────────────
with tab_feat:
    st.subheader("Engineered Betting Features")
    st.markdown("Feature matrix built from season-level stats — designed as inputs for ML models.")

    # Historical standings come from the precomputed artifact. Add live
    # standings so the current season is available without rebuilding it.
    all_standings = _pre["standings"].copy()

    live_standings = season_standings(
        min_year=MODERN_START,
        max_year=datetime.date.today().year,
    )

    precomputed_years = set(
        pd.to_numeric(
            all_standings["season"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    live_standings = live_standings[~live_standings["season"].isin(precomputed_years)].copy()

    all_standings = pd.concat(
        [all_standings, live_standings],
        ignore_index=True,
        sort=False,
    )

    all_standings = _pre["standings"].copy()
    all_standings["season"] = pd.to_numeric(
        all_standings["season"],
        errors="coerce",
    )

    live_standings = season_standings(
        min_year=MODERN_START,
        max_year=datetime.date.today().year,
    )
    live_standings["season"] = pd.to_numeric(
        live_standings["season"],
        errors="coerce",
    )

    precomputed_years = set(all_standings["season"].dropna().astype(int))

    live_standings = live_standings[~live_standings["season"].isin(precomputed_years)].copy()

    all_standings = pd.concat(
        [all_standings, live_standings],
        ignore_index=True,
        sort=False,
    )

    available_feature_years = sorted(
        [
            int(year)
            for year in all_standings["season"].dropna().unique()
            if int(year) <= _GAMEINFO_MAX_YEAR
        ],
        reverse=True,
    )

    if not available_feature_years:
        st.error("No overlapping seasons exist between standings data and game-level data.")
        st.stop()

    feat_season = st.selectbox(
        "Season",
        available_feature_years,
        key="feat_season",
    )
    st.caption(
        f"Game-level feature data is available through {_GAMEINFO_MAX_YEAR}. "
        "Live current-season game data is included when available."
    )

    with st.spinner("Building feature matrix…"):
        gi = load_gameinfo(min_year=feat_season, max_year=feat_season)

    if gi.empty:
        st.info("No games in selected season.")
    else:
        ts_yr = all_standings[all_standings["season"].eq(int(feat_season))].set_index("team")
        feat_df = _build_feature_matrix(gi, ts_yr)

        if feat_df.empty:
            st.info("No games with complete standings coverage for this season.")
        else:
            st.markdown(f"**{len(feat_df)} games** in {feat_season} with full feature coverage.")
            display_feat = feat_df.head(50).copy()
            display_feat["date"] = pd.to_datetime(display_feat["date"]).dt.date
            feat_rename = {
                **READABLE_COLS,
                "visitor": "Visitor",
                "home_team": "Home",
                "home_WPct": "Home Win %",
                "vis_WPct": "Visitor Win %",
                "home_RS_G": "Home RS/G",
                "home_RA_G": "Home RA/G",
                "vis_RS_G": "Visitor RS/G",
                "vis_RA_G": "Visitor RA/G",
                "home_PythWPct": "Home Pyth W%",
                "vis_PythWPct": "Visitor Pyth W%",
                "WPct_diff": "Win % Diff",
                "PythWPct_diff": "Pyth W% Diff",
                "RS_advantage": "RS Advantage",
                "RA_advantage": "RA Advantage",
                "daynight": "Day/Night",
                "attendance": "Attendance",
                "temp": "Temperature",
                "windspeed": "Wind Speed",
                "home_win": "Home Win?",
                "total_runs": "Total Runs",
            }
            st.dataframe(
                display_feat.rename(columns=feat_rename),
                width="stretch",
                hide_index=True,
                height=get_dataframe_height(display_feat),
            )

            num_feats = [
                "home_WPct",
                "vis_WPct",
                "WPct_diff",
                "PythWPct_diff",
                "home_RS_G",
                "home_RA_G",
                "vis_RS_G",
                "vis_RA_G",
                "RS_advantage",
                "RA_advantage",
                "home_win",
                "total_runs",
            ]
            readable_feats = [
                "Home Win %",
                "Visitor Win %",
                "Win % Diff",
                "Pyth W% Diff",
                "Home RS/G",
                "Home RA/G",
                "Visitor RS/G",
                "Visitor RA/G",
                "RS Advantage",
                "RA Advantage",
                "Home Win?",
                "Total Runs",
            ]
            corr = feat_df[num_feats].corr()
            corr.index = readable_feats
            corr.columns = readable_feats
            fig = px.imshow(
                corr,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                aspect="auto",
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander("Feature data dictionary"):
                dict_df = pd.DataFrame(
                    {
                        "column": num_feats,
                        "description": [
                            "Home Win %, season standings",
                            "Visitor Win %, season standings",
                            "Home WPct minus visitor WPct",
                            "Home Pythagorean WPct minus visitor",
                            "Home runs scored per game",
                            "Home runs allowed per game",
                            "Visitor runs scored per game",
                            "Visitor runs allowed per game",
                            "Home RS/G minus visitor RS/G",
                            "Visitor RA/G minus home RA/G",
                            "Indicator (1=home team won)",
                            "Total runs scored in game",
                        ],
                    }
                )
                st.dataframe(
                    dict_df.rename(columns=READABLE_COLS), width="stretch", hide_index=True
                )

            st.markdown("#### Home Win % over time")
            gameinfo_range_max = min(max_year, _GAMEINFO_MAX_YEAR)

            if min_year > gameinfo_range_max:
                st.info(
                    f"Home-field and weather trend data are available only through "
                    f"{_GAMEINFO_MAX_YEAR}."
                )
            else:
                if gameinfo_range_max < max_year:
                    st.caption(
                        f"Trend charts use {min_year}–{gameinfo_range_max}; "
                        f"game-level data for {gameinfo_range_max + 1}–{max_year} is not yet available."
                    )

                gi_all = load_gameinfo(min_year, gameinfo_range_max).copy()
                gi_all["home_win"] = (gi_all["wteam"] == gi_all["hometeam"]).astype(int)
                hfa = gi_all.groupby("season")["home_win"].mean().reset_index()
                hfa.columns = ["season", "home_win_pct"]

                fig2 = px.line(
                    hfa,
                    x="season",
                    y="home_win_pct",
                    title="Home Field Advantage — Win % by season",
                    labels={"home_win_pct": "Home Win %", "season": "Season"},
                )
                fig2.add_hline(
                    y=0.5,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="50%",
                )
                st.plotly_chart(fig2, width="stretch")

                if gi_all["temp"].notna().sum() > 100:
                    temp_df = gi_all.dropna(subset=["temp", "total_runs"])
                    temp_df = temp_df[temp_df["temp"] > 0]
                    fig3 = px.scatter(
                        temp_df.sample(min(5000, len(temp_df)), random_state=42),
                        x="temp",
                        y="total_runs",
                        trendline="lowess",
                        title="Temperature vs Total Runs (sample)",
                        labels={"temp": "Temp (°F)", "total_runs": "Total Runs"},
                        opacity=0.4,
                    )
                    st.plotly_chart(fig3, width="stretch")

            with st.expander("Download feature CSV"):
                st.download_button(
                    label="Download features.csv",
                    data=feat_df.to_csv(index=False),
                    file_name=f"features_{feat_season}.csv",
                    mime="text/csv",
                )


# ── ML Models ─────────────────────────────────────────────────────────────────
with tab_models:
    st.subheader("ML Betting Models")
    st.markdown(
        """
        Three XGBoost classifiers trained on **2020+ Retrosheet** game data.
        All models use a **chronological train/test split** (no lookahead).

        | Model | Target | Features |
        |-------|--------|----------|
        | **Moneyline** | P(home team wins) | Team stats, SP stats, weather |
        | **Spread** | P(home covers −1.5) | Same as moneyline |
        | **Over/Under** | P(total > expected) | Same + expected-total offset |
        """
    )

    results = st.session_state["ml_results"]
    if results is None:
        st.info("Pre-trained results not found. Run `scripts/train_models.py` to generate them.")
    else:
        model_labels = {
            "moneyline": "🏆 Moneyline (P home win)",
            "spread": "📏 Spread (P home covers −1.5)",
            "totals": "📈 Over/Under (P went over)",
        }

        st.markdown("### Model Performance (test set)")
        c1, c2, c3 = st.columns(3)
        for col, key in zip([c1, c2, c3], ["moneyline", "spread", "totals"]):
            metrics = results[key]["metrics"]
            col.markdown(f"**{model_labels[key]}**")
            col.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            col.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col.metric("Brier", f"{metrics['brier_score']:.4f}")
            col.metric("Log Loss", f"{metrics['log_loss']:.4f}")

        with st.expander("ℹ️ Metric guide"):
            st.markdown(
                """
                - **ROC-AUC** – 0.5 = random; 1.0 = perfect. 0.60–0.65 is solid for MLB.
                - **Accuracy** – fraction of correct binary picks on the test set.
                - **Brier Score** – mean squared error of predicted probabilities. Lower is better.
                - **Log Loss** – cross-entropy. Lower is better.
                """
            )

        st.markdown("### Feature Importances")
        feature_labels = {
            "WPct_diff": "Win % Diff",
            "PythWPct_diff": "Pythagorean W% Diff",
            "sp_ERA_gap": "SP ERA Gap",
            "home_WPct": "Home Win %",
            "away_WPct": "Away Win %",
            "home_PythWPct": "Home Pyth W%",
            "away_PythWPct": "Away Pyth W%",
            "home_RS_G": "Home RS / G",
            "home_RA_G": "Home RA / G",
            "away_RS_G": "Away RS / G",
            "away_RA_G": "Away RA / G",
            "home_RD_G": "Home RD / G",
            "away_RD_G": "Away RD / G",
            "ERA_diff": "ERA Diff",
            "WHIP_diff": "WHIP Diff",
            "temp": "Temperature (°F)",
            "windspeed": "Wind Speed",
            "is_day": "Day game?",
            "exp_total": "Expected Total Runs",
        }

        def importance_chart(model_key: str, top_n: int = 20) -> None:
            imp = results[model_key]["importances"].head(top_n).copy()
            imp["label"] = imp["feature"].map(lambda value: feature_labels.get(value, value))
            fig = px.bar(
                imp.sort_values("importance"),
                x="importance",
                y="label",
                orientation="h",
                title=f"Top {top_n} features — {model_labels[model_key]}",
                labels={"importance": "Importance", "label": "Feature"},
                color="importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        imp_tab_ml, imp_tab_sp, imp_tab_ou = st.tabs(["Moneyline", "Spread", "Over/Under"])
        with imp_tab_ml:
            importance_chart("moneyline")
        with imp_tab_sp:
            importance_chart("spread")
        with imp_tab_ou:
            importance_chart("totals")

        st.markdown("### Calibration: Predicted vs Actual Win Rate")

        def calibration_chart(model_key: str, prob_col: str, actual_col: str, label: str) -> None:
            test_df = results[model_key]["test_df"][[prob_col, actual_col]].copy()
            test_df["bin"] = pd.cut(test_df[prob_col], bins=10)
            cal = (
                test_df.groupby("bin", observed=False)
                .agg(
                    mean_pred=(prob_col, "mean"),
                    actual_rate=(actual_col, "mean"),
                    count=(prob_col, "count"),
                )
                .reset_index()
                .dropna()
            )
            fig = px.scatter(
                cal,
                x="mean_pred",
                y="actual_rate",
                size="count",
                title=f"{model_labels[model_key]} — calibration",
                labels={"mean_pred": f"Mean predicted {label}", "actual_rate": "Actual rate"},
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="gray"))
            st.plotly_chart(fig, width="stretch")

        cal_tab_ml, cal_tab_sp, cal_tab_ou = st.tabs(["Moneyline", "Spread", "Over/Under"])
        with cal_tab_ml:
            calibration_chart("moneyline", "pred_prob", "home_win", "home win probability")
        with cal_tab_sp:
            calibration_chart("spread", "pred_prob", "home_cover", "P(cover −1.5)")
        with cal_tab_ou:
            calibration_chart("totals", "pred_prob_over", "went_over", "P(over)")

        st.markdown("### Backtest Sample — Recent Test-Set Games")
        bt_model = st.selectbox(
            "Model",
            ["moneyline", "spread", "totals"],
            format_func=lambda value: model_labels[value],
            key="bt_model_sel",
        )
        n_show = st.slider("Games to display", 25, 200, 50, key="bt_n_show")
        bt_df = results[bt_model]["test_df"].tail(n_show).copy()
        try:
            bt_df["date"] = pd.to_datetime(bt_df["date"]).dt.date
        except (ValueError, TypeError):
            pass

        if bt_model == "moneyline" and "pred_win" in bt_df.columns:
            bt_df["pred_win"] = bt_df["pred_win"].map({1: "Home", 0: "Away"})
        if bt_model == "spread" and "pred_cover" in bt_df.columns:
            bt_df["pred_cover"] = bt_df["pred_cover"].map({1: "Home −1.5", 0: "Away +1.5"})
        if "correct" in bt_df.columns:
            bt_df["correct"] = bt_df["correct"].astype(bool).map({True: "✔", False: ""})
        for col in ("home_win", "home_cover", "went_over"):
            if col in bt_df.columns:
                bt_df[col] = bt_df[col].astype(bool).map({True: "✔", False: ""})

        if bt_model == "moneyline":
            display_cols = {
                "date": "Date",
                "hometeam": "Home",
                "visteam": "Away",
                "hruns": "H Runs",
                "vruns": "V Runs",
                "home_win": "Actually Won?",
                "pred_prob": "Pred. Prob",
                "pred_win": "Model Pick",
                "correct": "Correct?",
            }
        elif bt_model == "spread":
            display_cols = {
                "date": "Date",
                "hometeam": "Home",
                "visteam": "Away",
                "home_margin": "Margin",
                "home_cover": "Actually Covered?",
                "pred_prob": "P(cover −1.5)",
                "pred_cover": "Model Pick",
                "correct": "Correct?",
            }
        else:
            display_cols = {
                "date": "Date",
                "hometeam": "Home",
                "visteam": "Away",
                "total_runs": "Total Runs",
                "exp_total": "Exp Total",
                "went_over": "Went Over?",
                "pred_prob_over": "P(over)",
                "pick_side": "Pick",
                "correct": "Correct?",
            }

        existing = [col for col in display_cols if col in bt_df.columns]
        display_bt = bt_df[existing].reset_index(drop=True).rename(columns=display_cols)
        st.dataframe(
            display_bt,
            width="stretch",
            hide_index=True,
            height=get_dataframe_height(display_bt),
        )

        with st.expander("Download backtest CSVs"):
            c_dl1, c_dl2, c_dl3 = st.columns(3)
            for col, key, label in zip(
                [c_dl1, c_dl2, c_dl3],
                ["moneyline", "spread", "totals"],
                ["moneyline", "spread", "totals"],
            ):
                col.download_button(
                    label=f"Download {label}.csv",
                    data=results[key]["test_df"].to_csv(index=False),
                    file_name=f"backtest_{label}.csv",
                    mime="text/csv",
                )


# ── Model Evaluation ──────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Model Evaluation")
    st.markdown(
        "Walk-forward backtests, calibration analyses, and profitability reports. "
        "Results are pre-loaded from parquet files generated by the evaluation scripts."
    )

    if st.session_state["eval_backtests"]:
        leaderboard = [bt.summary() for bt in st.session_state["eval_backtests"].values()]
        lb_df = pd.DataFrame(leaderboard).sort_values("roi", ascending=False)
        if "period" in lb_df.columns:
            lb_df["period"] = lb_df["period"].astype(str).str.strip().str.slice(0, 10)

        st.markdown("### Backtest Leaderboard")
        st.dataframe(
            lb_df.rename(
                columns={
                    "model": "Model",
                    "pick_type": "Pick Type",
                    "period": "Period",
                    "total_bets": "Bets",
                    "wins": "Wins",
                    "losses": "Losses",
                    "pushes": "Pushes",
                    "win_rate": "Win Rate",
                    "total_units": "Units",
                    "max_drawdown": "Max Drawdown",
                    "roi": "ROI",
                }
            ),
            hide_index=True,
            width="stretch",
            height=get_dataframe_height(lb_df),
        )

        st.markdown("### Calibration Charts")
        for name, backtest in st.session_state["eval_backtests"].items():
            arr_true = [1 if bet.result == "win" else 0 for bet in backtest.bets]
            arr_prob = [bet.predicted_prob for bet in backtest.bets]
            cal_data = calibration_plot_data(np.array(arr_true), np.array(arr_prob))
            fig = px.scatter(
                x=cal_data["mean_predicted"],
                y=cal_data["fraction_positive"],
                size=[1] * len(cal_data["mean_predicted"]),
                title=f"{name.capitalize()} Calibration",
                labels={"x": "Mean pred", "y": "Actual rate"},
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="gray"))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "No evaluation data yet. Run `scripts/run_evaluation.py` or `scripts/train_models.py`."
        )


# ── Savant Research ───────────────────────────────────────────────────────────
with tab_savant:
    mc_ranking = _pre.get("mc_ranking")
    mc_trials = _pre.get("mc_trials")
    savant_metrics = _pre.get("savant_metrics")

    st.subheader("Savant Feature Research — Monte Carlo Selection")
    st.markdown(
        "Results of a Monte Carlo search over Baseball Savant advanced metrics. "
        "1,000 random trials were run, sampling random subsets of Statcast columns, "
        "training XGBoost with TimeSeriesSplit CV across three bet targets. "
        "Features are ranked by how often they appeared in the **top 10% of trials by ROC-AUC**."
    )

    if mc_ranking is None or mc_ranking.empty:
        st.info(
            "Monte Carlo results not found. "
            "Run `python scripts/monte_carlo_features.py --trials 1000` to generate them."
        )
    else:
        baseline_auc = {"moneyline": 0.6253, "spread": 0.6304, "totals": 0.6157}
        n_valid = len(mc_trials) if mc_trials is not None else None
        top_cutoff = mc_trials["mean_auc"].quantile(0.90) if mc_trials is not None else None

        col_m, col_s, col_t, col_v = st.columns(4)
        col_m.metric("Baseline Moneyline AUC", f"{baseline_auc['moneyline']:.4f}")
        col_s.metric("Baseline Spread AUC", f"{baseline_auc['spread']:.4f}")
        col_t.metric("Baseline Totals AUC", f"{baseline_auc['totals']:.4f}")
        col_v.metric("Valid Trials", f"{n_valid:,}" if n_valid is not None else "—")

        if savant_metrics is not None and not savant_metrics.empty:
            st.markdown("---")
            st.markdown("#### Savant-Enriched Model Performance")
            perf_cols = st.columns(3)
            for i, model_name in enumerate(["moneyline", "spread", "totals"]):
                row = savant_metrics[savant_metrics["model"] == model_name]
                if row.empty:
                    continue
                auc = float(row.iloc[0]["roc_auc"])
                perf_cols[i].metric(
                    label=f"{model_name.capitalize()} AUC (Savant)",
                    value=f"{auc:.4f}",
                    delta=f"{auc - baseline_auc[model_name]:+.4f} vs baseline",
                )

        if mc_trials is not None:
            st.markdown("---")
            auc_plot_cols = [
                col for col in mc_trials.columns if col.endswith("_auc") and col != "mean_auc"
            ]
            auc_long = mc_trials[auc_plot_cols + ["mean_auc"]].melt(
                var_name="target", value_name="auc"
            )
            auc_long["target"] = auc_long["target"].str.replace("_auc", "").str.capitalize()
            fig_dist = px.box(
                auc_long[auc_long["target"] != "Mean"],
                x="target",
                y="auc",
                color="target",
                points=False,
                title="Trial AUC by Bet Target (1,000 trials)",
                labels={"target": "Bet Target", "auc": "ROC-AUC"},
            )
            for target_name, baseline in [
                ("Moneyline", baseline_auc["moneyline"]),
                ("Spread", baseline_auc["spread"]),
                ("Totals", baseline_auc["totals"]),
            ]:
                fig_dist.add_hline(
                    y=baseline,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"{target_name} baseline",
                    annotation_position="right",
                )
            if top_cutoff is not None:
                fig_dist.add_hline(
                    y=top_cutoff,
                    line_dash="dash",
                    line_color="#7c3aed",
                    annotation_text="Top 10% cutoff",
                    annotation_position="left",
                )
            st.plotly_chart(fig_dist, width="stretch")

        bat_ranks = mc_ranking[mc_ranking["type"] == "batter"].head(20).copy()
        pit_ranks = mc_ranking[mc_ranking["type"] == "pitcher"].head(20).copy()
        bat_ranks["appearance_pct"] = bat_ranks["appearance_rate"] * 100
        pit_ranks["appearance_pct"] = pit_ranks["appearance_rate"] * 100

        st.markdown("#### Top Batter Features")
        fig_bat = px.bar(
            bat_ranks.sort_values("appearance_pct"),
            x="appearance_pct",
            y="feature",
            orientation="h",
            title="Batter Feature Selection Frequency",
            labels={
                "appearance_pct": "Appearance Rate in Top Trials (%)",
                "feature": "Savant Column",
            },
            color="appearance_pct",
            color_continuous_scale="Blues",
        )
        fig_bat.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_bat, width="stretch")

        st.markdown("#### Top Pitcher Features")
        fig_pit = px.bar(
            pit_ranks.sort_values("appearance_pct"),
            x="appearance_pct",
            y="feature",
            orientation="h",
            title="Pitcher Feature Selection Frequency",
            labels={
                "appearance_pct": "Appearance Rate in Top Trials (%)",
                "feature": "Savant Column",
            },
            color="appearance_pct",
            color_continuous_scale="Reds",
        )
        fig_pit.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_pit, width="stretch")

        with st.expander("Full feature ranking table"):
            display_rank = mc_ranking.copy()
            display_rank["appearance_rate"] = (display_rank["appearance_rate"] * 100).round(1)
            st.dataframe(
                display_rank.rename(
                    columns={
                        "feature": "Savant Column",
                        "type": "Type",
                        "top_trial_appearances": "Appearances in Top Trials",
                        "appearance_rate": "Rate (%)",
                    }
                ),
                hide_index=True,
                width="stretch",
                height=get_dataframe_height(display_rank),
            )
