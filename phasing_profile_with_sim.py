"""
Fits historical phasing data according to filters in dataset.

Plots phasing profiles.
Fits beta distributions.
Analyzes derived fit parameters.
Creates predicted fit parameters for MC sim.
Simulates fits and plots on chart.
Created on Wed Dec 31 2025

@author: stephen.colodner
"""

import pandas as pd
import numpy as np
from scipy.stats import beta
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

# --- Globals/Constants ---
MAX_ALPHA_PARAM = 10.0
MAX_BETA_PARAM = 10.0
N_SIMULATIONS = 200
Y_AXIS_UPPER_LIMIT = 0.5
CSV_FILE_PATH = "phasing_data.csv"
REQUIRED_COLUMNS = [
    "Year",
    "Project",
    "Include",
    "PARSID",
    "Cum_Time_Pct",
    "Inc_Expend_Pct",
]

# --- Helper Functions ---


def _beta_pdf_scaled(x, alpha, beta_param, scale_factor):
    """
    Scaled Beta PDF function for curve fitting.
    Clips x to avoid issues at zero or one and ensures positive alpha/beta.
    """
    x_clipped = np.clip(x, 0.0001, 0.9999)
    alpha = max(0.01, alpha)
    beta_param = max(0.01, beta_param)
    return beta.pdf(x_clipped, a=alpha, b=beta_param) * scale_factor


def load_data(file_path, required_cols):
    """
    Loads data from a CSV file, assumes it exists and has required columns for EDA.
    """
    df = pd.read_csv(file_path)
    df["Include"] = df["Include"].astype(bool)
    return df


def augment_data_with_zero_point(df):
    """
    Adds (0,0) point to each project's data, assuming Cum_Time_Pct=1 already exists.
    """
    augmented_list = []
    for project_name, group in df.groupby("Project"):
        current_group_data = group.copy().sort_values("Cum_Time_Pct")

        first_row_orig = current_group_data.iloc[0]
        new_first_point = first_row_orig.copy()
        new_first_point["Cum_Time_Pct"] = 0
        new_first_point["Cum_Expend_Pct"] = 0
        new_first_point["Inc_Expend_Pct"] = 0
        new_first_point["Year"] = np.nan

        augmented_list.append(
            pd.concat(
                [pd.DataFrame([new_first_point]), current_group_data]
            ).sort_values("Cum_Time_Pct")
        )
    return pd.concat(augmented_list).reset_index(drop=True)


def fit_beta_distributions(df_augmented, max_alpha, max_beta):
    """
    Fits beta distributions to the incremental expenditure for each included project.
    Assumes sufficient data points for fitting.
    """
    program_fits_list = []
    included_df = df_augmented[df_augmented["Include"] == True].copy()

    for project_name, group in included_df.groupby("Project"):
        fit_data = {"Project": project_name, "data": group}
        current_group_data = group.copy()

        x_data = current_group_data["Cum_Time_Pct"].values
        y_data = current_group_data["Inc_Expend_Pct"].values

        # Initial parameter estimation (simplified assumption of non-zero max_incr_exp)
        mean_time = np.mean(x_data[y_data > 0]) if np.any(y_data > 0) else 0.5
        initial_alpha_beta = 2
        max_incr_exp = current_group_data["Inc_Expend_Pct"].max()
        if max_incr_exp == 0:
            max_incr_exp = (
                0.01  # Prevent division by zero if all Inc_Expend_Pct are 0
            )

        initial_scale_factor = 1.0
        beta_val_at_mean = beta.pdf(
            np.clip(mean_time, 0.0001, 0.9999),
            initial_alpha_beta,
            initial_alpha_beta,
        )
        if beta_val_at_mean != 0:
            initial_scale_factor = max_incr_exp / beta_val_at_mean
        if np.isnan(initial_scale_factor) or np.isinf(initial_scale_factor):
            initial_scale_factor = 1.0  # Fallback

        initial_params = [
            initial_alpha_beta,
            initial_alpha_beta,
            initial_scale_factor,
        ]
        bounds = ([0.01, 0.01, 0], [max_alpha, max_beta, 100])

        try:
            params, _ = curve_fit(
                _beta_pdf_scaled,
                x_data,
                y_data,
                p0=initial_params,
                bounds=bounds,
                method="trf",
            )
            alpha_fit, beta_fit, scale_factor_fit = params

            fit_data["beta_params"] = {
                "alpha": max(0.01, alpha_fit),
                "beta": max(0.01, beta_fit),
                "scale_factor": scale_factor_fit,
            }
            seq_time = np.linspace(0, 1, 100)
            fit_data["fitted_curve"] = pd.DataFrame(
                {
                    "Cum_Time_Pct": seq_time,
                    "Fitted_Inc_Expend_Pct": _beta_pdf_scaled(
                        seq_time, alpha_fit, beta_fit, scale_factor_fit
                    ),
                }
            )
            fit_data["beta_label"] = f"α={alpha_fit:.2f}\nβ={beta_fit:.2f}"
        except RuntimeError as e:
            # For EDA, just report failure, no need to stop execution
            print(f"Beta fit failed for Project: {project_name} - {e}")
            fit_data["beta_params"] = None
            fit_data["fitted_curve"] = None
            fit_data["beta_label"] = "Beta Fit Failed"
        program_fits_list.append(fit_data)

    program_fits_df = pd.DataFrame(program_fits_list)
    # No empty checks required, assuming well-behaved means/medians
    program_fits_df = program_fits_df.join(
        program_fits_df["beta_params"].apply(pd.Series)
    )
    program_fits_df["beta_label"] = program_fits_df.apply(
        lambda row: (
            row["beta_label"]
            if row["beta_params"] is not None
            else "Beta Fit Failed"
        ),
        axis=1,
    )
    return program_fits_df


def calculate_predicted_profiles(fitted_params_df):
    """
    Calculates mean and median predicted beta profiles from historical fits.
    Assumes there are always more than 2 programs included and well-behaved parameters.
    """
    included_beta_params = fitted_params_df.dropna(
        subset=["alpha", "beta", "scale_factor"]
    ).copy()

    mean_alpha = included_beta_params["alpha"].mean()
    mean_beta = included_beta_params["beta"].mean()
    mean_scale_factor = included_beta_params["scale_factor"].mean()

    median_alpha = included_beta_params["alpha"].median()
    median_beta = included_beta_params["beta"].median()
    median_scale_factor = included_beta_params["scale_factor"].median()

    common_time_points_pred = np.linspace(0, 1, 100)

    predicted_incremental_mean = _beta_pdf_scaled(
        common_time_points_pred, mean_alpha, mean_beta, mean_scale_factor
    )
    beta_label_mean = f"Mean α={mean_alpha:.2f}\nMean β={mean_beta:.2f}"
    predicted_program_plot_data_mean = pd.DataFrame(
        {
            "Project": "Predicted Profile",
            "Profile_Type": "Mean",
            "Cum_Time_Pct": common_time_points_pred,
            "Inc_Expend_Pct": predicted_incremental_mean,
            "Fitted_Inc_Expend_Pct": predicted_incremental_mean,
            "beta_label": beta_label_mean,
        }
    )

    predicted_incremental_median = _beta_pdf_scaled(
        common_time_points_pred, median_alpha, median_beta, median_scale_factor
    )
    beta_label_median = f"Median α={
        median_alpha:.2f}\nMedian β={median_beta:.2f}"
    predicted_program_plot_data_median = pd.DataFrame(
        {
            "Project": "Predicted Profile",
            "Profile_Type": "Median",
            "Cum_Time_Pct": common_time_points_pred,
            "Inc_Expend_Pct": predicted_incremental_median,
            "Fitted_Inc_Expend_Pct": predicted_incremental_median,
            "beta_label": beta_label_median,
        }
    )
    predicted_program_plot_data_mean_median = pd.concat(
        [predicted_program_plot_data_mean, predicted_program_plot_data_median]
    )

    return predicted_program_plot_data_mean_median


def prepare_plotting_data(
    df_augmented, program_fits_df, predicted_program_plot_data_mean_median
):
    """
    Prepares dataframes for plotting historical and predicted phasing curves and bars.
    Assumes datasets created will not be empty.
    Historical data will be used for bars (plot_data_original).
    The 'Predicted Profile' will have an entry in plot_data_original for its subplot title,
    but with zero expenditure to prevent bars.
    """
    plot_data_original_historical = df_augmented[
        df_augmented["Include"] == True
    ].copy()
    plot_data_original_historical["period_start"] = (
        plot_data_original_historical.groupby("Project")["Cum_Time_Pct"]
        .shift(1)
        .fillna(0)
    )
    plot_data_original_historical["Profile_Type"] = "Historical"

    # Create 'Predicted Profile' to plot_data_original, but with 0 Inc_Expend_Pct
    # This ensures a subplot is created for it, but no bars are drawn

    predicted_dummy_for_subplot = pd.DataFrame(
        {
            "Project": ["Predicted Profile"],
            # Needs a Cum_Time_Pct value to create a row
            "Cum_Time_Pct": [1.0],
            "Inc_Expend_Pct": [0.0],  # Key: set to 0 to prevent bars
            "period_start": [0.0],
            "Profile_Type": ["Predicted"],
            # Fill other required columns with dummy values or np.nan if they exist
            "Year": [np.nan],
            "PARSID": [np.nan],
            "Include": [False],
            "Cum_Expend_Pct": [np.nan],
        }
    )

    plot_data_original = pd.concat(
        [plot_data_original_historical, predicted_dummy_for_subplot],
        ignore_index=True,
    )

    plot_data_fitted_curves_list = []

    for _, row in program_fits_df[
        program_fits_df["fitted_curve"].notna()
    ].iterrows():
        temp_df = row["fitted_curve"].copy()
        temp_df["Project"] = row["Project"]
        plot_data_fitted_curves_list.append(temp_df)

    plot_data_fitted_curves = pd.concat(
        plot_data_fitted_curves_list
    ).reset_index(drop=True)
    plot_data_fitted_curves["Profile_Type"] = "Historical"

    # Assuming predicted_program_plot_data_mean_median is not empty
    predicted_fitted_df_all = predicted_program_plot_data_mean_median[
        ["Project", "Profile_Type", "Cum_Time_Pct", "Fitted_Inc_Expend_Pct"]
    ].copy()
    plot_data_fitted_curves = pd.concat(
        [plot_data_fitted_curves, predicted_fitted_df_all], ignore_index=True
    )

    all_beta_labels_list = []
    # Assuming predicted_program_plot_data_mean_median is not empty
    predicted_labels = predicted_program_plot_data_mean_median[
        ["Project", "beta_label", "Profile_Type"]
    ].copy()
    predicted_labels["beta_label"] = (
        predicted_labels["Profile_Type"] + " " + predicted_labels["beta_label"]
    )
    all_beta_labels_list.append(predicted_labels)

    historical_labels = (
        program_fits_df[["Project", "beta_label"]]
        .dropna(subset=["beta_label"])
        .copy()
    )
    historical_labels["Profile_Type"] = "Historical"
    all_beta_labels_list.append(historical_labels)
    all_beta_labels = (
        pd.concat(all_beta_labels_list)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    historical_beta_params = program_fits_df.dropna(
        subset=["alpha", "beta"]
    ).copy()
    historical_beta_params["skew_metric"] = (
        historical_beta_params["alpha"] - historical_beta_params["beta"]
    )

    # Simplify extraction of mean/median skew metric, assuming labels are always there
    mean_label_data = predicted_program_plot_data_mean_median[
        predicted_program_plot_data_mean_median["Profile_Type"] == "Mean"
    ]
    label_text = mean_label_data["beta_label"].iloc[0]
    match_alpha = re.search(r"α=([0-9]+\.[0-9]+)", label_text)
    match_beta = re.search(r"β=([0-9]+\.[0-9]+)", label_text)
    predicted_skew_metric_mean = float(match_alpha.group(1)) - float(
        match_beta.group(1)
    )

    ordered_projects_df = (
        pd.concat(
            [
                historical_beta_params[["Project", "skew_metric"]],
                pd.DataFrame(
                    [
                        {
                            "Project": "Predicted Profile",
                            "skew_metric": predicted_skew_metric_mean,
                        }
                    ]
                ),
            ]
        )
        .sort_values("skew_metric")
        .reset_index(drop=True)
    )
    project_levels_ordered = ordered_projects_df["Project"].tolist()

    plot_data_original["Project"] = pd.Categorical(
        plot_data_original["Project"],
        categories=project_levels_ordered,
        ordered=True,
    )
    plot_data_fitted_curves["Project"] = pd.Categorical(
        plot_data_fitted_curves["Project"],
        categories=project_levels_ordered,
        ordered=True,
    )
    all_beta_labels["Project"] = pd.Categorical(
        all_beta_labels["Project"],
        categories=project_levels_ordered,
        ordered=True,
    )

    return (
        plot_data_original,
        plot_data_fitted_curves,
        all_beta_labels,
        project_levels_ordered,
    )


def plot_phasing_profiles(
    plot_data_original,
    plot_data_fitted_curves,
    all_beta_labels,
    project_levels_ordered,
    y_axis_limit,
):
    """
    Plots the phasing profiles for historical and predicted projects.
    Assumes datasets will not be empty before plotting.
    """
    plt.style.use("seaborn-v0_8-white")
    n_rows = 3
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(8.5, 5.2),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    fill_colors = {"Predicted": "red", "Historical": "gray"}
    line_colors = {
        "Predicted_Fit_Mean": "darkred",
        "Predicted_Fit_Median": "darkgreen",
        "Historical_Fit": "darkblue",
    }

    # Proxy artists for legend
    proxy_fill_predicted = plt.Rectangle(
        (0, 0), 1, 1, fc=fill_colors["Predicted"], alpha=0.9
    )
    proxy_fill_historical = plt.Rectangle(
        (0, 0), 1, 1, fc=fill_colors["Historical"], alpha=0.9
    )
    proxy_line_predicted_mean = plt.Line2D(
        [0],
        [0],
        linestyle="solid",
        color=line_colors["Predicted_Fit_Mean"],
        linewidth=1.2,
    )
    proxy_line_predicted_median = plt.Line2D(
        [0],
        [0],
        linestyle="solid",
        color=line_colors["Predicted_Fit_Median"],
        linewidth=1.2,
    )
    proxy_line_historical = plt.Line2D(
        [0],
        [0],
        linestyle="dashed",
        color=line_colors["Historical_Fit"],
        linewidth=1.2,
    )

    for i, project in enumerate(project_levels_ordered):
        ax = axes[i]

        original_data_project = plot_data_original[
            plot_data_original["Project"] == project
        ]
        fitted_data_project = plot_data_fitted_curves[
            plot_data_fitted_curves["Project"] == project
        ]
        beta_labels_project = all_beta_labels[
            all_beta_labels["Project"] == project
        ]

        # No empty checks for these dataframes as per assumption
        for _, row in original_data_project.iterrows():
            ax.fill_between(
                [row["period_start"], row["Cum_Time_Pct"]],
                0,
                row["Inc_Expend_Pct"],
                color=(
                    fill_colors["Predicted"]
                    if project == "Predicted Profile"
                    else fill_colors["Historical"]
                ),
                alpha=0.9,
            )

        for profile_type in fitted_data_project["Profile_Type"].unique():
            subset = fitted_data_project[
                fitted_data_project["Profile_Type"] == profile_type
            ]
            if project == "Predicted Profile":
                line_key = f"Predicted_Fit_{profile_type}"
                ax.plot(
                    subset["Cum_Time_Pct"],
                    subset["Fitted_Inc_Expend_Pct"],
                    color=line_colors.get(line_key, "black"),
                    linestyle="solid",
                    linewidth=1.2,
                )
            else:
                ax.plot(
                    subset["Cum_Time_Pct"],
                    subset["Fitted_Inc_Expend_Pct"],
                    color=line_colors["Historical_Fit"],
                    linestyle="dashed",
                    linewidth=1,
                )

        # Assuming beta_labels_project is not empty
        if project == "Predicted Profile":
            mean_label_series = beta_labels_project[
                beta_labels_project["Profile_Type"] == "Mean"
            ]
            median_label_series = beta_labels_project[
                beta_labels_project["Profile_Type"] == "Median"
            ]

            # Mean label in darkred
            if (
                not mean_label_series.empty
                and mean_label_series["beta_label"].iloc[0] != "Mean N/A"
            ):
                ax.text(
                    0.05,
                    0.9,
                    mean_label_series["beta_label"]
                    .iloc[0]
                    .replace("Mean Mean ", "Mean "),
                    transform=ax.transAxes,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=7,
                    fontweight="normal",
                    color=line_colors["Predicted_Fit_Mean"],
                )

            # Median label in darkgreen, positioned below the mean label
            if (
                not median_label_series.empty
                and median_label_series["beta_label"].iloc[0] != "Median N/A"
            ):
                ax.text(
                    0.05,
                    0.7,
                    median_label_series["beta_label"]
                    .iloc[0]
                    .replace("Median Median ", "Median "),
                    transform=ax.transAxes,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=7,
                    fontweight="normal",
                    color=line_colors["Predicted_Fit_Median"],
                )
        else:  # Historical programs
            historical_label_data = beta_labels_project[
                beta_labels_project["Profile_Type"] == "Historical"
            ]
            ax.text(
                0.05,
                0.9,
                historical_label_data["beta_label"].iloc[0],
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
                fontweight="normal",
                color=line_colors["Historical_Fit"],
            )
            ax.text(
                0.05,
                0.9,
                historical_label_data["beta_label"].iloc[0],
                transform=ax.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
                fontweight="normal",
                color="darkblue",
            )

        ax.set_title(project, fontsize=6, fontweight="bold", wrap = True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_axis_limit)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, y_axis_limit + 0.01, 0.1))
        ax.grid(True, linestyle="--", alpha=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Historical & Predicted Program Spending Profiles (Incremental Expenditure with Beta Fit)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5, 0.07, "Normalized Duration", ha="center", va="center", fontsize=12
    )
    fig.text(
        0.04,
        0.5,
        "Incremental Expenditure",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    handles_legend = [
        proxy_fill_historical,
        proxy_line_predicted_mean,
        proxy_line_predicted_median,
        proxy_line_historical,
    ]
    labels_legend = [
        "Historical Program (Original)",
        "Predicted Profile (Mean Beta Fit)",
        "Predicted Profile (Median Beta Fit)",
        "Historical Program (Beta Fit)",
    ]
    fig.legend(
        handles_legend,
        labels_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        title="",
        fontsize=10,
        title_fontsize=12,
        frameon=False,
    )
    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.98])
    plt.show()


def prepare_correlation_data(program_fits_df, original_df):
    """
    Prepares a DataFrame containing max_year, alpha, and β for correlation analysis.

    Args:
        program_fits_df (pd.DataFrame): DataFrame with fitted alpha/β parameters for each project.
        original_df (pd.DataFrame): The original loaded DataFrame (df_augmented or df).

    Returns:
        pd.DataFrame: A DataFrame suitable for correlation.
    """
    # Filter for successfully fitted historical programs
    historical_fitted_params = program_fits_df.dropna(
        subset=["alpha", "beta"]
    ).copy()

    # Get the max 'Year' for each project from the original data
    max_year_per_project = (
        original_df.groupby("Project")["Year"].max().reset_index()
    )
    max_year_per_project.rename(columns={"Year": "Max_Year"}, inplace=True)

    # Merge the max_year with the fitted parameters
    correlation_data = pd.merge(
        historical_fitted_params[["Project", "alpha", "beta"]],
        max_year_per_project,
        on="Project",
        how="inner",
    )

    # Ensure Max_Year is numeric and handle potential NaNs if any projects don't have a year
    correlation_data["Max_Year"] = pd.to_numeric(
        correlation_data["Max_Year"], errors="coerce"
    )
    correlation_data.dropna(subset=["Max_Year"], inplace=True)

    return correlation_data[["Max_Year", "alpha", "beta"]]


def analyze_duration_alpha_beta_correlation(correlation_data):
    """
    Performs and visualizes correlation analysis between Max_Year, alpha, and β.

    Args:
        correlation_data (pd.DataFrame): DataFrame with Max_Year, alpha, and β.
    """
    if correlation_data.empty:
        print(
            "Not enough data to perform duration-alpha-β correlation analysis."
        )
        return

    print("\n--- Correlation Analysis: Max_Year, α, β ---")

    # Pearson Correlation
    pearson_corr = correlation_data.corr(method="pearson")
    print("\nPearson Correlation Matrix:")
    print(pearson_corr)

    # Spearman Correlation
    spearman_corr = correlation_data.corr(method="spearman")
    print("\nSpearman Correlation Matrix (Rank Correlation):")
    print(spearman_corr)

    # Visualize with a heatmap
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2))

    sns.heatmap(
        pearson_corr,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title(
        "Pearson Correlation: Duration, α, β",
        fontsize=12,
        fontweight="bold",
    )

    sns.heatmap(
        spearman_corr,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        linewidths=0.5,
        ax=axes[1],
    )
    axes[1].set_title(
        "Spearman Correlation: Duration, α, β",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()

    # Optional: Pair plot for scatter visualization
    # This helps visualize the pairwise relationships and check for linearity/monotonicity
    g = sns.pairplot(correlation_data)
    g.fig.suptitle(
        "Pair Plot: Duration, α, β",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


def analyze_beta_parameters(
    program_fits_df, predicted_program_plot_data_mean_median
):
    """
    Analyzes fitted beta parameters and derived statistics for scatter plots.
    Assumes sufficient data and well-behaved parameters.
    """
    fitted_historical_params_df = program_fits_df.dropna(
        subset=["alpha", "beta", "scale_factor"]
    ).copy()
    # Ensure scale_factor is included here
    fitted_historical_params_df = fitted_historical_params_df[
        ["Project", "alpha", "beta", "scale_factor"]
    ]
    fitted_historical_params_df = fitted_historical_params_df[
        (fitted_historical_params_df["alpha"] > 0.01)
        & (fitted_historical_params_df["beta"] > 0.01)
    ].copy()
    fitted_historical_params_df["log_alpha"] = np.log(
        fitted_historical_params_df["alpha"]
    )
    fitted_historical_params_df["log_beta"] = np.log(
        fitted_historical_params_df["beta"]
    )

    predicted_beta_params_for_scatter_list = []
    # Assuming predicted_program_plot_data_mean_median is not empty and labels are there
    for _, row in predicted_program_plot_data_mean_median.iterrows():
        label_text = row["beta_label"]
        match_alpha = re.search(r"α=([0-9]+\.[0-9]+)", label_text)
        match_beta = re.search(r"β=([0-9]+\.[0-9]+)", label_text)
        alpha_val = float(match_alpha.group(1))
        beta_val = float(match_beta.group(1))
        # Note: predicted_beta_params_for_scatter does not necessarily have 'scale_factor'
        # if it's not explicitly extracted from the label.
        # For the purpose of MC simulation, we only need 'scale_factor' from historical data.
        predicted_beta_params_for_scatter_list.append(
            {
                "Project": f"Predicted {row['Profile_Type']}",
                "alpha": alpha_val,
                "beta": beta_val,
            }
        )
    predicted_beta_params_for_scatter = pd.DataFrame(
        predicted_beta_params_for_scatter_list
    )

    # CONCATENATE ALL COLUMNS NEEDED LATER, including 'scale_factor' from historical data
    # We need to correctly align these.
    # The 'scale_factor' from historical_log_params will be used for MC.
    # For 'all_params_for_scatter' which is used for plotting alpha/beta,
    # we don't *strictly* need scale_factor unless plotting it.
    # But to get 'historical_log_params' later, we need it in `fitted_historical_params_df` used to make `all_params_for_scatter`.

    # The simplest way to handle this for EDA is to make sure `fitted_historical_params_df`
    # has all the columns needed for `historical_log_params` and then append/merge carefully.

    # Let's rebuild all_params_for_scatter to ensure flexibility:
    all_params_for_scatter = fitted_historical_params_df.copy()
    # Add a column to distinguish historical from predicted for concatenation
    all_params_for_scatter["is_predicted"] = False

    # Now add predicted parameters. They don't have 'scale_factor' from fit,
    # so we will fill it with NaN or a placeholder, but it won't be used for MC simulation.
    predicted_beta_params_for_scatter["scale_factor"] = (
        np.nan
    )  # Add scale_factor column, will be NaN for predicted
    predicted_beta_params_for_scatter["is_predicted"] = True
    predicted_beta_params_for_scatter["log_alpha"] = np.log(
        predicted_beta_params_for_scatter["alpha"]
    )
    predicted_beta_params_for_scatter["log_beta"] = np.log(
        predicted_beta_params_for_scatter["beta"]
    )

    all_params_for_scatter = pd.concat(
        [all_params_for_scatter, predicted_beta_params_for_scatter],
        ignore_index=True,
    ).reset_index(drop=True)

    all_params_for_scatter = all_params_for_scatter.dropna(
        subset=["alpha", "beta"]
    ).copy()
    all_params_for_scatter = all_params_for_scatter[
        (all_params_for_scatter["alpha"] > 0.01)
        & (all_params_for_scatter["beta"] > 0.01)
    ].copy()

    def get_beta_stats(row):
        # Assuming alpha and beta are always > 0 for this EDA context
        mean, var, skew, _ = beta.stats(
            row["alpha"], row["beta"], moments="mvsk"
        )
        return pd.Series(
            {"beta_mean": mean, "beta_variance": var, "beta_skewness": skew}
        )

    all_params_for_scatter[["beta_mean", "beta_variance", "beta_skewness"]] = (
        all_params_for_scatter.apply(get_beta_stats, axis=1)
    )

    print("\nCorrelation Analysis for Beta Parameters:")
    correlation_data = all_params_for_scatter[
        [
            "alpha",
            "beta",
            "log_alpha",
            "log_beta",
            "beta_mean",
            "beta_variance",
            "beta_skewness",
        ]
    ].dropna()
    # Assuming enough data points for correlation
    print(correlation_data.corr())
    print("Correlation analysis complete.")
    return all_params_for_scatter


def plot_alpha_beta_histograms(all_params_for_scatter):
    """
    Plots histograms of the fitted α, β, log_α, and log_β parameters
    for historical projects.
    The KDE will be scaled to density to match the histogram's normalization.
    """
    # Filter for historical projects only
    historical_params = all_params_for_scatter[
        ~all_params_for_scatter["is_predicted"]
    ].copy()

    if historical_params.empty:
        print("No historical parameters available to plot histograms.")
        return

    # Create a 2x2 grid of subplots
    # Adjusted figsize for 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2))
    fig.suptitle(
        "Distribution of Fitted α, β, Log(α), and Log(β) Parameters (Historical Projects)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # --- Row 1: Alpha and Beta Histograms ---
    # Histogram for Alpha
    sns.histplot(
        historical_params["alpha"],
        kde=True,
        stat="density",
        ax=axes[0, 0],
        color="skyblue",
        bins=8,
    )  # 8 bins
    axes[0, 0].set_title(
        "Distribution of Fitted α Parameter",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("α (Shape1)", fontsize=10)
    axes[0, 0].set_ylabel("Density", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Histogram for Beta
    sns.histplot(
        historical_params["beta"],
        kde=True,
        stat="density",
        ax=axes[0, 1],
        color="lightcoral",
        bins=8,
    )  # 8 bins
    axes[0, 1].set_title(
        "Distribution of Fitted β Parameter",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("β (Shape2)", fontsize=10)
    axes[0, 1].set_ylabel("Density", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # --- Row 2: Log Alpha and Log Beta Histograms ---
    # Histogram for Log Alpha
    # Ensure log_alpha is present in historical_params (it should be from analyze_beta_parameters)
    sns.histplot(
        historical_params["log_alpha"],
        kde=True,
        stat="density",
        ax=axes[1, 0],
        color="mediumseagreen",
        bins=8,
    )  # 8 bins
    axes[1, 0].set_title(
        "Distribution of Fitted Log(α) Parameter",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Log(α)", fontsize=10)
    axes[1, 0].set_ylabel("Density", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Histogram for Log Beta
    # Ensure log_beta is present in historical_params
    sns.histplot(
        historical_params["log_beta"],
        kde=True,
        stat="density",
        ax=axes[1, 1],
        color="orchid",
        bins=8,
    )  # 8 bins
    axes[1, 1].set_title(
        "Distribution of Fitted Log(β) Parameter",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Log(β)", fontsize=10)
    axes[1, 1].set_ylabel("Density", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )  # Adjust rect if needed for suptitle and bottom margin
    plt.show()


def plot_beta_parameter_analysis(all_params_for_scatter):
    """
    Generate a 2x2 grid of scatter plots for beta parameter analysis.

    Assumes all_params_for_scatter is never empty.
    """
    all_params_for_scatter = all_params_for_scatter[
        (all_params_for_scatter['is_predicted'] == False) |
        (all_params_for_scatter['Project'] == 'Predicted Median')]
    

    # Calculate dynamic axis limits, assuming data is present
    x_limits_ab = [0, all_params_for_scatter["alpha"].max() * 1.05]
    y_limits_ab = [0, all_params_for_scatter["beta"].max() * 1.05]

    min_log_alpha = all_params_for_scatter["log_alpha"].min()
    max_log_alpha = all_params_for_scatter["log_alpha"].max()
    min_log_beta = all_params_for_scatter["log_beta"].min()
    max_log_beta = all_params_for_scatter["log_beta"].max()
    padding_factor_log = 0.1
    x_limits_lab = [
        min(0, min_log_alpha),
        max(0, max_log_alpha)
        + abs(max(0, max_log_alpha)) * padding_factor_log,
    ]
    y_limits_lab = [
        min(0, min_log_beta),
        max(0, max_log_beta) + abs(max(0, max_log_beta)) * padding_factor_log,
    ]

    x_limits_mv = [0, all_params_for_scatter["beta_mean"].max() * 1.05]
    y_limits_mv = [0, all_params_for_scatter["beta_variance"].max() * 1.05]

    min_skew = all_params_for_scatter["beta_skewness"].min()
    max_skew = all_params_for_scatter["beta_skewness"].max()
    x_limits_ms = [0, all_params_for_scatter["beta_mean"].max() * 1.05]
    y_limits_ms = [min(0, min_skew) * 1.05, max(0, max_skew) * 1.05]

    fig, axs2 = plt.subplots(2, 2, figsize=(8.5, 5.2))
    fig.suptitle(  # Changed from fig2 to fig
        "Beta Distribution Parameter and Property Analysis",
        fontsize=16,
        fontweight="bold",
        y=.95,
    )

    # Plot 1: Alpha vs Beta
    ax = axs2[0, 0]
    sns.scatterplot(
        data=all_params_for_scatter,
        x="alpha",
        y="beta",
        hue="is_predicted",
        palette={True: "red", False: "blue"},
        # s=100,
        alpha=0.8,
        ax=ax,
    )
    ax.plot(
        [0, max(x_limits_ab[1], y_limits_ab[1])],
        [0, max(x_limits_ab[1], y_limits_ab[1])],
        linestyle="dashed",
        color="gray",
    )
    ax.axhline(0, color="black", linestyle="solid")
    ax.axvline(0, color="black", linestyle="solid")
    ax.set_title(
        "Distribution of Beta Parameters (α vs. β)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("α (Shape1) Parameter", fontsize=10)
    ax.set_ylabel("β (Shape2) Parameter", fontsize=10)
    ax.set_xlim(x_limits_ab)
    ax.set_ylim(y_limits_ab)
    ax.legend().set_visible(False)

    # Plot 2: Log Alpha vs Log Beta
    ax = axs2[0, 1]
    sns.scatterplot(
        data=all_params_for_scatter,
        x="log_alpha",
        y="log_beta",
        hue="is_predicted",
        palette={True: "red", False: "blue"},
        # s=100,
        alpha=0.8,
        ax=ax,
    )
    # # No empty check here as per assumption
    # for idx, row in all_params_for_scatter.iterrows():
    #     if row["is_predicted"]:
    #         ax.text(
    #             row["log_alpha"] + 0.05,
    #             row["log_beta"] + 0.05,
    #             row["Project"],
    #             fontsize=8,
    #             color="red",
    #         )
    ax.plot(
        [
            min(x_limits_lab[0], y_limits_lab[0]),
            max(x_limits_lab[1], y_limits_lab[1]),
        ],
        [
            min(x_limits_lab[0], y_limits_lab[0]),
            max(x_limits_lab[1], y_limits_lab[1]),
        ],
        linestyle="dashed",
        color="gray",
    )
    ax.axhline(0, color="black", linestyle="solid")
    ax.axvline(0, color="black", linestyle="solid")
    ax.set_title(
        "Distribution of Log(Beta Parameters) (Log(α) vs. Log(β))",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Log(α) Parameter", fontsize=10)
    ax.set_ylabel("Log(β) Parameter", fontsize=10)
    ax.set_xlim(x_limits_lab)
    ax.set_ylim(y_limits_lab)
    ax.legend().set_visible(False)

    # Plot 3: Mean vs Variance
    ax = axs2[1, 0]
    sns.scatterplot(
        data=all_params_for_scatter,
        x="beta_mean",
        y="beta_variance",
        hue="is_predicted",
        palette={True: "red", False: "blue"},
        # s=100,
        alpha=0.8,
        ax=ax,
    )
    # No empty check here as per assumption
    # for idx, row in all_params_for_scatter.iterrows():
    #     if row["is_predicted"]:
    #         ax.text(
    #             row["beta_mean"] + 0.01,
    #             row["beta_variance"] + 0.001,
    #             row["Project"],
    #             fontsize=8,
    #             color="red",
    #         )
    ax.set_title(
        "Distribution of Beta Properties (Mean vs. Variance)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Mean", fontsize=10)
    ax.set_ylabel("Variance", fontsize=10)
    ax.set_xlim(x_limits_mv)
    ax.set_ylim(y_limits_mv)
    ax.legend().set_visible(False)

    # Plot 4: Mean vs Skewness
    ax = axs2[1, 1]
    sns.scatterplot(
        data=all_params_for_scatter,
        x="beta_mean",
        y="beta_skewness",
        hue="is_predicted",
        palette={True: "red", False: "blue"},
        # s=100,
        alpha=0.8,
        ax=ax,
    )
    # # No empty check here as per assumption
    # for idx, row in all_params_for_scatter.iterrows():
    #     if row["is_predicted"]:
    #         ax.text(
    #             row["beta_mean"] + 0.01,
    #             row["beta_skewness"] + 0.05 * np.sign(row["beta_skewness"]),
    #             row["Project"],
    #             fontsize=8,
    #             color="red",
    #         )
    ax.axhline(0, color="black", linestyle="solid")
    ax.set_title(
        "Distribution of Beta Properties (Mean vs. Skewness)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Mean", fontsize=10)
    ax.set_ylabel("Skewness", fontsize=10)
    ax.set_xlim(x_limits_ms)
    ax.set_ylim(y_limits_ms)
    ax.legend().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    custom_labels = ["Historical Program", "Predicted Profile"]
    fig.legend(
        handles=[handles[0], handles[1]],
        labels=custom_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        title="Program Type",
        fontsize=10,
        title_fontsize=12,
        frameon=False,
    )
    plt.tight_layout(rect=[0.0, 0.05, 1, 0.98])
    plt.show()


def perform_monte_carlo_simulation(
    historical_log_params_df, n_simulations, max_alpha, max_beta
):
    """
    Perform Monte Carlo simulation of beta parameters and generates predicted profiles.

    Assumes sufficient historical data points (more than 2 programs).
    Returns simulated profiles AND the mean and covariance of the log-parameters,
    as well as the capped simulated α and β values.
    """
    mu_log_params = (
        historical_log_params_df[["log_alpha", "log_beta"]].mean().values
    )
    cov_log_params = (
        historical_log_params_df[["log_alpha", "log_beta"]].cov().values
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            np.linalg.cholesky(cov_log_params)
        except np.linalg.LinAlgError:
            cov_log_params = cov_log_params + np.eye(2) * 1e-6

    median_hist_scale_factor = historical_log_params_df[
        "scale_factor"
    ].median()

    rng = np.random.default_rng(123)
    simulated_log_params = rng.multivariate_normal(
        mean=mu_log_params, cov=cov_log_params, size=n_simulations
    )
    simulated_params_capped = pd.DataFrame(
        simulated_log_params, columns=["log_alpha_sim", "log_beta_sim"]
    )
    simulated_params_capped["alpha_sim"] = np.exp(
        simulated_params_capped["log_alpha_sim"]
    ).clip(upper=max_alpha, lower=0.01)
    simulated_params_capped["beta_sim"] = np.exp(
        simulated_params_capped["log_beta_sim"]
    ).clip(upper=max_beta, lower=0.01)
    simulated_params_capped["simulation_id"] = range(n_simulations)

    mc_predicted_profiles_list = []
    common_time_points_pred = np.linspace(0, 1, 100)
    for _, row in simulated_params_capped.iterrows():
        fitted_inc_expend_pct = _beta_pdf_scaled(
            common_time_points_pred,
            row["alpha_sim"],
            row["beta_sim"],
            median_hist_scale_factor,
        )
        mc_predicted_profiles_list.append(
            pd.DataFrame(
                {
                    "Cum_Time_Pct": common_time_points_pred,
                    "Fitted_Inc_Expend_Pct": fitted_inc_expend_pct,
                    "simulation_id": row["simulation_id"],
                }
            )
        )
    return (
        pd.concat(mc_predicted_profiles_list, ignore_index=True),
        mu_log_params,
        cov_log_params,
        simulated_params_capped,
    )  # Return simulated_params_capped


def plot_monte_carlo_results(
    # Plot median Beta distribution. Plot faded MC draws of other parameters.
    mc_predicted_profiles_df,
    predicted_mean_median_df,
    n_simulations,
    y_axis_limit,
    mu_log_params,
    cov_log_params,
    simulated_params_capped,
):
    """
    Visualizes the results of the Monte Carlo simulation.
    Assumes predicted_mean_median_df is not empty.
    Describes the log-normal parameters, their medians, and the sampled Beta parameters.
    """
    plt.figure(figsize=(8.5, 5.2))
    for sim_id in mc_predicted_profiles_df["simulation_id"].unique():
        subset = mc_predicted_profiles_df[
            mc_predicted_profiles_df["simulation_id"] == sim_id
        ]
        plt.plot(
            subset["Cum_Time_Pct"],
            subset["Fitted_Inc_Expend_Pct"],
            color="gray",
            alpha=0.1,
        )

    mean_pred = predicted_mean_median_df[
        predicted_mean_median_df["Profile_Type"] == "Mean"
    ]
    median_pred = predicted_mean_median_df[
        predicted_mean_median_df["Profile_Type"] == "Median"
    ]

    plt.plot(
        mean_pred["Cum_Time_Pct"],
        mean_pred["Fitted_Inc_Expend_Pct"],
        color="darkred",
        linestyle="solid",
        linewidth=2,
        label="Mean Predicted Profile",
    )
    plt.plot(
        median_pred["Cum_Time_Pct"],
        median_pred["Fitted_Inc_Expend_Pct"],
        color="darkgreen",
        linestyle="solid",
        linewidth=2,
        label="Median Predicted Profile",
    )

    plt.title(
        "Monte Carlo Simulated Predicted Expenditure Profiles",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Normalized Duration", fontsize=12)
    plt.ylabel("Incremental Expenditure", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, y_axis_limit)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, y_axis_limit + 0.01, 0.1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(
        title="Overall Prediction",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )

    # --- Calculations for the new label ---
    # Log-normal parameters
    mu_log_alpha = mu_log_params[0]
    mu_log_beta = mu_log_params[1]

    sigma_log_alpha = np.sqrt(cov_log_params[0][0])
    sigma_log_beta = np.sqrt(cov_log_params[1][1])

    # Median of log-normal (e^mu)
    median_alpha_lognorm = np.exp(mu_log_alpha)
    median_beta_lognorm = np.exp(mu_log_beta)

    # Correlation coefficient
    if sigma_log_alpha > 0 and sigma_log_beta > 0:
        correlation_coeff = cov_log_params[0][1] / (
            sigma_log_alpha * sigma_log_beta
        )
    else:
        correlation_coeff = np.nan

    # Sampled Beta parameters (alpha_sim, beta_sim) statistics
    mean_alpha_sampled = simulated_params_capped["alpha_sim"].mean()
    std_alpha_sampled = simulated_params_capped["alpha_sim"].std()

    mean_beta_sampled = simulated_params_capped["beta_sim"].mean()
    std_beta_sampled = simulated_params_capped["beta_sim"].std()

    # --- Construct the label text ---
    label_text_parts = []

    # Part 1: Sampled Beta distribution (alpha, beta) characteristics
    # label_text_parts.append(
    #     r"$\alpha \approx %.2f \pm %.2f$"
    #     % (mean_alpha_sampled, std_alpha_sampled)
    #     + "\n"
    #     + r"$\beta \approx %.2f \pm %.2f$"
    #     % (mean_beta_sampled, std_beta_sampled)
    # )

    # Part 2: Log-Normal distribution definitions and medians
    label_text_parts.append(
        r"$\alpha \sim \text{LogNormal}(\mu_{\ln\alpha}=%.2f, \sigma_{\ln\alpha}=%.2f)$"
        % (mu_log_alpha, sigma_log_alpha)
        + "\n"
        + r"$\text{Median}_{\alpha} = %.2f$" % median_alpha_lognorm
    )
    label_text_parts.append(
        r"$\beta \sim \text{LogNormal}(\mu_{\ln\beta}=%.2f, \sigma_{\ln\beta}=%.2f)$"
        % (mu_log_beta, sigma_log_beta)
        + "\n"
        + r"$\text{Median}_{\beta} = %.2f$" % median_beta_lognorm
    )

    # Part 3: Correlation
    label_text_parts.append(
        r"$\text{Correl.}(\ln\alpha, \ln\beta) = %.2f$" % correlation_coeff
    )

    full_label_text = "\n".join(label_text_parts)

    plt.text(
        0.02,
        0.98,
        full_label_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    plt.suptitle(
        f"{n_simulations} simulations based on historical log-α/log-β correlation",
        fontsize=10,
        y=0.92,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    plt.show()


def main():
    """Orchestrate data processing, fitting, analysis, and plotting for EDA."""
    df = load_data(CSV_FILE_PATH, REQUIRED_COLUMNS)
    df_augmented = augment_data_with_zero_point(df)

    program_fits_df = fit_beta_distributions(
        df_augmented, MAX_ALPHA_PARAM, MAX_BETA_PARAM
    )
    predicted_program_plot_data_mean_median = calculate_predicted_profiles(
        program_fits_df
    )

    (
        plot_data_original,
        plot_data_fitted_curves,
        all_beta_labels,
        project_levels_ordered,
    ) = prepare_plotting_data(
        df_augmented, program_fits_df, predicted_program_plot_data_mean_median
    )

    plot_phasing_profiles(
        plot_data_original,
        plot_data_fitted_curves,
        all_beta_labels,
        project_levels_ordered,
        Y_AXIS_UPPER_LIMIT,
    )

    all_params_for_scatter = analyze_beta_parameters(
        program_fits_df, predicted_program_plot_data_mean_median
    )

    plot_alpha_beta_histograms(all_params_for_scatter)
    plot_beta_parameter_analysis(all_params_for_scatter)

    correlation_data_for_duration = prepare_correlation_data(
        program_fits_df, df
    )  # Use original df for Max_Year
    analyze_duration_alpha_beta_correlation(correlation_data_for_duration)

    historical_log_params = (
        all_params_for_scatter[~all_params_for_scatter["is_predicted"]][
            ["log_alpha", "log_beta", "scale_factor"]
        ]
        .dropna()
        .copy()
    )

    # Capture the returned simulated_params_capped
    (
        mc_predicted_profiles,
        mc_mu_log_params,
        mc_cov_log_params,
        simulated_params_capped_for_label,
    ) = perform_monte_carlo_simulation(
        historical_log_params, N_SIMULATIONS, MAX_ALPHA_PARAM, MAX_BETA_PARAM
    )
    # Pass simulated_params_capped_for_label to the plotting function
    plot_monte_carlo_results(
        mc_predicted_profiles,
        predicted_program_plot_data_mean_median,
        N_SIMULATIONS,
        Y_AXIS_UPPER_LIMIT,
        mc_mu_log_params,
        mc_cov_log_params,
        simulated_params_capped_for_label,
    )


if __name__ == "__main__":
    main()
