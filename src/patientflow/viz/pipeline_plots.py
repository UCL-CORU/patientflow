"""Visualization functions for the whole-pipeline overview figures.

This module provides functions for creating the pipeline overview plots
that illustrate the end-to-end patient flow prediction process: from
a snapshot of patients currently in the ED, through individual admission
predictions, to aggregate probability distributions of bed demand.

Functions
---------
create_colour_dict : function
    Build a colour mapping dictionary for specialty categories.
prepare_snapshot_data : function
    Filter and prepare a single snapshot of ED visits for plotting.
add_specialty_predictions : function
    Attach a per-row specialty probability dict to a snapshot DataFrame.
build_pipeline_prediction_inputs : function
    Build per-service prediction inputs for a given snapshot using
    ``build_service_data`` from ``patientflow.predict.service``.
in_ed_now_plot : function
    Scatter plot of patients currently in the ED, optionally coloured by
    predicted probability of admission.
main : function
    End-to-end pipeline that trains models, selects a random snapshot,
    and generates all overview plots (a, b, c, e, f, g, h, i, j, k) to a
    timestamped output folder.
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LOCATION_MAPPING = {
    "utc": "Minors",
    "majors": "Majors/Resus",
    "sdec": "Other",
    "paeds": "Other",
    "waiting": "Other",
    "rat": "Majors/Resus",
    "sdec_waiting": "Other",
    "resus": "Majors/Resus",
    "taf": "Other",
}
#: Default mapping from raw ``current_location_type`` values to
#: simplified display categories used on the y-axis of pipeline plots.

DEFAULT_CATEGORY_ORDER = ["Majors/Resus", "Minors", "Other"]
#: Default ordered list of display category names for the
#: y-axis of pipeline plots.


def create_colour_dict():
    """Build a colour mapping dictionary for specialty categories.

    Returns a nested dictionary with two keys:

    * ``"single"`` – maps each specialty name to a single hex colour.
    * ``"spectrum"`` – maps each specialty name to a
      `matplotlib.colors.LinearSegmentedColormap` that runs from
      a pale tint to the corresponding single colour at full intensity.

    Returns
    -------
    dict
        ``{"single": {<specialty>: <hex>}, "spectrum": {<specialty>: <cmap>}}``
    """
    spec_colour_dict = {
        "single": {
            "medical": "#ED7D31",
            "surgical": "#70AD47",
            "haem/onc": "#FFC000",
            "paediatric": "#5B9BD5",
            "all": "#44546A",
            "window": "#A9A9A9",
        },
        "spectrum": {},
    }

    def _generate_continuous_colormap(base_color, start_fraction=0.2):
        base = mcolors.to_rgb(base_color)
        # Blend from a tinted white towards the full base colour.
        # start_fraction=0 gives pure white at the low end;
        # higher values (e.g. 0.2–0.6) make the pale end more saturated.
        start = tuple(1.0 - start_fraction * (1.0 - c) for c in base)
        cdict = {
            "red": [(0.0, start[0], start[0]), (1.0, base[0], base[0])],
            "green": [(0.0, start[1], start[1]), (1.0, base[1], base[1])],
            "blue": [(0.0, start[2], start[2]), (1.0, base[2], base[2])],
        }
        return mcolors.LinearSegmentedColormap(
            "custom_colormap", segmentdata=cdict, N=256
        )

    for spec, color in spec_colour_dict["single"].items():
        spec_colour_dict["spectrum"][spec] = _generate_continuous_colormap(color)

    return spec_colour_dict


def prepare_snapshot_data(
    visits,
    snapshot_date,
    prediction_time,
    location_mapping=None,
    category_order=None,
    exclude_locations=None,
):
    """Filter and prepare a single snapshot of ED visits for plotting.

    Selects visits for the given *snapshot_date* and *prediction_time*,
    optionally excludes certain location types (e.g. ``"OTF"``), and maps
    the raw ``current_location_type`` values to simplified display
    categories stored in a new ``loc_new`` column.

    Parameters
    ----------
    visits : pandas.DataFrame
        The full visits DataFrame.  Must contain at least the columns
        ``snapshot_date``, ``prediction_time``, and
        ``current_location_type``.
    snapshot_date : object
        The date to select (must match values in ``visits["snapshot_date"]``).
    prediction_time : tuple
        ``(hour, minute)`` prediction time to select.
    location_mapping : dict or None, optional
        Mapping from raw ``current_location_type`` values to display
        category names.  Defaults to :data:`DEFAULT_LOCATION_MAPPING`.
    category_order : list of str or None, optional
        Ordered list of display category names for the y-axis.
        Defaults to :data:`DEFAULT_CATEGORY_ORDER`.
    exclude_locations : list of str or None, optional
        Raw location types to exclude.  Defaults to ``["OTF"]``.

    Returns
    -------
    pandas.DataFrame
        A copy of the filtered data with a ``loc_new`` categorical column
        added.
    """
    if location_mapping is None:
        location_mapping = DEFAULT_LOCATION_MAPPING
    if category_order is None:
        category_order = DEFAULT_CATEGORY_ORDER
    if exclude_locations is None:
        exclude_locations = ["OTF"]

    mask = (visits["snapshot_date"] == snapshot_date) & (
        visits["prediction_time"] == prediction_time
    )
    for loc in exclude_locations:
        mask = mask & (visits["current_location_type"] != loc)

    ex = visits[mask].copy()

    ex["loc_new"] = ex["current_location_type"].map(location_mapping)
    ex["loc_new"] = pd.Categorical(
        ex["loc_new"], categories=category_order, ordered=True
    )

    return ex


def add_specialty_predictions(
    df,
    specialty_model,
    specialties=None,
):
    """Attach a per-row specialty probability dict to a snapshot DataFrame.

    For each row the trained *specialty_model* is called with the full row
    (as a `pandas.Series`) so that models which need additional
    context (e.g. [MultiSubgroupPredictor][patientflow.predictors.subgroup_predictor.MultiSubgroupPredictor])
    can determine the correct sub-model.  The result is stored in a new
    ``specialty_prob`` column whose values are ``dict[str, float]``.

    Parameters
    ----------
    df : pandas.DataFrame
        Snapshot data – typically the output of [prepare_snapshot_data][patientflow.viz.pipeline_plots.prepare_snapshot_data].
        Must contain at least the column used by
        ``specialty_model.input_var`` (usually ``consultation_sequence``).
    specialty_model : object
        Trained specialty prediction model.  Must expose ``input_var``
        (str) and ``predict(row)`` where *row* is a `pandas.Series`.
    specialties : list of str or None, optional
        Specialty names to ensure are present in every returned dict
        (missing keys are filled with ``0``).
        Defaults to ``["medical", "surgical", "haem/onc", "paediatric"]``.

    Returns
    -------
    pandas.DataFrame
        A copy of *df* with a ``specialty_prob`` column added.  Each
        value is a ``dict`` mapping specialty name to probability.
    """
    if specialties is None:
        specialties = ["medical", "surgical", "haem/onc", "paediatric"]

    result = df.copy()

    # Ensure the input sequence column contains tuples (handles NaN / list)
    input_var = specialty_model.input_var

    def _to_tuple(x):
        if isinstance(x, float) and pd.isna(x):
            return ()
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        return ()

    result[input_var] = result[input_var].apply(_to_tuple)

    # Predict per row, passing the full Series so subgroup models work
    def _predict_row(row):
        probs = specialty_model.predict(row)
        # Ensure all requested specialties are present
        return {s: probs.get(s, 0) for s in set(specialties) | set(probs.keys())}

    result["specialty_prob"] = result.apply(_predict_row, axis=1)

    return result


def build_pipeline_prediction_inputs(
    prediction_inputs,
    prediction_time,
    prediction_snapshots,
    prediction_window,
    use_admission_in_window_prob=True,
):
    """Build per-service prediction inputs for a snapshot.

    This is a convenience wrapper around
    [build_service_data][patientflow.predict.service.build_service_data] that unpacks
    the prediction inputs dictionary returned by
    [prepare_prediction_inputs][patientflow.train.emergency_demand.prepare_prediction_inputs]
    and retrieves the appropriate admission classifier for the given
    *prediction_time*.

    Parameters
    ----------
    prediction_inputs : dict
        Dictionary returned by ``prepare_prediction_inputs()``, containing
        keys ``"admission_models"``, ``"specialty_model"``,
        ``"yta_model"``, ``"specialties"``, and ``"config"``.
    prediction_time : tuple of (int, int)
        ``(hour, minute)`` of the prediction moment.
    prediction_snapshots : pandas.DataFrame
        Snapshot of patients currently in the ED at the prediction moment.
        Must contain an ``elapsed_los`` column (in seconds – it will be
        converted to [timedelta][datetime.timedelta] internally).
    prediction_window : datetime.timedelta
        Horizon over which to predict demand.
    use_admission_in_window_prob : bool, default=True
        Whether to weight current ED admissions by their probability of
        being admitted within the prediction window.  When ``False``,
        every current ED patient is treated as certain to be admitted
        (probability = 1.0).

    Returns
    -------
    dict of str to ServicePredictionInputs
        Per-service prediction input objects ready for downstream
        aggregation or plotting.
    """
    from patientflow.load import get_model_key
    from patientflow.predict.service import build_service_data

    admission_models = prediction_inputs["admission_models"]
    specialty_model = prediction_inputs["specialty_model"]
    yta_model = prediction_inputs["yta_model"]
    specialties = prediction_inputs["specialties"]
    config = prediction_inputs["config"]

    # Find the admission classifier that matches the prediction_time
    admission_model = None
    for _, trained_model in admission_models.items():
        if trained_model.training_results.prediction_time == prediction_time:
            admission_model = trained_model
            break

    if admission_model is None:
        # Fall back to key-based lookup
        admission_model = admission_models[get_model_key("admissions", prediction_time)]

    # Ensure elapsed_los is timedelta
    snapshots = prediction_snapshots.copy(deep=True)
    if not pd.api.types.is_timedelta64_dtype(snapshots["elapsed_los"]):
        snapshots["elapsed_los"] = pd.to_timedelta(snapshots["elapsed_los"], unit="s")

    x1, y1 = config["x1"], config["y1"]
    x2, y2 = config["x2"], config["y2"]

    return build_service_data(
        models=(
            admission_model,
            None,
            specialty_model,
            yta_model,
            None,
            None,
            None,
        ),
        prediction_time=prediction_time,
        ed_snapshots=snapshots,
        inpatient_snapshots=None,
        specialties=specialties,
        prediction_window=prediction_window,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        use_admission_in_window_prob=use_admission_in_window_prob,
    )


def in_ed_now_plot(
    ex,
    title,
    media_file_path=None,
    file_name=None,
    figsize=(6, 3),
    include_titles=False,
    truncate_at_hours=8,
    colour=False,
    text_size=None,
    jitter_amount=0.1,
    size=50,
    preds_col="preds",
    colour_map="Spectral_r",
    return_figure=False,
    ax=None,
):
    """Scatter plot of patients currently in the ED.

    Each dot represents one patient, positioned by elapsed length-of-stay
    on the x-axis and ED pathway (``loc_new`` column) on the y-axis.
    Optionally, dots can be coloured by predicted probability of admission.

    Parameters
    ----------
    ex : pandas.DataFrame
        Snapshot data.  Must contain ``elapsed_los`` (in seconds) and
        ``loc_new`` (categorical) columns.  When *colour* is ``True`` the
        column named by *preds_col* must also be present.
    title : str
        Title text for the figure.
    media_file_path : str, Path or None, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    file_name : str or None, optional
        File name (without path) to save the figure as.  Spaces are
        replaced with underscores.  Only used when *media_file_path* is
        not ``None``.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Default ``(6, 3)``.
    include_titles : bool, optional
        If ``True``, render the title, x-label, and y-label on the plot.
    truncate_at_hours : int or float, optional
        Maximum elapsed hours to show.  Patients with a longer stay are
        excluded.  Default ``8``.
    colour : bool, optional
        If ``True``, colour the dots by the values in *preds_col* and
        show a colour-bar.  Default ``False``.
    text_size : int or None, optional
        Font size for tick labels and (if shown) titles.
    jitter_amount : float, optional
        Vertical jitter applied to dots.  Default ``0.1``.
    size : int, optional
        Marker size.  Default ``50``.
    preds_col : str, optional
        Column name for the prediction probabilities.
        Default ``"preds"``.
    colour_map : str or Colormap, optional
        Matplotlib colour map (or name) used when *colour* is ``True``.
        Default ``"Spectral_r"``.
    return_figure : bool, optional
        If ``True`` return ``(fig, ax)`` instead of calling
        ``plt.show()``.  Default ``False``.
    ax : matplotlib.axes.Axes or None, optional
        An existing Axes to draw into.  When provided, *figsize*,
        *media_file_path*, *file_name*, and *return_figure* are ignored
        and the function draws directly into *ax* (useful for subplots).
        Default is ``None`` (creates a new figure).

    Returns
    -------
    matplotlib.axes.Axes or tuple of (Figure, Axes) or None
        Returns *ax* when an external Axes is provided, ``(fig, ax)``
        when *return_figure* is ``True``, or ``None`` otherwise.
    """
    spec_colour_dict = create_colour_dict()

    figsize_x, figsize_y = figsize

    ex = ex[ex.elapsed_los / 3600 < truncate_at_hours].copy()

    # Map ordinal categories to numerical y-values
    unique_locations = sorted(ex["loc_new"].unique())
    loc_to_num = {loc: i for i, loc in enumerate(unique_locations)}

    # When an external Axes is provided, draw into it directly
    use_external_ax = ax is not None

    cbar = None  # keep reference for text_size adjustment

    if colour:
        if use_external_ax:
            target_ax = ax
            fig = ax.figure
        else:
            fig, target_ax = plt.subplots(figsize=(figsize_x, figsize_y))
        for location, group in ex.groupby("loc_new", observed=True):
            jittered_y = loc_to_num[location] + np.random.uniform(
                -jitter_amount, jitter_amount, size=len(group)
            )
            target_ax.scatter(
                group["elapsed_los"] / 3600,
                jittered_y,
                c=group[preds_col],
                cmap=colour_map,
                vmin=0,
                vmax=1,
                label=location,
                s=size,
            )
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=colour_map, norm=plt.Normalize(vmin=0, vmax=1)),
            ax=target_ax,
            orientation="vertical",
        )
    else:
        if use_external_ax:
            target_ax = ax
            fig = ax.figure
        else:
            fig, target_ax = plt.subplots(figsize=(figsize_x - 1, figsize_y))
        for location, group in ex.groupby("loc_new", observed=True):
            jittered_y = loc_to_num[location] + np.random.uniform(
                -jitter_amount, jitter_amount, size=len(group)
            )
            target_ax.scatter(
                group["elapsed_los"] / 3600,
                jittered_y,
                color=spec_colour_dict["single"]["all"],
                label=location,
                s=size,
            )

    target_ax.set_xlim(0, truncate_at_hours)
    target_ax.invert_yaxis()

    target_ax.set_yticks(range(len(unique_locations)))
    target_ax.set_yticklabels(unique_locations)

    if text_size:
        target_ax.tick_params(axis="both", which="major", labelsize=text_size)
        if colour and cbar is not None:
            cbar.ax.tick_params(labelsize=text_size)

    if include_titles:
        target_ax.set_title(title, fontsize=text_size)
        target_ax.set_xlabel("Hours since admission")
        target_ax.set_ylabel("ED Pathway")

    # If using an external Axes, skip figure-level operations and return
    if use_external_ax:
        return target_ax

    fig.tight_layout()

    if media_file_path is not None and file_name is not None:
        directory_path = Path(media_file_path)
        os.makedirs(directory_path, exist_ok=True)
        fig.savefig(directory_path / file_name.replace(" ", "_"), dpi=300)

    if return_figure:
        return fig, target_ax

    plt.show()
    return None


def _save_fig(fig, save_dir, filename, label):
    """Save a figure to *save_dir* and close it."""
    fig.savefig(save_dir / filename, dpi=300)
    plt.close(fig)
    print(f"  Saved figure ({label})")


def main(
    prediction_time=(9, 30),
    output_dir="pipeline_plots_output",
    data_folder_name="data-public",
    random_state=0,
    include_titles=False,
    show_observed=False,
):
    """Run the full pipeline-overview plot generation.

    Trains all prediction models, randomly selects a snapshot date from
    the ED visits data, and generates a set of overview figures saved to
    a timestamped sub-folder inside *output_dir*.

    The generated figures are:

    * **(a)** Patients currently in the ED (scatter, no colour).
    * **(b)** Patients currently in the ED coloured by predicted
      probability of admission.
    * **(c)** Aggregate probability distribution for number of beds
      needed.
    * **(e)** Per-specialty scatter plots using specialty-specific
      colour maps.
    * **(f)** Per-specialty (2×2) probability distributions for beds
      needed from current ED patients (without admission-in-window
      probability).
    * **(g)** Aspirational curve showing the probability of admission
      within the prediction window as a function of elapsed time.
    * **(h)** Patients coloured by probability of admission within the
      prediction window.
    * **(i)** Per-specialty (2×2) probability distributions for beds
      needed with admission-in-window probability.
    * **(j)** Per-specialty (2×2) probability distributions for beds
      needed from yet-to-arrive patients only.
    * **(k)** Per-specialty (2×2) probability distributions for beds
      needed combining current ED and yet-to-arrive patients.

    Parameters
    ----------
    prediction_time : tuple of (int, int), optional
        ``(hour, minute)`` of the prediction moment.
        Default ``(9, 30)``.
    output_dir : str or Path, optional
        Root directory in which a dated sub-folder will be created for
        the plots.  Default ``"pipeline_plots_output"``.
    data_folder_name : str, optional
        Name of the data folder containing the training datasets.
        Default ``"data-public"``.
    random_state : int, optional
        Random state used for sampling the snapshot date.
        Default ``0``.
    include_titles : bool, optional
        If ``True``, render titles, axis labels, and colour-bar labels
        on every plot.  The output folder name is also suffixed with
        ``_with_titles`` so titled and untitled runs do not overwrite
        each other.  Default ``False``.
    show_observed : bool, optional
        If ``True``, draw a vertical line on probability distribution
        plots to indicate the actual observed value.
        Default ``False``.

    Returns
    -------
    Path
        The directory where the figures were saved.
    """
    from patientflow.aggregate import (
        model_input_to_pred_proba,
        pred_proba_to_agg_predicted,
    )
    from patientflow.calculate.admission_in_prediction_window import (
        calculate_probability,
    )
    from patientflow.predict.demand import DemandPredictor, FlowSelection
    from patientflow.train.emergency_demand import prepare_prediction_inputs
    from patientflow.viz.probability_distribution import plot_prob_dist
    from patientflow.viz.utils import format_prediction_time

    # ------------------------------------------------------------------
    # 1. Train / load models
    # ------------------------------------------------------------------
    print("Training models …")
    prediction_inputs = prepare_prediction_inputs(data_folder_name)

    ed_visits = prediction_inputs["ed_visits"]
    config = prediction_inputs["config"]

    # ------------------------------------------------------------------
    # 2. Pick a random snapshot date
    # ------------------------------------------------------------------
    snapshot_date = (
        ed_visits["snapshot_date"].sample(n=1, random_state=random_state).iloc[0]
    )
    print(f"Selected snapshot date: {snapshot_date}")

    # ------------------------------------------------------------------
    # 3. Prepare snapshot data
    # ------------------------------------------------------------------
    ex = prepare_snapshot_data(
        ed_visits,
        snapshot_date=snapshot_date,
        prediction_time=prediction_time,
    )

    # ------------------------------------------------------------------
    # 4. Create output folder:  <output_dir>/<snapshot_date>_<HH>-<MM>
    # ------------------------------------------------------------------
    time_str = format_prediction_time(prediction_time).replace(":", "-")
    folder_name = f"{snapshot_date}_{time_str}"
    if include_titles:
        folder_name += "_with_titles"
    save_dir = (Path(output_dir) / folder_name).expanduser().resolve()
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to {save_dir}")

    spec_colour_dict = create_colour_dict()
    title_base = (
        f"Patients in ED at {snapshot_date} "
        f"{format_prediction_time(prediction_time)}"
    )
    prediction_window = timedelta(minutes=config["prediction_window"])
    plot_order = ["medical", "surgical", "haem/onc", "paediatric"]

    # ------------------------------------------------------------------
    # Figure (a) – patients in ED, no colour
    # ------------------------------------------------------------------
    fig_a, _ = in_ed_now_plot(
        ex,
        title=title_base,
        include_titles=include_titles,
        return_figure=True,
    )
    _save_fig(fig_a, save_dir, "figure_a_patients_in_ed.png", "a")

    # ------------------------------------------------------------------
    # Figure (b) – patients coloured by admission probability
    # ------------------------------------------------------------------
    admission_model = None
    for _key, trained_model in prediction_inputs["admission_models"].items():
        if trained_model.training_results.prediction_time == prediction_time:
            admission_model = trained_model.calibrated_pipeline
            break

    if admission_model is None:
        raise RuntimeError(
            f"No admission model found for prediction_time={prediction_time}"
        )

    preds = model_input_to_pred_proba(ex, admission_model)
    ex["preds"] = preds["pred_proba"].values

    fig_b, _ = in_ed_now_plot(
        ex,
        title=f"{title_base} with predicted probability of admission",
        colour=True,
        colour_map="Spectral_r",
        include_titles=include_titles,
        return_figure=True,
    )
    _save_fig(fig_b, save_dir, "figure_b_admission_probability.png", "b")

    # ------------------------------------------------------------------
    # Figure (c) – aggregate probability distribution for beds needed
    # ------------------------------------------------------------------
    agg_predicted = pred_proba_to_agg_predicted(preds)
    fig_c = plot_prob_dist(
        agg_predicted,
        bar_colour=spec_colour_dict["single"]["all"],
        truncate_at_beds=20,
        title=(
            f"Probability distribution for number of beds needed\n"
            f"for patients in ED at {snapshot_date} "
            f"{format_prediction_time(prediction_time)}"
        ),
        include_titles=include_titles,
        return_figure=True,
    )
    if show_observed:
        observed_admissions = int(ex["is_admitted"].sum())
        ax_c = fig_c.gca()
        ax_c.axvline(
            x=observed_admissions,
            linewidth=2,
            linestyle="--",
            color="red",
            label=f"Observed: {observed_admissions}",
        )
        ax_c.legend(loc="upper right")
        fig_c.tight_layout()
    _save_fig(fig_c, save_dir, "figure_c_aggregate_bed_demand.png", "c")

    # ------------------------------------------------------------------
    # Figure (e) – per-specialty scatter plots (specialty colour maps)
    # ------------------------------------------------------------------
    ex_with_specialty = add_specialty_predictions(
        ex,
        specialty_model=prediction_inputs["specialty_model"],
        specialties=prediction_inputs["specialties"],
    )

    fig_e, axes_e = plt.subplots(2, 2, figsize=(12, 8))
    for ax, specialty in zip(axes_e.flat, plot_order):
        in_ed_now_plot(
            ex_with_specialty,
            title=specialty.title(),
            colour=True,
            colour_map=spec_colour_dict["spectrum"].get(specialty, "Spectral_r"),
            include_titles=include_titles,
            ax=ax,
            text_size=14,
        )
    if include_titles:
        fig_e.suptitle(
            f"{title_base}\nwith predicted probability of admission",
            fontsize=16,
        )
    fig_e.tight_layout()
    _save_fig(fig_e, save_dir, "figure_e_specialty_scatter.png", "e")

    # ------------------------------------------------------------------
    # Figure (f) – per-specialty bed demand (no admission-in-window prob)
    # ------------------------------------------------------------------
    predictor = DemandPredictor(k_sigma=8.0)

    service_inputs = build_pipeline_prediction_inputs(
        prediction_inputs,
        prediction_time=prediction_time,
        prediction_snapshots=ex,
        prediction_window=prediction_window,
        use_admission_in_window_prob=False,
    )

    fig_f, axes_f = plt.subplots(2, 2, figsize=(12, 8))
    for ax, specialty in zip(axes_f.flat, plot_order):
        bundle = predictor.predict_service(
            inputs=service_inputs[specialty],
            flow_selection=FlowSelection.custom(
                include_ed_current=True,
                include_ed_yta=False,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
                cohort="emergency",
            ),
        )
        plot_prob_dist(
            bundle.arrivals.probabilities,
            title=specialty.title(),
            truncate_at_beds=15,
            bar_colour=spec_colour_dict["single"].get(specialty, "#5B9BD5"),
            include_titles=include_titles,
            ax=ax,
            text_size=14,
        )
    if include_titles:
        fig_f.suptitle(
            f"Beds needed by specialty\n{title_base}",
            fontsize=16,
        )
    fig_f.tight_layout()
    _save_fig(fig_f, save_dir, "figure_f_per_specialty_bed_demand.png", "f")

    # ------------------------------------------------------------------
    # Figure (g) – aspirational curve for admission in prediction window
    # ------------------------------------------------------------------
    from patientflow.viz.aspirational_curve import plot_curve

    fig_g = plot_curve(
        title=("Aspirational curve for admission\n" "within prediction window"),
        x1=config["x1"],
        y1=config["y1"],
        x2=config["x2"],
        y2=config["y2"],
        figsize=(6, 3) if not include_titles else (10, 5),
        include_titles=include_titles,
        legend_loc="lower right",
        return_figure=True,
    )
    _save_fig(fig_g, save_dir, "figure_g_aspirational_curve.png", "g")

    # ------------------------------------------------------------------
    # Figure (h) – patients coloured by prob of admission in window
    # ------------------------------------------------------------------
    ex["prob_adm_in_window"] = ex.apply(
        lambda row: calculate_probability(
            timedelta(seconds=row["elapsed_los"]),
            prediction_window,
            x1=config["x1"],
            y1=config["y1"],
            x2=config["x2"],
            y2=config["y2"],
        ),
        axis=1,
    )

    fig_h, _ = in_ed_now_plot(
        ex,
        title=(
            f"{title_base}\n"
            f"with predicted probability of admission in prediction window"
        ),
        colour=True,
        colour_map=spec_colour_dict["spectrum"].get("all", "Spectral_r"),
        preds_col="prob_adm_in_window",
        include_titles=include_titles,
        return_figure=True,
    )
    _save_fig(fig_h, save_dir, "figure_h_admission_in_window.png", "h")

    # ------------------------------------------------------------------
    # Figure (i) – per-specialty bed demand WITH admission-in-window prob
    # ------------------------------------------------------------------
    service_inputs_with_window = build_pipeline_prediction_inputs(
        prediction_inputs,
        prediction_time=prediction_time,
        prediction_snapshots=ex,
        prediction_window=prediction_window,
        use_admission_in_window_prob=True,
    )

    fig_i, axes_i = plt.subplots(2, 2, figsize=(12, 8))
    for ax, specialty in zip(axes_i.flat, plot_order):
        bundle = predictor.predict_service(
            inputs=service_inputs_with_window[specialty],
            flow_selection=FlowSelection.custom(
                include_ed_current=True,
                include_ed_yta=False,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
                cohort="emergency",
            ),
        )
        plot_prob_dist(
            bundle.arrivals.probabilities,
            title=specialty.title(),
            truncate_at_beds=15,
            bar_colour=spec_colour_dict["single"].get(specialty, "#5B9BD5"),
            include_titles=include_titles,
            ax=ax,
            text_size=14,
        )
    if include_titles:
        fig_i.suptitle(
            f"Beds needed by specialty (within prediction window)\n" f"{title_base}",
            fontsize=16,
        )
    fig_i.tight_layout()
    _save_fig(
        fig_i,
        save_dir,
        "figure_i_per_specialty_bed_demand_in_window.png",
        "i",
    )

    # ------------------------------------------------------------------
    # Figure (j) – per-specialty yet-to-arrive bed demand
    # ------------------------------------------------------------------
    fig_j, axes_j = plt.subplots(2, 2, figsize=(12, 8))
    for ax, specialty in zip(axes_j.flat, plot_order):
        bundle = predictor.predict_service(
            inputs=service_inputs_with_window[specialty],
            flow_selection=FlowSelection.custom(
                include_ed_current=False,
                include_ed_yta=True,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
                cohort="emergency",
            ),
        )
        plot_prob_dist(
            bundle.arrivals.probabilities,
            title=specialty.title(),
            truncate_at_beds=15,
            bar_colour=spec_colour_dict["single"].get(specialty, "#5B9BD5"),
            include_titles=include_titles,
            ax=ax,
            text_size=14,
        )
    if include_titles:
        fig_j.suptitle(
            f"Beds needed by specialty (yet-to-arrive)\n{title_base}",
            fontsize=16,
        )
    fig_j.tight_layout()
    _save_fig(
        fig_j,
        save_dir,
        "figure_j_per_specialty_yet_to_arrive.png",
        "j",
    )

    # ------------------------------------------------------------------
    # Figure (k) – combined ED + yet-to-arrive bed demand
    # ------------------------------------------------------------------
    fig_k, axes_k = plt.subplots(2, 2, figsize=(12, 8))
    for ax, specialty in zip(axes_k.flat, plot_order):
        bundle = predictor.predict_service(
            inputs=service_inputs_with_window[specialty],
            flow_selection=FlowSelection.custom(
                include_ed_current=True,
                include_ed_yta=True,
                include_non_ed_yta=False,
                include_elective_yta=False,
                include_transfers_in=False,
                include_departures=False,
                cohort="emergency",
            ),
        )
        plot_prob_dist(
            bundle.arrivals.probabilities,
            title=specialty.title(),
            truncate_at_beds=20,
            bar_colour=spec_colour_dict["single"].get(specialty, "#5B9BD5"),
            include_titles=include_titles,
            ax=ax,
            text_size=14,
        )
    if include_titles:
        fig_k.suptitle(
            f"Beds needed by specialty (current ED and yet-to-arrive)\n"
            f"{title_base}",
            fontsize=16,
        )
    fig_k.tight_layout()
    _save_fig(
        fig_k,
        save_dir,
        "figure_k_per_specialty_combined.png",
        "k",
    )

    print(f"\nAll figures saved to {save_dir}")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate whole-pipeline overview plots.",
    )
    parser.add_argument(
        "--prediction-time",
        type=str,
        default="9,30",
        help=(
            "Prediction time as 'HOUR,MINUTE' (e.g. '9,30' or '15,30'). "
            "Default: '9,30'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline_plots_output",
        help=(
            "Root directory for saving the plots.  A sub-folder named "
            "'<snapshot_date>_<HH>-<MM>' will be created inside it. "
            "Default: 'pipeline_plots_output'."
        ),
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="data-public",
        help="Name of the data folder.  Default: 'data-public'.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for snapshot date sampling.  Default: 0.",
    )
    parser.add_argument(
        "--include-titles",
        action="store_true",
        default=False,
        help=(
            "Include titles and axis labels on every plot.  "
            "Appends '_with_titles' to the output folder name."
        ),
    )
    parser.add_argument(
        "--show-observed",
        action="store_true",
        default=False,
        help="Show observed values as vertical lines on distribution plots.",
    )
    args = parser.parse_args()

    # Parse prediction_time from "H,M" string to tuple
    parts = args.prediction_time.split(",")
    if len(parts) != 2:
        parser.error("prediction-time must be in 'HOUR,MINUTE' format (e.g. '9,30')")
    prediction_time_tuple = (int(parts[0]), int(parts[1]))

    main(
        prediction_time=prediction_time_tuple,
        output_dir=args.output_dir,
        data_folder_name=args.data_folder,
        random_state=args.random_state,
        include_titles=args.include_titles,
        show_observed=args.show_observed,
    )
