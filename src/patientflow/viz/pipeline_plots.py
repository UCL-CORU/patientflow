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
"""

import os
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
"""dict : Default mapping from raw ``current_location_type`` values to
simplified display categories used on the y-axis of pipeline plots."""

DEFAULT_CATEGORY_ORDER = ["Majors/Resus", "Minors", "Other"]
"""list of str : Default ordered list of display category names for the
y-axis of pipeline plots."""


def create_colour_dict():
    """Build a colour mapping dictionary for specialty categories.

    Returns a nested dictionary with two keys:

    * ``"single"`` – maps each specialty name to a single hex colour.
    * ``"spectrum"`` – maps each specialty name to a
      :class:`~matplotlib.colors.LinearSegmentedColormap` that runs from
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
    (as a :class:`pandas.Series`) so that models which need additional
    context (e.g. :class:`~patientflow.predictors.subgroup_predictor.MultiSubgroupPredictor`)
    can determine the correct sub-model.  The result is stored in a new
    ``specialty_prob`` column whose values are ``dict[str, float]``.

    Parameters
    ----------
    df : pandas.DataFrame
        Snapshot data – typically the output of :func:`prepare_snapshot_data`.
        Must contain at least the column used by
        ``specialty_model.input_var`` (usually ``consultation_sequence``).
    specialty_model : object
        Trained specialty prediction model.  Must expose ``input_var``
        (str) and ``predict(row)`` where *row* is a :class:`pandas.Series`.
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
    :func:`~patientflow.predict.service.build_service_data` that unpacks
    the prediction inputs dictionary returned by
    :func:`~patientflow.train.emergency_demand.prepare_prediction_inputs`
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
        converted to :class:`~datetime.timedelta` internally).
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
        admission_model = admission_models[
            get_model_key("admissions", prediction_time)
        ]

    # Ensure elapsed_los is timedelta
    snapshots = prediction_snapshots.copy(deep=True)
    if not pd.api.types.is_timedelta64_dtype(snapshots["elapsed_los"]):
        snapshots["elapsed_los"] = pd.to_timedelta(
            snapshots["elapsed_los"], unit="s"
        )

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
    title_suffix="admission",
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
    title_suffix : str, optional
        Appended to *title* when *colour* is ``True``
        (e.g. ``"admission"``).  Default ``"admission"``.
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
        title = title + " with predicted probability of " + title_suffix
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
