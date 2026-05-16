"""Utility functions for visualization and data formatting.

This module provides helper functions for cleaning and formatting data
for visualization purposes, including filename cleaning and prediction time formatting.

Functions
---------
clean_title_for_filename : function
    Clean a title string to make it suitable for use in filenames
format_prediction_time : function
    Format prediction time to 'HH:MM' format
pyplot_show_if : function
    Call ``matplotlib.pyplot.show`` only when the caller requests display
apply_figure_suptitle : function
    Set a figure suptitle with room reserved above the subplot grid
"""


def clean_title_for_filename(title):
    """Clean a title string to make it suitable for use in filenames.

    Parameters
    ----------
    title : str
        The title to clean.

    Returns
    -------
    str
        The cleaned title, safe for use in filenames.
    """
    replacements = {" ": "_", "%": "", "\n": "", ",": "", ".": ""}

    clean_title = title
    for old, new in replacements.items():
        clean_title = clean_title.replace(old, new)
    return clean_title


def format_prediction_time(prediction_time):
    """Format prediction time to 'HH:MM' format.

    Parameters
    ----------
    prediction_time : str or tuple
        Either:
            - A string in 'HHMM' format, possibly containing underscores
            - A tuple of (hour, minute)

    Returns
    -------
    str
        Formatted time string in 'HH:MM' format.
    """
    if isinstance(prediction_time, tuple):
        hour, minute = prediction_time
        return f"{hour:02d}:{minute:02d}"
    else:
        # Split the string by underscores and take the last element
        last_part = prediction_time.split("_")[-1]
        # Add a colon in the middle
        return f"{last_part[:2]}:{last_part[2:]}"


def apply_figure_suptitle(
    fig,
    suptitle: str | None,
    *,
    fontsize: int = 14,
    top: float = 0.88,
) -> None:
    """Set a figure suptitle and shrink the subplot area so the title is not clipped.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate.
    suptitle : str or None
        Title text; no-op when ``None`` or empty.
    fontsize : int, default=14
        Suptitle font size.
    top : float, default=0.88
        Top margin for ``subplots_adjust`` (lower leaves more room for the title).
    """
    if not suptitle:
        return
    fig.suptitle(suptitle, fontsize=fontsize, y=1.02)
    fig.subplots_adjust(top=top)


def pyplot_show_if(show: bool) -> None:
    """Call :func:`matplotlib.pyplot.show` only when ``show`` is true.

    Non-interactive backends (for example Agg, common in notebooks and CI)
    warn if ``show()`` is called with no interactive display. Plotting helpers
    should default ``show`` to false and only call this when the user opts in.
    """
    if not show:
        return
    import matplotlib.pyplot as plt

    plt.show()
