"""Validation helpers for the predict package."""

import warnings


def warn_specialty_mismatch(
    requested: set,
    trained: set,
    source_label: str,
    *,
    stacklevel: int = 3,
) -> None:
    """Emit warnings when requested and trained specialty sets diverge.

    Parameters
    ----------
    requested : set
        Specialties coming from the current request (e.g. Clarity).
    trained : set
        Specialties the model was trained on.
    source_label : str
        Human-readable name for the trained artefact, used in messages
        (e.g. ``"yet-to-arrive model"`` or ``"special_category_dict"``).
    stacklevel : int, optional
        Passed to :func:`warnings.warn` so the warning points to the
        caller rather than this helper.  Default is 3 (caller's caller).
    """
    new_in_request = requested - trained
    missing_from_request = trained - requested
    if new_in_request:
        warnings.warn(
            f"{len(new_in_request)} specialties found in the request but absent "
            f"from the trained {source_label} (models may need retraining): "
            f"{sorted(new_in_request)}",
            stacklevel=stacklevel,
        )
    if missing_from_request:
        warnings.warn(
            f"{len(missing_from_request)} specialties present in the trained "
            f"{source_label} but absent from the request: "
            f"{sorted(missing_from_request)}",
            stacklevel=stacklevel,
        )
