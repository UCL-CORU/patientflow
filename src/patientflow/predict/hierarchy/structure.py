"""Hierarchical structure definitions.

This module provides the Hierarchy class and related utilities for representing
and populating organizational hierarchies from YAML configurations or DataFrames.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml
import pandas as pd


class EntityType:
    """Represents an entity type in the hierarchy.

    This class is used to represent entity types dynamically based on the
    hierarchy configuration.
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"EntityType('{self.name}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, EntityType):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)

    @classmethod
    def from_string(cls, value: str) -> "EntityType":
        """Create EntityType from string."""
        return cls(value)


@dataclass
class HierarchyLevel:
    """Represents a level in the hierarchy with its configuration."""

    entity_type: EntityType
    parent_type: Optional[EntityType]
    level_order: int  # 0 = bottom level, higher numbers = higher levels


class Hierarchy:
    """Generic hierarchical structure that can represent any organisational hierarchy.

    This class removes any hardcoding of entities within a hierarchy, by using
    a generic approach that works with any hierarchical structure.

    Attributes
    ----------
    levels : Dict[EntityType, HierarchyLevel]
        Configuration of each level in the hierarchy
    relationships : Dict[str, str]
        Mapping from child_id to parent_id for all entities
    entity_types : Dict[str, EntityType]
        Mapping from entity_id to its EntityType

    Notes
    -----
    **Prefixed ID System:**

    Internally, the hierarchy uses prefixed IDs (e.g., ``"subspecialty:Gsurg LowGI"``)
    to ensure unique identification across entity types. This prevents name collisions
    when the same entity name appears at different levels (e.g., a subspecialty and
    a reporting unit both named "Acute Medicine").

    **When You See Prefixed IDs vs Original Names:**

    - **Internal storage**: All entity IDs are stored with prefixes (e.g.,
      ``"subspecialty:Gsurg LowGI"``, ``"division:Surgery Division"``)
    - **Public API**: Methods that return entity IDs use **original names** (without prefixes)
      for user convenience. For example:
      - `get_children()` returns ``['Gsurg LowGI', 'Gsurg UppGI']`` (original names)
      - `get_parent()` returns ``'Surgery Division'`` (original name)
      - `predict_all_levels()` returns results keyed by original names
    - **Input methods**: Methods that accept entity IDs accept **either** original names
      or prefixed IDs. The class automatically handles the conversion.

    **Why This Design:**

    - Prevents entity name collisions across different hierarchy levels
    - Allows flexible entity naming (same name can appear at different levels)
    - Keeps the public API clean (users work with original names)
    - Maintains internal consistency (unique IDs for all entities)

    **Handling Entity Name Collisions:**

    If you have entities with the same name at different levels (e.g., a subspecialty
    "Surgery Division" and a division "Surgery Division"), the prefixed ID system automatically
    handles this. When calling methods, you can use either:

    - Original name: ``hierarchy.get_children("Gastrointestinal Surgery")`` - the method will
      attempt to infer the entity type from context
    - Prefixed ID: ``hierarchy.get_children("reporting_unit:Gastrointestinal Surgery")`` - explicitly
      specifies the entity type

    If there's ambiguity (same name at multiple levels), prefer using prefixed IDs
    or providing the `entity_type` parameter when available.
    """

    def __init__(self, levels: List[HierarchyLevel]):
        self.levels = {level.entity_type: level for level in levels}
        self.relationships: Dict[str, str] = {}  # child_id -> parent_id
        self.entity_types: Dict[str, EntityType] = {}  # entity_id -> EntityType

        # Validate that levels form a proper hierarchy
        self._validate_levels()

    def _validate_levels(self):
        # Check that there's exactly one top level (no parent)
        top_levels = [
            level for level in self.levels.values() if level.parent_type is None
        ]
        if len(top_levels) != 1:
            raise ValueError("Hierarchy must have exactly one top level")

        # Check that all parent types exist
        for level in self.levels.values():
            if level.parent_type is not None and level.parent_type not in self.levels:
                raise ValueError(f"Parent type {level.parent_type} not found in levels")

    @classmethod
    def from_yaml(cls, config_path: str) -> "Hierarchy":
        """Create hierarchy from YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file

        Returns
        -------
        Hierarchy
            Configured hierarchy instance

        Examples
        --------
        >>> hierarchy = Hierarchy.from_yaml("hierarchy_config.yaml")

        Notes
        -----
        The YAML file defines **structure only** (entity types and their
        parent-child relationships), not the actual entity IDs. Entity IDs are
        populated later via ``populate_hierarchy_from_dataframe()``.

        The YAML must define a proper tree with exactly one top-level entity
        type (``parent_type: null``). Level order starts at 0 for the bottom
        level and increases upward. Entity type names are case-sensitive.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        levels = []
        for level_config in config["levels"]:
            entity_type = EntityType.from_string(level_config["entity_type"])
            parent_type = None
            if level_config.get("parent_type"):
                parent_type = EntityType.from_string(level_config["parent_type"])

            level = HierarchyLevel(
                entity_type, parent_type, level_config["level_order"]
            )
            levels.append(level)

        return cls(levels)

    @classmethod
    def create_default_hospital(cls) -> "Hierarchy":
        """Create default hospital hierarchy.

        Returns
        -------
        Hierarchy
            Default hospital hierarchy with standard levels
        """
        subspecialty = EntityType("subspecialty")
        reporting_unit = EntityType("reporting_unit")
        division = EntityType("division")
        board = EntityType("board")
        hospital = EntityType("hospital")

        levels = [
            HierarchyLevel(subspecialty, reporting_unit, 0),
            HierarchyLevel(reporting_unit, division, 1),
            HierarchyLevel(division, board, 2),
            HierarchyLevel(board, hospital, 3),
            HierarchyLevel(hospital, None, 4),
        ]
        return cls(levels)

    def add_entity(self, entity_id: str, entity_type: EntityType):
        """Add an entity to the hierarchy.

        Parameters
        ----------
        entity_id : str
            Original identifier for the entity
        entity_type : EntityType
            Type of the entity
        """
        if entity_type not in self.levels:
            raise ValueError(f"Unknown entity type: {entity_type}")

        # Create unique ID by prefixing with entity type
        unique_id = f"{entity_type.name}:{entity_id}"

        # Store entity with its type
        self.entity_types[unique_id] = entity_type

    def _find_entity_type_by_name(self, entity_name: str) -> Optional[EntityType]:
        for unique_id, entity_type in self.entity_types.items():
            # Extract original name from prefixed ID
            if ":" in unique_id:
                original_name = unique_id.split(":", 1)[1]
                if original_name == entity_name:
                    return entity_type
        return None

    def _get_original_name(self, unique_id: str) -> str:
        if ":" in unique_id:
            return unique_id.split(":", 1)[1]
        return unique_id

    def _get_prefixed_id(
        self, entity_name: str, entity_type: EntityType
    ) -> Optional[str]:
        prefixed_id = f"{entity_type.name}:{entity_name}"
        return prefixed_id if prefixed_id in self.entity_types else None

    def get_children(
        self, parent_id: str, parent_type: Optional[EntityType] = None
    ) -> List[str]:
        """Get all direct children of a parent entity.

        Parameters
        ----------
        parent_id : str
            Parent entity identifier (original name or prefixed ID)
        parent_type : EntityType, optional
            Entity type of the parent. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        List[str]
            List of child entity identifiers (original names)
        """
        # Convert parent_id to prefixed format if needed
        prefixed_parent_id = parent_id
        if ":" not in parent_id:
            if parent_type is not None:
                # Use the specified parent type
                prefixed_parent_id = f"{parent_type.name}:{parent_id}"
            else:
                # Try to find the prefixed version (may not work with entity collisions)
                parent_entity_type = self._find_entity_type_by_name(parent_id)
                if parent_entity_type is not None:
                    prefixed_parent_id = f"{parent_entity_type.name}:{parent_id}"

        children = []
        for child_id, pid in self.relationships.items():
            if pid == prefixed_parent_id:
                # Return original entity name
                children.append(self._get_original_name(child_id))
        return children

    def get_parent(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[str]:
        """Get the parent of an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier (original name or prefixed ID)
        entity_type : EntityType, optional
            Entity type of the entity. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        Optional[str]
            Parent entity identifier (original name), or None if entity is top-level
        """
        # Convert entity_id to prefixed format if needed
        prefixed_entity_id = entity_id
        if ":" not in entity_id:
            if entity_type is not None:
                # Use the specified entity type
                prefixed_entity_id = f"{entity_type.name}:{entity_id}"
            else:
                # Try to find the prefixed version (may not work with entity collisions)
                entity_entity_type = self._find_entity_type_by_name(entity_id)
                if entity_entity_type is not None:
                    prefixed_entity_id = f"{entity_entity_type.name}:{entity_id}"

        parent_id = self.relationships.get(prefixed_entity_id)
        if parent_id is not None:
            return self._get_original_name(parent_id)
        return None

    def get_entity_type(
        self, entity_id: str, entity_type: Optional[EntityType] = None
    ) -> Optional[EntityType]:
        """Get the type of an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier (original name or prefixed ID)
        entity_type : EntityType, optional
            Entity type of the entity. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        Optional[EntityType]
            Entity type, or None if entity not found
        """
        # Convert entity_id to prefixed format if needed
        prefixed_entity_id = entity_id
        if ":" not in entity_id:
            if entity_type is not None:
                # Use the specified entity type
                prefixed_entity_id = f"{entity_type.name}:{entity_id}"
            else:
                # Try to find the prefixed version (may not work with entity collisions)
                entity_entity_type = self._find_entity_type_by_name(entity_id)
                if entity_entity_type is not None:
                    prefixed_entity_id = f"{entity_entity_type.name}:{entity_id}"

        return self.entity_types.get(prefixed_entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[str]:
        """Get all entities of a specific type.

        Parameters
        ----------
        entity_type : EntityType
            Type of entities to retrieve

        Returns
        -------
        List[str]
            List of entity identifiers (original names) of the specified type
        """
        entities = []
        for entity_id, et in self.entity_types.items():
            if et == entity_type:
                # Return original entity name
                entities.append(self._get_original_name(entity_id))
        return entities

    def get_all_entities(self) -> List[str]:
        """Get all entity identifiers in the hierarchy.

        Returns
        -------
        List[str]
            List of all entity identifiers (original names)
        """
        return [
            self._get_original_name(entity_id) for entity_id in self.entity_types.keys()
        ]

    def get_levels_ordered(self) -> List[EntityType]:
        """Get all entity types ordered from bottom to top level.

        Returns
        -------
        List[EntityType]
            Entity types ordered from bottom to top
        """
        return sorted(self.levels.keys(), key=lambda et: self.levels[et].level_order)

    def get_entity_type_names(self) -> List[str]:
        """Get all entity type names in the hierarchy.

        Returns
        -------
        List[str]
            List of entity type names
        """
        return [et.name for et in self.levels.keys()]

    def get_entity_info(
        self, entity_name: str, entity_type: Optional[EntityType] = None
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information about an entity.

        Parameters
        ----------
        entity_name : str
            Original entity name
        entity_type : EntityType, optional
            Entity type of the entity. If not provided, will try to find it automatically.
            This is useful when there are multiple entities with the same name at different levels.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing entity information:
            - entity_id: original name
            - entity_type: EntityType
            - parent: parent entity name (if any)
            - children: list of child entity names
            - prefixed_id: internal prefixed ID
        """
        if entity_type is None:
            entity_type = self._find_entity_type_by_name(entity_name)
            if entity_type is None:
                return None

        prefixed_id = f"{entity_type.name}:{entity_name}"
        parent = self.get_parent(entity_name, entity_type)
        children = self.get_children(entity_name, entity_type)

        return {
            "entity_id": entity_name,
            "entity_type": entity_type,
            "parent": parent,
            "children": children,
            "prefixed_id": prefixed_id,
        }

    def get_bottom_level_type(self) -> EntityType:
        """Get the entity type at the bottom level of the hierarchy.

        Returns
        -------
        EntityType
            Bottom level entity type
        """
        bottom_level = min(self.levels.values(), key=lambda level: level.level_order)
        return bottom_level.entity_type

    def get_top_level_type(self) -> EntityType:
        """Get the entity type at the top level of the hierarchy.

        Returns
        -------
        EntityType
            Top level entity type
        """
        top_level = max(self.levels.values(), key=lambda level: level.level_order)
        return top_level.entity_type

    def __repr__(self) -> str:
        lines = []
        lines.append("Hierarchy:")
        for entity_type in self.get_levels_ordered():
            count = len(self.get_entities_by_type(entity_type))
            lines.append(f"  {entity_type.name}: {count}")
        return "\n".join(lines)


def populate_hierarchy_from_dataframe(
    hierarchy: Hierarchy,
    hierarchy_df: pd.DataFrame,
    column_mapping: Dict[str, str],
    top_level_id: str,
) -> None:
    """Populate a hierarchy with entities and relationships from a DataFrame.

    Each row of the DataFrame represents one path through the hierarchy from
    bottom to top. The ``column_mapping`` maps DataFrame column names to entity
    type names defined in the hierarchy.

    Parameters
    ----------
    hierarchy : Hierarchy
        Hierarchy instance with levels already defined (e.g. from
        ``Hierarchy.from_yaml()`` or ``Hierarchy.create_default_hospital()``).
    hierarchy_df : pandas.DataFrame
        Organisational structure where each row is one bottom-to-top path.
        Must form a proper tree (each child has exactly one parent).
    column_mapping : Dict[str, str]
        Mapping from DataFrame column names to entity type names.
        Only entity types with a mapping will be populated from the DataFrame.
    top_level_id : str
        Identifier for the single top-level entity. Created automatically
        if absent from the DataFrame; consolidates multiple top-level
        entries if present.

    Raises
    ------
    ValueError
        If required columns are missing, entity types are invalid, or parent
        entities cannot be found.

    Examples
    --------
    >>> import pandas as pd
    >>> hierarchy = Hierarchy.from_yaml("config.yaml")
    >>> hierarchy_df = pd.DataFrame({
    ...     'specialty_names': ['medical', 'surgical', 'haem/onc', 'paediatric'],
    ...     'division_names': ['Medical', 'Surgical', 'Medical', 'Specialist'],
    ... })
    >>> column_mapping = {
    ...     'specialty_names': 'specialty',
    ...     'division_names': 'division',
    ... }
    >>> populate_hierarchy_from_dataframe(
    ...     hierarchy, hierarchy_df, column_mapping, top_level_id="Hospital"
    ... )
    >>> hierarchy.get_children("Medical")
    ['haem/onc', 'medical']

    Notes
    -----
    The DataFrame must represent a tree structure. If the same child appears
    with different parents across rows, the last relationship processed
    overwrites earlier ones, which may silently produce an invalid hierarchy.

    Duplicate rows and rows with missing values are dropped before processing.
    Any entities left without a parent are automatically linked to the
    top-level entity.
    """
    # Validate that all required columns exist
    required_columns = list(column_mapping.keys())
    missing_columns = [
        col for col in required_columns if col not in hierarchy_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate that all mapped entity types exist in the hierarchy
    hierarchy_entity_types = set(hierarchy.get_entity_type_names())
    mapped_entity_types = set(column_mapping.values())
    invalid_mappings = mapped_entity_types - hierarchy_entity_types
    if invalid_mappings:
        raise ValueError(f"Invalid entity types in mapping: {invalid_mappings}")

    # Remove duplicates and any rows with missing values
    df = hierarchy_df.dropna().drop_duplicates()

    # Get hierarchy levels in order from bottom to top
    levels = hierarchy.get_levels_ordered()

    # PASS 1: Create all entities without relationships
    for i, entity_type in enumerate(levels):
        entity_type_name = entity_type.name

        # Find the column that maps to this entity type
        entity_column = None
        for col, et_name in column_mapping.items():
            if et_name == entity_type_name:
                entity_column = col
                break

        if entity_column is None:
            # If no column mapping found, skip this entity type
            # This happens for entity types that are created separately (like hospital)
            continue

        # Create all entities of this type without parents
        for _, row in df[[entity_column]].drop_duplicates().iterrows():
            entity_id = row[entity_column]
            hierarchy.add_entity(entity_id, entity_type)

    # PASS 2: Establish parent-child relationships
    for i, entity_type in enumerate(levels):
        entity_type_name = entity_type.name

        # Find the column that maps to this entity type
        entity_column = None
        for col, et_name in column_mapping.items():
            if et_name == entity_type_name:
                entity_column = col
                break

        if entity_column is None:
            continue

        # Skip top level (no parents)
        if i == len(levels) - 1:
            continue

        # Find the parent column for this level
        parent_type = levels[i + 1]
        parent_type_name = parent_type.name

        parent_column = None
        for col, et_name in column_mapping.items():
            if et_name == parent_type_name:
                parent_column = col
                break

        if parent_column is None:
            continue

        # Establish parent-child relationships with error checking
        df_subset = (
            df[[entity_column, parent_column]]
            .drop_duplicates()
            .sort_values([parent_column, entity_column])
        )
        for _, row in df_subset.iterrows():
            entity_id = row[entity_column]
            parent_id = row[parent_column]

            # Check if parent exists before trying to link
            parent_entity_type = hierarchy.get_entity_type(parent_id)
            if parent_entity_type is None:
                raise ValueError(
                    f"Parent entity '{parent_id}' not found for child '{entity_id}' of type '{entity_type_name}'"
                )

            # Allow same entity name at different levels (entity collision scenario)
            if parent_entity_type != parent_type:
                # Check if there's a parent with the correct type
                correct_parent_prefixed = f"{parent_type.name}:{parent_id}"
                if correct_parent_prefixed in hierarchy.entity_types:
                    # Use the correct parent
                    parent_prefixed = correct_parent_prefixed
                else:
                    raise ValueError(
                        f"Parent entity '{parent_id}' has type '{parent_entity_type.name}' but expected type '{parent_type.name}' for child '{entity_id}'"
                    )
            else:
                parent_prefixed = f"{parent_type.name}:{parent_id}"

            # Update the relationship
            child_prefixed = f"{entity_type.name}:{entity_id}"
            hierarchy.relationships[child_prefixed] = parent_prefixed

    # Link all entities to the top-level entity
    top_level_type = hierarchy.get_top_level_type()
    top_level_entities = hierarchy.get_entities_by_type(top_level_type)

    if not top_level_entities:
        # Create the top-level entity if it doesn't exist
        hierarchy.add_entity(top_level_id, top_level_type)
    else:
        # If multiple top-level entities exist, link them to the specified one
        for entity_id in top_level_entities:
            if entity_id != top_level_id:
                # Get prefixed IDs for both entities
                entity_prefixed = hierarchy._get_prefixed_id(entity_id, top_level_type)
                top_level_prefixed = hierarchy._get_prefixed_id(
                    top_level_id, top_level_type
                )

                if entity_prefixed and top_level_prefixed:
                    # Update the entity to have the specified top-level as parent
                    hierarchy.relationships[entity_prefixed] = top_level_prefixed

    # Link all entities that don't have parents to the top-level entity
    # This ensures the hierarchy is properly connected
    for entity_id, entity_type in hierarchy.entity_types.items():
        # Skip the top-level entity itself
        if entity_type == top_level_type:
            continue

        # Check if this entity already has a parent
        if entity_id not in hierarchy.relationships:
            # Link to the top-level entity
            top_level_prefixed = hierarchy._get_prefixed_id(
                top_level_id, top_level_type
            )
            if top_level_prefixed:
                hierarchy.relationships[entity_id] = top_level_prefixed
