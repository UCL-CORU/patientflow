"""Hierarchical structure definitions."""

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
    """Generic hierarchical structure that can represent any organizational hierarchy.

    This class eliminates the need for specific methods for each entity type by using
    a generic approach that works with any hierarchical structure.

    Attributes
    ----------
    levels : Dict[EntityType, HierarchyLevel]
        Configuration of each level in the hierarchy
    relationships : Dict[str, str]
        Mapping from child_id to parent_id for all entities
    entity_types : Dict[str, EntityType]
        Mapping from entity_id to its EntityType
    """

    def __init__(self, levels: List[HierarchyLevel]):
        self.levels = {level.entity_type: level for level in levels}
        self.relationships: Dict[str, str] = {}  # child_id -> parent_id
        self.entity_types: Dict[str, EntityType] = {}  # entity_id -> EntityType

        # Validate that levels form a proper hierarchy
        self._validate_levels()

    def _validate_levels(self):
        """Validate that the hierarchy levels are properly configured."""
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
        """Add an entity to the hierarchy with type-prefixed unique ID.

        This method is used in Pass 1 to create entities without relationships.
        Relationships are established separately in Pass 2.

        Parameters
        ----------
        entity_id : str
            Original identifier for the entity (will be prefixed with entity type)
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
        """Find the entity type for a given entity name by searching through existing entities.

        Parameters
        ----------
        entity_name : str
            Original entity name to search for

        Returns
        -------
        Optional[EntityType]
            Entity type if found, None otherwise
        """
        for unique_id, entity_type in self.entity_types.items():
            # Extract original name from prefixed ID
            if ":" in unique_id:
                original_name = unique_id.split(":", 1)[1]
                if original_name == entity_name:
                    return entity_type
        return None

    def _get_original_name(self, unique_id: str) -> str:
        """Extract original entity name from prefixed ID.

        Parameters
        ----------
        unique_id : str
            Prefixed entity ID (e.g., "subspecialty:Cardiology")

        Returns
        -------
        str
            Original entity name (e.g., "Cardiology")
        """
        if ":" in unique_id:
            return unique_id.split(":", 1)[1]
        return unique_id

    def _get_prefixed_id(
        self, entity_name: str, entity_type: EntityType
    ) -> Optional[str]:
        """Get prefixed ID for an entity name and type.

        Parameters
        ----------
        entity_name : str
            Original entity name
        entity_type : EntityType
            Entity type

        Returns
        -------
        Optional[str]
            Prefixed ID if entity exists, None otherwise
        """
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
    """Populate hierarchy from a pandas DataFrame with explicit column mapping.

    This function extracts the organizational hierarchy from a DataFrame using
    an explicit mapping of DataFrame columns to entity type names. It works
    with any hierarchy structure defined in the YAML configuration.

    Parameters
    ----------
    hierarchy : Hierarchy
        Hierarchy instance to populate
    hierarchy_df : pandas.DataFrame
        DataFrame containing organizational structure
    column_mapping : Dict[str, str]
        Mapping from DataFrame column names to entity type names.
        Example: {'employee_id': 'employee', 'team_id': 'team', 'dept_id': 'department'}
    top_level_id : str
        Identifier for the top-level entity in the hierarchy

    Raises
    ------
    ValueError
        If required columns are missing or entity types don't match hierarchy

    Notes
    -----
    The function:

    1. Validates that all required columns exist in the DataFrame
    2. Validates that all mapped entity types exist in the hierarchy
    3. Removes duplicate rows and rows with missing values
    4. Establishes parent-child relationships based on the hierarchy structure
    5. Links all entities to the specified top-level entity

    Duplicate relationships are automatically handled.
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
