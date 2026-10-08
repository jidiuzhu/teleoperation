#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rename_observations_processor")
class RenameObservationsProcessorStep(ObservationProcessorStep):
    """
    A processor step that renames keys in an observation dictionary.

    This step is useful for creating a standardized data interface by mapping keys
    from an environment's format to the format expected by a LeRobot policy or
    other downstream components.

    Attributes:
        rename_map: A dictionary mapping from old key names to new key names.
                    Keys present in an observation that are not in this map will
                    be kept with their original names.
    """

    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation):
        processed_obs = {}
        rename_targets = set(self.rename_map.values())

        # Keep non-renamed keys, but drop keys that are explicit rename targets.
        # This lets user-selected keys (e.g. action.wrist_z_gesture -> action)
        # override an existing canonical key instead of being overwritten by it.
        for key, value in observation.items():
            if key in self.rename_map or key in rename_targets:
                continue
            processed_obs[key] = value

        for source_key, target_key in self.rename_map.items():
            if source_key in observation:
                processed_obs[target_key] = observation[source_key]

        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transforms:
        - Each key in the observation that appears in `rename_map` is renamed to its value.
        - Keys not in `rename_map` remain unchanged.
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {
            self.rename_map.get(k, k): v for k, v in features[PipelineFeatureType.OBSERVATION].items()
        }
        return new_features


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    Renames the top-level keys in a statistics dictionary using a provided mapping.

    This is a helper function typically used to keep normalization statistics
    consistent with renamed observation or action features. It performs a defensive
    deep copy to avoid modifying the original `stats` dictionary.

    Args:
        stats: A nested dictionary of statistics, where top-level keys are
               feature names (e.g., `{"observation.state": {"mean": 0.5}}`).
        rename_map: A dictionary mapping old feature names to new feature names.

    Returns:
        A new statistics dictionary with its top-level keys renamed. Returns an
        empty dictionary if the input `stats` is empty.
    """
    if not stats:
        return {}
    renamed: dict[str, dict[str, Any]] = {}
    rename_targets = set(rename_map.values())

    for old_key, sub_stats in stats.items():
        if old_key in rename_map or old_key in rename_targets:
            continue
        renamed[old_key] = deepcopy(sub_stats) if sub_stats is not None else {}

    for old_key, new_key in rename_map.items():
        if old_key in stats:
            sub_stats = stats[old_key]
            renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}

    return renamed
