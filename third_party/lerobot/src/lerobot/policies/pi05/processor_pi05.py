#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="pi05_concat_state_features_processor_step")
@dataclass
class Pi05ConcatStateFeaturesProcessorStep(ProcessorStep):
    """
    Concatenate multiple normalized STATE features into `observation.state`.

    PI0.5 expects a single state vector for prompt construction, but datasets may
    expose several STATE features (for example `observation.state` together with
    `observation.distances`). This step merges them in a deterministic order after
    normalization and before state tokenization.
    """

    state_keys: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION, {})
        if observation is None:
            raise ValueError("Observation is required for PI05")

        tensors: list[torch.Tensor] = []
        reference_shape: tuple[int, ...] | None = None
        for key in self.state_keys:
            if key not in observation:
                raise ValueError(f"Missing state key '{key}' required by PI05 state_keys={self.state_keys}")
            tensor = torch.as_tensor(observation[key])
            if reference_shape is None:
                reference_shape = tuple(tensor.shape[:-1])
            elif tuple(tensor.shape[:-1]) != reference_shape:
                raise ValueError(
                    f"State key '{key}' has incompatible leading shape {tuple(tensor.shape[:-1])}; "
                    f"expected {reference_shape}"
                )
            tensors.append(tensor)

        if not tensors:
            raise ValueError("PI05 state_keys must not be empty")

        observation = observation.copy()
        observation[OBS_STATE] = torch.cat(tensors, dim=-1)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        obs_features = transformed.get(PipelineFeatureType.OBSERVATION, {})

        total_dim = 0
        for key in self.state_keys:
            feature = obs_features.get(key)
            if feature is None:
                continue
            if feature.type is not FeatureType.STATE:
                raise ValueError(f"PI05 state key '{key}' must be a STATE feature, got {feature.type}")
            if len(feature.shape) != 1:
                raise ValueError(
                    f"PI05 state key '{key}' must be 1D before concatenation, got shape {feature.shape}"
                )
            total_dim += feature.shape[0]

        if total_dim > 0:
            obs_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(total_dim,))
            transformed[PipelineFeatureType.OBSERVATION] = obs_features

        return transformed


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    rename_map: dict[str, str] = {}
    if config.use_expert_point_cloud_condition and config.point_cloud_source_key is not None:
        rename_map[config.point_cloud_source_key] = config.point_cloud_key

    normalized_dataset_stats = rename_stats(dataset_stats, rename_map) if dataset_stats is not None else None

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map=rename_map),
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=normalized_dataset_stats,
        ),
        Pi05ConcatStateFeaturesProcessorStep(state_keys=config.state_keys),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=normalized_dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
