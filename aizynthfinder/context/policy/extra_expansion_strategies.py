"""Module containing classes that implements different expansion policy strategies"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aizynthfinder.chem import SmilesBasedRetroReaction, TemplatedRetroReaction
from aizynthfinder.context.policy.utils import _make_fingerprint
from aizynthfinder.utils.exceptions import PolicyException
from aizynthfinder.utils.logging import logger
from aizynthfinder.utils.models import load_model
from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy

from external_modelzoo.ssbenchmark.model_zoo import ModelZoo
# import line_profiler
if TYPE_CHECKING:
    from aizynthfinder.chem import TreeMolecule
    from aizynthfinder.chem.reaction import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.utils.type_utils import (
        Any,
        Dict,
        List,
        Optional,
        Sequence,
        StrDict,
        Tuple,
    )


class GeneralSmilesBasedModel(abc.ABC):
    """

    """

    def __init__(self, model):
        # in case you need to initialize the model, overrite the init and call super()
        # model.model_setup(args)
        self.model = model

    def predict(self, mol: TreeMolecule):       
        # model call requires a list of smiles
        if isinstance(mol, list):
            reactants, priors = self.model.model_call([one_mol.smiles for one_mol in mol])
        else:
            reactants, priors = self.model.model_call([mol.smiles])
        # the model returns a nested list, flatten it
        # breakpoint()
        if isinstance(priors, list):
            return reactants, priors
        
        reactants = np.array(reactants).flatten()
        priors = np.array(priors).flatten()
        return reactants, priors


class ModelZooExpansionStrategy(ExpansionStrategy):
    """
    An expansion strategy that uses the singleStepBaseModel to operate on a Smiles-level of abstraction, where only
    the product smiles string is used

    :param key: the key or label
    :param config: the configuration of the tree search
    :param source: the source of the policy model
    :raises PolicyException: 
    """

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)
        gpu_mode = kwargs.pop('gpu_mode', False)
        single_step_model_kwargs = kwargs["data"]
        
        self._logger.info(
            f"Processing model data: {single_step_model_kwargs} to {self.key}"
        )

        modelImplementation = ModelZoo(key = key, use_gpu = gpu_mode, **single_step_model_kwargs)

        self.model: GeneralSmilesBasedModel = GeneralSmilesBasedModel(modelImplementation)

    # @line_profiler.profile
    def get_actions(self, molecules: Sequence[TreeMolecule], cache_molecules: Optional[Sequence[TreeMolecule]] = None
                    ) -> Tuple[List[RetroReaction], List[float]]:
        batched_possible_actions = []

        predicted_reactants_list, predicted_probs_list = self.model.predict(molecules) # smiles, probabilities
        for i in range(len(molecules)):
            possible_actions = []
            mol = molecules[i]
            predicted_reactants = predicted_reactants_list[i]  # -> # list of length K
            predicted_probs = predicted_probs_list[i]  # -> # np_array float of len K

            assert len(predicted_reactants) == len(predicted_probs)

            for idx, move in enumerate(predicted_reactants):
                metadata = dict()
                metadata["reaction"] = move
                metadata["policy_probability"] = float(predicted_probs[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key

                smilesReaction = SmilesBasedRetroReaction(mol, reactants_str=move, metadata=metadata)
                possible_actions.append(smilesReaction)
            batched_possible_actions.append(possible_actions)
        return batched_possible_actions, predicted_probs_list  # List of len B of list of K SmilesBasedRetroReactions; list of B np_arrays(float32) of length K

class SMILEStoSMILESExpansionStrategy(ExpansionStrategy):
    """
    A template-free expansion strategy that will return `SmilesBasedRetroReaction` objects.

    :param key: the key or label
    :param config: the configuration of the tree search
    :raises PolicyException
    """
    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)
        self.model = None # setup

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        possible_actions = []
        possible_actions_priors = [] # Log likelihoods

        for mol in molecules:

            predicted_reactants, predicted_priors = self.model.predict(mol)

            assert len(predicted_reactants) == len(predicted_priors)

            probable_transforms_idx = self._cutoff_predictions(predicted_priors)
            possible_moves = predicted_reactants[probable_transforms_idx]
            possible_moves_probabilities = predicted_priors[probable_transforms_idx]

            possible_actions_priors.extend(possible_moves_probabilities)
            for idx, move in enumerate(possible_moves):
                metadata = dict()
                metadata["reaction"] = move
                metadata["policy_probability"] = float(possible_moves_probabilities[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key

                # add the SmilesBasedRetroReaction
                smilesReaction = SmilesBasedRetroReaction(mol, reactants_str=move, metadata=metadata)
                possible_actions.append(smilesReaction)

        return possible_actions, possible_actions_priors
