# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
from inspect import signature
from random import getrandbits
from typing import Union, Optional
from typing_extensions import get_args, get_origin

from sagemaker import Model, PipelineModel
from sagemaker.estimator import EstimatorBase, Estimator
from sagemaker.processing import Processor
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline import Pipeline
from tests.unit.sagemaker.workflow.test_mechanism import (
    EXPECTED_ERRORS_FOR_BONDED_PARAMS,
    STEP_CLASS,
    FIXED_ARGUMENTS,
    STR_VAL,
    PARAM_BONDS,
    PIPELINE_VARIABLES,
    NONE_PARAMS_TO_SKIP,
    PIPELINE_SESSION,
)


class PipelineVarCompatiTestTemplate:
    """Check the compatibility between Pipeline variables and the given class, target method"""

    def __init__(self, clazz: type, default_args: dict):
        """Initialize a `PipelineVarCompatiTestTemplate` instance.

        Args:
            clazz (type): The class to test the compatibility.
            default_args (dict): The given default arguments for the class and its target method.
        """
        self._clazz = clazz
        self._clazz_type = self._get_clazz_type()
        self._default_args = default_args
        self._target_func = self._target_class_function()
        self._clz_params = _get_param_dict(clazz.__init__, clazz)
        self._func_params = _get_param_dict(self._target_func)

    def _get_clazz_type(self) -> str:
        """Get the type (in str) of the downstream class"""
        if issubclass(self._clazz, Processor):
            return "processor"
        if issubclass(self._clazz, EstimatorBase):
            return "estimator"
        if issubclass(self._clazz, Transformer):
            return "transformer"
        if issubclass(self._clazz, HyperparameterTuner):
            return "tuner"
        if issubclass(self._clazz, (Model, PipelineModel)):
            return "model"
        raise TypeError(f"Unsupported downstream class: {self._clazz}")

    def check_compatibility(self):
        """The entry to check the compatibility"""

        print(
            "Starting to check Pipeline variable compatibility for class (%s) and target method (%s)\n"
            % (self._clazz.__name__, self._target_func.__name__)
        )
        self._check_or_fill_in_args(
            params={**self._clz_params["required"], **self._clz_params["optional"]},
            default_args=self._default_args["clazz_args"],
        )
        self._check_or_fill_in_args(
            params={**self._func_params["required"], **self._func_params["optional"]},
            default_args=self._default_args["func_args"],
        )

        # Check the case when all args are assigned not-None values
        print("### Starting to check the compatibility when all optional args are not None ###")
        self._iterate_params_to_check_compatibility()

        # Check the case when one of the optional arg is None
        print(
            "### Starting to check the compatibility when one of the optional arg is None in each round ###"
        )
        self._iterate_optional_params_to_check_compatibility()

    def _iterate_params_to_check_compatibility(self, param_with_none: Optional[str] = None):
        """Iterate each parameter and assign a pipeline var to it to test compatibility

        Args:
            param_with_none (str): The name of the parameter with None value.
        """
        self._iterate_clz_params_to_check_compatibility(param_with_none)
        self._iterate_func_params_to_check_compatibility(param_with_none)

    def _iterate_optional_params_to_check_compatibility(self):
        """Iterate each optional parameter and set it to none to test compatibility"""
        self._iterate_optional_params(
            optional_params=self._clz_params["optional"],
            default_args=self._default_args["clazz_args"],
        )
        self._iterate_optional_params(
            optional_params=self._func_params["optional"],
            default_args=self._default_args["func_args"],
        )

    def _iterate_optional_params(
        self,
        optional_params: dict,
        default_args: dict,
    ):
        """Helper function to iterate each parameter and check compatibility

        Args:
            optional_params (dict): The dict containing the optional parameters of a class or method.
            default_args (dict): The dict containing the default arguments of a class or method.
        """
        for param_name in optional_params.keys():
            if param_name in NONE_PARAMS_TO_SKIP[self._clazz_type]:
                continue
            origin_val = default_args[param_name]
            default_args[param_name] = None
            print("=== Parameter (%s) is None in this round ===" % param_name)
            self._iterate_params_to_check_compatibility(param_name)
            default_args[param_name] = origin_val

    def _iterate_clz_params_to_check_compatibility(self, param_with_none: Optional[str] = None):
        """Iterate each class parameter and assign a pipeline var to it to test compatibility

        Args:
            param_with_none (str): The name of the parameter with None value.
        """
        clz_params = {**self._clz_params["required"], **self._clz_params["optional"]}
        # Iterate through each default arg
        for clz_param_name, clz_default_arg in self._default_args["clazz_args"].items():
            if clz_param_name == param_with_none:
                continue
            if _are_bonded_params(param_with_none, clz_param_name):
                # No need to check the case: if a param is none,
                # the clz param does not show up in definition json consequently
                continue
            clz_param_type = clz_params[clz_param_name]["type"]
            if not _does_support_pipeline_variable(clz_param_type):
                continue

            # For each arg which supports pipeline variables,
            # Replace it with each one of generated pipeline variables
            ppl_vars = _generate_pipeline_vars_per_type(clz_param_name, clz_param_type)
            for clz_ppl_var, expected_clz_expr in ppl_vars:
                # print(
                #     'Replacing class arg (%s) with pipeline variable which is expected to be (%s)'
                #     % (clz_param_name, expected_clz_expr)
                # )
                self._default_args["clazz_args"][clz_param_name] = clz_ppl_var

                obj = None
                try:
                    obj = self._clazz(**self._default_args["clazz_args"])
                    self._generate_and_verify_pipeline_definition(
                        target_func=getattr(obj, self._target_func.__name__),
                        expected_expr=expected_clz_expr,
                        param_with_none=param_with_none,
                    )
                except ValueError as val_err:
                    if not param_with_none:
                        raise val_err
                    exp_val_errs = EXPECTED_ERRORS_FOR_BONDED_PARAMS["ValueError"]
                    if param_with_none in exp_val_errs and exp_val_errs[param_with_none][
                        clz_param_name
                    ] in str(val_err):
                        continue
                    raise val_err

            # print("============================\n")
            self._default_args["clazz_args"][clz_param_name] = clz_default_arg

    def _iterate_func_params_to_check_compatibility(self, param_with_none: Optional[str] = None):
        """Iterate each target func parameter and assign a pipeline var to it

        Args:
            param_with_none (str): The name of the parameter with None value.
        """
        obj = self._clazz(**self._default_args["clazz_args"])

        func_params = {**self._func_params["required"], **self._func_params["optional"]}
        for func_param_name, func_default_arg in self._default_args["func_args"].items():
            if func_param_name == param_with_none:
                continue
            if _are_bonded_params(param_with_none, func_param_name):
                # No need to check the case: if a param is none,
                # the clz param will not show up in definition json consequently
                continue
            func_param_type = func_params[func_param_name]["type"]
            if not _does_support_pipeline_variable(func_param_type):
                continue

            # For each arg which supports pipeline variables,
            # Replace it with each one of generated pipeline variables
            ppl_vars = _generate_pipeline_vars_per_type(
                func_param_name, func_params[func_param_name]["type"]
            )
            for func_ppl_var, expected_func_expr in ppl_vars:
                # print(
                #     'Replacing func arg (%s) with pipeline variable which is expected to be (%s)'
                #     % (func_param_name, expected_func_expr)
                # )
                self._default_args["func_args"][func_param_name] = func_ppl_var
                target_func = getattr(obj, self._target_func.__name__)
                self._generate_and_verify_pipeline_definition(
                    target_func=target_func,
                    expected_expr=expected_func_expr,
                    param_with_none=param_with_none,
                )

            self._default_args["func_args"][func_param_name] = func_default_arg
            # print("-------------------------\n")

    def _generate_and_verify_pipeline_definition(
        self,
        target_func,
        expected_expr: dict,
        param_with_none: str,
    ):
        """Generate a pipeline and verify the pipeline definition

        Args:
            target_func (function): The function to generate step_args.
            expected_expr (dict): The expected json expression of a class or method argument.
            param_with_none (str): The name of the parameter with None value.
        """
        args = dict(
            name="MyStep",
            step_args=target_func(**self._default_args["func_args"]),
        )
        step = STEP_CLASS[self._clazz_type](**args)
        pipeline = Pipeline(
            name="MyPipeline",
            steps=[step],
            sagemaker_session=PIPELINE_SESSION,
        )
        step_dsl = json.dumps(json.loads(pipeline.definition())["Steps"][0])
        exp_origin = json.dumps(expected_expr["origin"])
        exp_to_str = json.dumps(expected_expr["to_string"])
        # if the testing arg is a dict, we may need to remove the outer {} of its expected expr
        # to compare, since for HyperParameters, some other arguments are auto inserted to the dict
        assert (
            exp_origin in step_dsl
            or exp_to_str in step_dsl
            or exp_origin[1:-1] in step_dsl
            or exp_to_str[1:-1] in step_dsl
        )

        # TODO: remove the following hard code assertion once recursive assignment is added
        if issubclass(self._clazz, Processor):
            if param_with_none != "network_config":
                assert json.dumps({"Get": "Parameters.nw_cfg_subnets"}) in step_dsl
                assert json.dumps({"Get": "Parameters.nw_cfg_security_group_ids"}) in step_dsl
                assert json.dumps({"Get": "Parameters.nw_cfg_enable_nw_isolation"}) in step_dsl
            if param_with_none != "outputs":
                assert json.dumps({"Get": "Parameters.proc_output_source"}) in step_dsl
                assert json.dumps({"Get": "Parameters.proc_output_dest"}) in step_dsl
                assert json.dumps({"Get": "Parameters.proc_output_app_managed"}) in step_dsl
            if param_with_none != "inputs":
                assert json.dumps({"Get": "Parameters.proc_input_source"}) in step_dsl
                assert json.dumps({"Get": "Parameters.proc_input_dest"}) in step_dsl
                assert json.dumps({"Get": "Parameters.proc_input_s3_data_type"}) in step_dsl
                assert json.dumps({"Get": "Parameters.proc_input_app_managed"}) in step_dsl
        if issubclass(self._clazz, EstimatorBase):
            if isinstance(self._clazz, Estimator):
                if param_with_none != "debugger_hook_config":
                    assert json.dumps({"Get": "Parameters.debugger_hook_s3_output"}) in step_dsl
                if param_with_none != "profiler_config":
                    assert (
                        json.dumps({"Get": "Parameters.profile_config_system_monitor"}) in step_dsl
                    )
            if param_with_none != "tensorboard_output_config":
                assert json.dumps({"Get": "Parameters.tensorboard_s3_output"}) in step_dsl
            if param_with_none != "inputs":
                assert json.dumps({"Get": "Parameters.train_inputs_s3_data"}) in step_dsl
                assert json.dumps({"Get": "Parameters.train_inputs_content_type"}) in step_dsl
            if param_with_none != "rules":
                assert json.dumps({"Get": "Parameters.rules_instance_type"}) in step_dsl

    def _get_non_pipeline_val(self, n: str, t: type) -> object:
        """Get the value (not a Pipeline variable) based on parameter type and name

        Args:
            n (str): The parameter name. If a parameter has a pre-defined value,
                it will be returned directly.
            t (type): The parameter type. If a parameter does not have a pre-defined value,
                an arg will be auto-generated based on the type.

        Return:
            object: A Python primitive value is returned.
        """
        if n in FIXED_ARGUMENTS["common"]:
            return FIXED_ARGUMENTS["common"][n]
        if n in FIXED_ARGUMENTS[self._clazz_type]:
            return FIXED_ARGUMENTS[self._clazz_type][n]
        if t is str:
            return STR_VAL
        if t is int:
            return 1
        if t is float:
            return 1.5
        if t is bool:
            return bool(getrandbits(1))
        if t in [list, tuple, dict, set]:
            return t()

        raise TypeError(f"Unable to parse type: {t}.")

    def _check_or_fill_in_args(self, params: dict, default_args: dict):
        """Check if every args are provided and not None

        Otherwise fill in with some default values

        Args:
            params (dict): The dict indicating the type of each parameter.
            default_args (dict): The dict of args to be checked or filled in.
        """
        for param_name, value in params.items():
            if param_name in default_args and default_args[param_name] is not None:
                continue
            clean_type = _clean_up_types(value["type"])
            origin_type = get_origin(clean_type)
            if origin_type is None:
                default_args[param_name] = self._get_non_pipeline_val(param_name, clean_type)
            else:
                default_args[param_name] = self._get_non_pipeline_val(param_name, origin_type)

        self._check_or_update_default_args(default_args)

    def _check_or_update_default_args(self, default_args: dict):
        """To check if the default args are valid and update them if not

        Args:
            default_args (dict): The dict of args to be checked or updated.
        """
        if issubclass(self._clazz, EstimatorBase):
            if "disable_profiler" in default_args and default_args["disable_profiler"] is True:
                default_args["profiler_config"] = None

    def _target_class_function(self) -> object:
        """Fetch the target function based on class

        Return:
            object: The target function is returned.
        """
        # TODO: update to auto detect method annotated with @runnable_by_pipeline
        # https://stackoverflow.com/questions/5707589/calling-functions-by-array-index-in-python/5707605#5707605
        # https://stackoverflow.com/questions/3232024/introspection-to-get-decorator-names-on-a-method
        if issubclass(self._clazz, Processor):
            return self._clazz.run
        if issubclass(self._clazz, EstimatorBase):
            return self._clazz.fit
        raise TypeError(f"Unable to get target function for class {self._clazz}")


def _are_bonded_params(param_with_none: Optional[str], param_name: str) -> bool:
    """Check if the two params are bonded.

    I.e. if one of the parameter (param_with_none) is None,
    the other (param_name) does not present in the definition json

    Args:
        param_with_none (str): The name of the parameter with None value.
        param_name (str): The name of the other parameter, which is to be checked for
            the bond relationship with param_with_none

    Return:
        bool: True if the two parameters are bonded. False otherwise.
    """
    return (
        param_with_none
        and (param_with_none in PARAM_BONDS)
        and (param_name in PARAM_BONDS[param_with_none])
    )


def _does_support_pipeline_variable(t: type) -> bool:
    """Check if pipeline variable is supported by a parameter according to its type

    Args:
        t (type): The type to be checked

    Return:
        bool: True if it supports. False otherwise.
    """
    return "PipelineVariable" in str(t)


def _get_param_dict(func, clazz=None) -> dict:
    """Get a parameter dict of a given function

    The parameter dict indicates if a parameter is required or not, as well as its type.

    Arg:
        func (function): A class constructor method or other class methods.
        clazz (type): The corresponding class whose method is passed in.

    Return:
        dict: A parameter dict is returned.
    """
    params = list()
    params.append(signature(func))
    if func.__name__ == "__init__" and issubclass(clazz, (EstimatorBase, Model)):
        # Go through all parent classes constructor function to get the entire parameters since
        # estimator and model classes use **kwargs for parameters defined in parent classes
        # The leaf class's parameters should be on top of the params list and have high priority
        _get_params_from_parent_class_constructors(clazz, params)

    params_dict = dict(
        required=dict(),
        optional=dict(),
    )
    for param in params:
        for param_val in param.parameters.values():
            if param_val.annotation is param_val.empty:
                continue
            val = dict(type=param_val.annotation)
            if param_val.name == "sagemaker_session":
                # Treat sagemaker_session as required as it must be a PipelineSession obj
                if not _is_in_params_dict(param_val.name, params_dict):
                    # Some parameters e.g. entry_point in TensorFlow appears as both required (in Framework)
                    # and optional (in EstimatorBase) parameter. The annotation defined in the
                    # class node (i.e. Framework) which is closer to the leaf class (TensorFlow) should win.
                    params_dict["required"][param_val.name] = val
            elif param_val.default is param_val.empty:
                if not _is_in_params_dict(param_val.name, params_dict):
                    params_dict["required"][param_val.name] = val
            else:
                if not _is_in_params_dict(param_val.name, params_dict):
                    params_dict["optional"][param_val.name] = val
    return params_dict


def _is_in_params_dict(param_name: str, params_dict: dict):
    """To check if the parameter is in the parameter dict

    Args:
        param_name (str): The name of the parameter to be checked
        params_dict (dict): The parameter dict among which to check if the param_name exists
    """
    return param_name in params_dict["required"] or param_name in params_dict["optional"]


def _get_params_from_parent_class_constructors(clazz: type, params: list):
    """Get constructor parameters from parent class

    Args:
        clazz (type): The downstream class to collect parameters from all its parent constructors
        params (list): The list to collect all parameters
    """
    while clazz.__name__ not in {"EstimatorBase", "Model"}:
        parent_class = clazz.__base__
        params.append(signature(parent_class.__init__))
        clazz = parent_class


def _generate_pipeline_vars_per_type(
    param_name: str,
    param_type: type,
) -> list:
    """Provide a list of possible PipelineVariable objects.

    For example, if type_hint is Union[str, PipelineVariable],
    return [ParameterString, Properties, JsonGet, Join, ExecutionVariable]

    Args:
        param_name (str): The name of the parameter to generate the pipeline variable list.
        param_type (type): The type of the parameter to generate the pipeline variable list.

    Return:
        list: A list of possible PipelineVariable objects are returned.
    """
    # verify if params allow pipeline variables
    if "PipelineVariable" not in str(param_type):
        raise TypeError(("The type: %s does not support PipelineVariable.", param_type))

    types = get_args(param_type)
    # e.g. Union[str, PipelineVariable] or Union[str, PipelineVariable, NoneType]
    if PipelineVariable in types:
        # PipelineVariable corresponds to Python Primitive types
        # i.e. str, int, float, bool
        ppl_var = _get_pipeline_var(types=types)
        return ppl_var

    # e.g. Union[List[...], NoneType] or Union[Dict[...], NoneType] etc.
    clean_type = _clean_up_types(param_type)
    origin_type = get_origin(clean_type)
    if origin_type not in [list, dict, set, tuple]:
        raise TypeError(f"Unsupported type: {param_type} for param: {param_name}")
    sub_types = get_args(clean_type)

    # e.g. List[...], Tuple[...], Set[...]
    if origin_type in [list, tuple, set]:
        ppl_var_list = _generate_pipeline_vars_per_type(param_name, sub_types[0])
        return [
            (
                origin_type([var]),
                dict(
                    origin=origin_type([expected["origin"]]),
                    to_string=origin_type([expected["to_string"]]),
                ),
            )
            for var, expected in ppl_var_list
        ]

    # e.g. Dict[...]
    if origin_type is dict:
        key_type = sub_types[0]
        if key_type is not str:
            raise TypeError(
                f"Unsupported type: {key_type} for dict key in {param_name} of {param_type} type"
            )
        ppl_var_list = _generate_pipeline_vars_per_type(param_name, sub_types[1])
        return [
            (
                dict(MyKey=var),
                dict(
                    origin=dict(MyKey=expected["origin"]),
                    to_string=dict(MyKey=expected["to_string"]),
                ),
            )
            for var, expected in ppl_var_list
        ]
    return list()


def _clean_up_types(t: type) -> type:
    """Clean up the Union type and return the first subtype (not a NoneType) of it

    For example for Union[str, int, NoneType], it will return str

    Args:
        t (type): The type of a parameter to be cleaned up.

    Return:
        type: The cleaned up type is returned.
    """
    if get_origin(t) == Union:
        types = get_args(t)
        return list(filter(lambda t: "NoneType" not in str(t), types))[0]
    return t


def _get_pipeline_var(types: tuple) -> list:
    """Get a Pipeline variable based on one kind of the parameter types.

    Args:
        types (tuple): The possible types of a parameter.

    Return:
        list: a list of possible PipelineVariable objects are returned
    """
    if str in types:
        return PIPELINE_VARIABLES["str"]
    if int in types:
        return PIPELINE_VARIABLES["int"]
    if float in types:
        return PIPELINE_VARIABLES["float"]
    if bool in types:
        return PIPELINE_VARIABLES["bool"]
    raise TypeError(f"Unable to parse types: {types}.")
