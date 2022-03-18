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

import sagemaker
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.workflow import Properties
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep

_STR_VAL = "MyString"
_KEY_VAL = "MyKey"


def _generate_all_pipeline_vars() -> dict:
    """Generate a dic with all kinds of Pipeline variables"""
    # Parameter
    ppl_param_str = ParameterString(name="MyString")
    ppl_param_int = ParameterInteger(name="MyInt")
    ppl_param_float = ParameterFloat(name="MyFloat")
    ppl_param_bool = ParameterBoolean(name="MyBool")

    # Function
    ppl_join = Join(on=" ", values=[ppl_param_int, ppl_param_float])
    property_file = PropertyFile(
        name="name",
        output_name="result",
        path="output",
    )
    ppl_json_get = JsonGet(
        step_name="my-step",
        property_file=property_file,
        json_path="my-json-path",
    )

    # Properties
    ppl_prop = Properties("Steps.MyStep", "DescribeTrainingJobResponse")
    ppl_prop_str = ppl_prop.OutputDataConfig.S3OutputPath
    ppl_prop_int = ppl_prop.TrainingTimeInSeconds
    ppl_prop_float = ppl_prop.FinalMetricDataList[0].Value
    ppl_prop_bool = ppl_prop.EnableManagedSpotTraining

    # Execution Variables
    ppl_exe_var = ExecutionVariables.PIPELINE_NAME

    return dict(
        str=[
            (
                ppl_param_str,
                dict(origin=ppl_param_str.expr, to_string=ppl_param_str.to_string().expr),
            ),
            (ppl_join, dict(origin=ppl_join.expr, to_string=ppl_join.to_string().expr)),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (ppl_prop_str, dict(origin=ppl_prop_str.expr, to_string=ppl_prop_str.to_string().expr)),
            (ppl_exe_var, dict(origin=ppl_exe_var.expr, to_string=ppl_exe_var.to_string().expr)),
        ],
        int=[
            (
                ppl_param_int,
                dict(origin=ppl_param_int.expr, to_string=ppl_param_int.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (ppl_prop_int, dict(origin=ppl_prop_int.expr, to_string=ppl_prop_int.to_string().expr)),
        ],
        float=[
            (
                ppl_param_float,
                dict(origin=ppl_param_float.expr, to_string=ppl_param_float.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (
                ppl_prop_float,
                dict(origin=ppl_prop_float.expr, to_string=ppl_prop_float.to_string().expr),
            ),
        ],
        bool=[
            (
                ppl_param_bool,
                dict(origin=ppl_param_bool.expr, to_string=ppl_param_bool.to_string().expr),
            ),
            (ppl_json_get, dict(origin=ppl_json_get.expr, to_string=ppl_json_get.to_string().expr)),
            (
                ppl_prop_bool,
                dict(origin=ppl_prop_bool.expr, to_string=ppl_prop_bool.to_string().expr),
            ),
        ],
    )


_PIPELINE_VARIABLES = _generate_all_pipeline_vars()
_FIXED_ARGUMENTS = dict(
    framework_version="1.8",
    instance_type="ml.m5.xlarge",
    role=sagemaker.get_execution_role(),  # TODO: need to mock this up
    sagemaker_session=PipelineSession(),
    network_config=NetworkConfig(
        subnets=[ParameterString(name="nw_cfg_subnets")],
        security_group_ids=[ParameterString(name="nw_cfg_security_group_ids")],
        enable_network_isolation=ParameterBoolean(name="nw_cfg_enable_nw_isolation"),
        encrypt_inter_container_traffic=ParameterBoolean(
            name="nw_cfg_encrypt_inter_container_traffic"
        ),
    ),  # TODO: need to recursively assign with Pipeline Variable in later changes
    estimator_cls=PyTorch,
    code="s3://my-bucket/code",
    inputs=[
        ProcessingInput(
            source="s3://my-bucket/inputs",
            destination=ParameterString(name="proc_input_dest"),
            s3_data_type=ParameterString(name="proc_input_s3_data_type"),
            app_managed=ParameterBoolean(name="proc_input_app_managed"),
        ),  # TODO: need to recursively assign with Pipeline Variable in later changes
    ],
    outputs=[
        ProcessingOutput(
            source=ParameterString(name="proc_output_source"),
            destination=ParameterString(name="proc_output_dest"),
            app_managed=ParameterBoolean(name="proc_output_app_managed"),
        ),  # TODO: need to recursively assign with Pipeline Variable in later changes
    ],
)
_EXPECTED_ERRORS = dict(
    ValueError=dict(
        image_uri=dict(
            instance_type="While image_uri is None, a Pipeline variable was given as instance_type"
        )
    )
)
# A dict to present the bond relationship between params
# E.g. if outputs (a parameter in FrameworkProcessor.run) is None,
# output_kms_key (a parameter in constructor) is omitted
_PARAM_BONDS = dict(outputs=["output_kms_key"])


class PipelineVarCompatiTestTemplate:
    """Check the compatibility between Pipeline variables and the given class, target method"""

    def __init__(self, clazz: type, default_args: dict, pipeline_session: PipelineSession):
        """Initialize a `PipelineVarCompatiTestTemplate` instance.

        Args:
            clazz (type): The class to test the compatibility.
            default_args (dict): The given default arguments for the class and its target method.
            pipeline_session (PipelineSession): The pipeline session.
        """
        self._clazz = clazz
        self._default_args = default_args
        self._pipeline_session = pipeline_session
        self._target_func = _target_class_function(clazz)
        self._clz_params = _get_param_dict(clazz.__init__)
        self._func_params = _get_param_dict(self._target_func)

    def check_compatibility(self):
        """The entry to check the compatibility"""

        print(
            "Starting to check Pipeline variable compatibility for class (%s) and target method (%s)\n"
            % (self._clazz.__name__, self._target_func.__name__)
        )
        _check_or_fill_in_args(
            params={**self._clz_params["required"], **self._clz_params["optional"]},
            default_args=self._default_args["clazz_args"],
        )
        _check_or_fill_in_args(
            params={**self._func_params["required"], **self._func_params["optional"]},
            default_args=self._default_args["func_args"],
        )

        # Check the case when all args are assigned a not-None value
        print("### Starting to check the compatibility when all optional args are not None ###")
        self._iterate_clz_params_to_check_compatibility()

        # Check the case when one of the optional arg is None
        print(
            "### Starting to check the compatibility when one of the optional arg is None in each round ###"
        )
        self._iterate_optional_params_to_check_compatibility()

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
            origin_val = default_args[param_name]
            default_args[param_name] = None
            print("=== Parameter (%s) is None in this round ===" % param_name)
            self._iterate_clz_params_to_check_compatibility(param_name)
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
            for clz_ppl_var, expected_clz_expr in _generate_pipeline_vars_per_type(
                clz_param_name, clz_param_type
            ):
                # print(
                #     'Replacing class arg (%s) with pipeline variable which is expected to be (%s)'
                #     % (clz_param_name, expected_clz_expr)
                # )
                self._default_args["clazz_args"][clz_param_name] = clz_ppl_var

                obj = None
                try:
                    obj = self._clazz(**self._default_args["clazz_args"])
                except ValueError as val_err:
                    if param_with_none and _EXPECTED_ERRORS["ValueError"][param_with_none][
                        clz_param_name
                    ] in str(val_err):
                        continue
                    raise val_err

                self._iterate_func_params_to_check_compatibility(
                    obj=obj,
                    expected_clz_expr=expected_clz_expr,
                    param_with_none=param_with_none,
                )
            # print("============================\n")
            self._default_args["clazz_args"][clz_param_name] = clz_default_arg

    def _iterate_func_params_to_check_compatibility(
        self,
        obj: object,
        expected_clz_expr: str,
        param_with_none: Optional[str] = None,
    ):
        """Iterate each target func parameter and assign a pipeline var to it

        Args:
            obj (object): The class object.
            expected_clz_expr (str): The expected json expression of a class argument.
            param_with_none (str): The name of the parameter with None value.
        """
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
            for func_ppl_var, expected_func_expr in _generate_pipeline_vars_per_type(
                func_param_name, func_params[func_param_name]["type"]
            ):
                # print(
                #     'Replacing func arg (%s) with pipeline variable which is expected to be (%s)'
                #     % (func_param_name, expected_func_expr)
                # )
                self._default_args["func_args"][func_param_name] = func_ppl_var
                target_func = getattr(obj, self._target_func.__name__)
                step = ProcessingStep(
                    name="MyProcessingStep",
                    run_args=target_func(**self._default_args["func_args"]),
                )
                pipeline = Pipeline(
                    name="MyPipeline",
                    steps=[step],
                    sagemaker_session=self._pipeline_session,
                )
                step_dsl = json.dumps(json.loads(pipeline.definition())["Steps"][0])
                assert (
                    json.dumps(expected_clz_expr["origin"]) in step_dsl
                    or json.dumps(expected_clz_expr["to_string"]) in step_dsl
                )
                assert (
                    json.dumps(expected_func_expr["origin"]) in step_dsl
                    or json.dumps(expected_func_expr["origin"]) in step_dsl
                )

                # TODO: remove the following hard code assertion once recursive assignment is added
                if param_with_none != "network_config":
                    assert json.dumps({"Get": "Parameters.nw_cfg_subnets"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.nw_cfg_security_group_ids"}) in step_dsl
                    assert json.dumps({"Get": "Parameters.nw_cfg_enable_nw_isolation"}) in step_dsl

                # Uncomment if testing on any Processor
                # if param_with_none != "outputs":
                #     assert json.dumps({"Get": "Parameters.proc_output_source"}) in step_dsl
                #     assert json.dumps({"Get": "Parameters.proc_output_dest"}) in step_dsl
                #     assert json.dumps({"Get": "Parameters.proc_output_app_managed"}) in step_dsl
                # if param_with_none != "inputs":
                #     assert json.dumps({"Get": "Parameters.proc_input_dest"}) in step_dsl
                #     assert json.dumps({"Get": "Parameters.proc_input_s3_data_type"}) in step_dsl
                #     assert json.dumps({"Get": "Parameters.proc_input_app_managed"}) in step_dsl

            self._default_args["func_args"][func_param_name] = func_default_arg
            # print()
        # print("-------------------------\n")


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
        and (param_with_none in _PARAM_BONDS)
        and (param_name in _PARAM_BONDS[param_with_none])
    )


def _does_support_pipeline_variable(t: type) -> bool:
    """Check if pipeline variable is supported by a parameter according to its type

    Args:
        t (type): The type to be checked

    Return:
        bool: True if it supports. False otherwise.
    """
    return "PipelineVariable" in str(t)


def _target_class_function(clz: type) -> object:
    """Fetch the target function based on class

    Args:
        clz (type): The class to fetch its target function for test.

    Return:
        object: The target function is returned.
    """
    # TODO: add other down stream class
    if "Processor" in clz.__name__:
        return clz.run
    raise TypeError(f"Unable to get target function for class {clz}")


def _get_param_dict(obj: type) -> dict:
    """Get a parameter dict of a given class or method

    The dict indicates if a parameter is required or not, as well as its type.

    Arg:
        obj (type): The type of the object (class or method) to generate
            the parameter dict.

    Return:
        dict: A parameter dict is returned.
    """
    params = signature(obj)
    params_dict = dict(
        required=dict(),
        optional=dict(),
    )
    for param in params.parameters.values():
        if param.annotation is param.empty:
            continue
        if param.name == "sagemaker_session":
            # Treat sagemaker_session as required as it must be a PipelineSession obj
            params_dict["required"][param.name] = dict(type=param.annotation)
        elif param.default is param.empty:
            params_dict["required"][param.name] = dict(type=param.annotation)
        else:
            params_dict["optional"][param.name] = dict(type=param.annotation)
    return params_dict


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


def _get_pipeline_var(
    types: tuple,
) -> list:
    """Get a Pipeline variable based on one kind of the parameter types.

    Args:
        types (tuple): The possible types of a parameter.

    Return:
        list: a list of possible PipelineVariable objects are returned
    """
    if str in types:
        return _PIPELINE_VARIABLES["str"]
    if int in types:
        return _PIPELINE_VARIABLES["int"]
    if float in types:
        return _PIPELINE_VARIABLES["float"]
    if bool in types:
        return _PIPELINE_VARIABLES["bool"]
    raise TypeError(f"Unable to parse types: {types}.")


def _get_non_pipeline_val(n: str, t: type) -> object:
    """Get the value (not a Pipeline variable) based on parameter type and name

    Args:
        n (str): The parameter name. If a parameter has a pre-defined value,
            it will be returned directly.
        t (type): The parameter type. If a parameter does not have a pre-defined value,
            an arg will be auto-generated based on the type.

    Return:
        object: A Python primitive value is returned.
    """
    if n in _FIXED_ARGUMENTS:
        return _FIXED_ARGUMENTS[n]
    if t is str:
        return _STR_VAL
    if t is int:
        return 1
    if t is float:
        return 1.5
    if t is bool:
        return bool(getrandbits(1))
    if t in [list, tuple, dict, set]:
        return t()

    raise TypeError(f"Unable to parse type: {t}.")


def _check_or_fill_in_args(params: dict, default_args: dict):
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
            default_args[param_name] = _get_non_pipeline_val(param_name, clean_type)
        else:
            default_args[param_name] = _get_non_pipeline_val(param_name, origin_type)
