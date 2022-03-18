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

import pytest

import sagemaker
from sagemaker.network import NetworkConfig
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline_context import PipelineSession
from tests.unit.sagemaker.workflow.test_pipeline_var_compatibility_template import (
    PipelineVarCompatiTestTemplate,
)

DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
DUMMY_LOCAL_SCRIPT_PATH = "file://dummy-local/dummy_script.py"


@pytest.fixture
def pipeline_session():
    # TODO: need to mock this up
    return PipelineSession()


@pytest.fixture
def network_config():
    return NetworkConfig(
        subnets=["my_subnet_id"],
        security_group_ids=["my_security_group_id"],
        enable_network_isolation=True,
        encrypt_inter_container_traffic=True,
    )


@pytest.fixture
def processing_inputs():
    return [
        ProcessingInput(
            source="s3://my-bucket/inputs",
            destination="destination",
        )
    ]


@pytest.fixture
def processing_outputs():
    return [
        ProcessingOutput(
            source="/opt/ml/processing/outputs/",
            destination="s3://my-bucket/outputs",
            s3_upload_mode="Continuous",
        )
    ]


# @pytest.mark.skip(reason="This is just a test example. Please unmark to use if needed")
def test_framework_processor_compatibility_with_partial_args(
    pipeline_session, network_config, processing_inputs, processing_outputs
):
    # This test provides the incomplete default arg dict
    # within which some class or target func parameters are missing or assigned a None value.
    # The test template will fill in those missing args
    default_args = dict(
        clazz_args=dict(
            estimator_cls=None,
            role=sagemaker.get_execution_role(),  # TODO: need to mock this up
            py_version="py3",
            volume_size_in_gb=None,
            sagemaker_session=pipeline_session,
        ),
        func_args=dict(
            code=DUMMY_S3_SCRIPT_PATH,
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=FrameworkProcessor,
        default_args=default_args,
        pipeline_session=pipeline_session,
    )
    test_template.check_compatibility()
