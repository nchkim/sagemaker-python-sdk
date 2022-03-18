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

from sagemaker.processing import FrameworkProcessor, ScriptProcessor, Processor
from tests.unit.sagemaker.workflow.test_mechanism.test_pipeline_var_compatibility_template import (
    PipelineVarCompatiTestTemplate,
)
from tests.unit.sagemaker.workflow.test_mechanism import (
    ROLE,
    DUMMY_S3_SCRIPT_PATH,
    PIPELINE_SESSION,
)


# These tests provide the incomplete default arg dict
# within which some class or target func parameters are missing or assigned a None value.
# The test template will fill in those missing/None args
# Note: the default args should not include PipelineVariable objects
def test_processor_compatibility():
    default_args = dict(
        clazz_args=dict(
            role=ROLE,
            volume_size_in_gb=None,
            sagemaker_session=PIPELINE_SESSION,
        ),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Processor,
        default_args=default_args,
    )
    test_template.check_compatibility()


def test_script_processor_compatibility():
    default_args = dict(
        clazz_args=dict(
            role=ROLE,
            volume_size_in_gb=None,
            sagemaker_session=PIPELINE_SESSION,
        ),
        func_args=dict(
            code=DUMMY_S3_SCRIPT_PATH,
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=ScriptProcessor,
        default_args=default_args,
    )
    test_template.check_compatibility()


def test_framework_processor_compatibility():
    default_args = dict(
        clazz_args=dict(
            estimator_cls=None,
            role=ROLE,
            py_version="py3",
            volume_size_in_gb=None,
            sagemaker_session=PIPELINE_SESSION,
        ),
        func_args=dict(
            code=DUMMY_S3_SCRIPT_PATH,
        ),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=FrameworkProcessor,
        default_args=default_args,
    )
    test_template.check_compatibility()
