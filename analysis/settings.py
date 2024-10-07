import os
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator


def path_exists(filename: str):
    assert os.path.exists(filename), f"{filename} does not exist"
    return filename


def is_file(filename: str):
    assert os.path.isfile(filename), f"{filename} is not a file"
    return filename


FileName = Annotated[str, AfterValidator(path_exists), AfterValidator(is_file)]


class FileSettings(BaseModel):
    datafile: FileName
    data_settings: dict = Field(default={})


class TrainingSettings(FileSettings):
    epochs: int = Field(default=10, gt=0)
    core_limit: int = Field(default=0, ge=0)
