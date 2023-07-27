import re
import setuptools
from os import path

requirements_file = path.join(path.dirname(__file__), "requirements.in")
requirements = [r for r in open(requirements_file).read().split("\n") if not re.match(r"^\-", r)]

setuptools.setup(
    name="causal_eval",
    version="0.1",
    url="https://github.com/kakeith/rct_rejection_sampling/",
    packages=setuptools.find_packages(),
    install_requires=requirements,  # dependencies specified in requirements.in
)