from pathlib import Path

from setuptools import find_packages, setup
from setuptools.dist import Distribution


README = Path(__file__).with_name("README.md")

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True  # forces a platform-specific wheel

setup(
    name="sibylengine",
    version="0.0.5",
    description="Custom Vulkan renderer",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["sibylengine.__pycache__", "sibylengine.__pycache__.*"]),
    include_package_data=True,
    package_data={
        "sibylengine": [
            "*.so",
            "*.pyd",
            "*.pyi",
            "layouts/*.ini",
            "assets/**/*",
            "shaders/**/*.slang",
        ],
    },
    distclass=BinaryDistribution,
)
