import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brainscapes-client",
    version="0.0.1.dev1",
    author="Vadim Marcenko",
    author_email="v.marcenko@fz-juelich.de",
    description="brainscapes client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/HumanBrainProject/atlas-core-lib.git",
    packages=setuptools.find_packages(include=['brainscapes_client']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],
)
