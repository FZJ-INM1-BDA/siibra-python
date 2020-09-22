from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="brainscapes",
    version="0.0.1.dev1",
    author="Vadim Marcenko",
    author_email="v.marcenko@fz-juelich.de",
    description="brainscapes client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/HumanBrainProject/atlas-core-lib.git",
    scripts=['brainscapes/cli.py'],
    entry_points='''
        [console_scripts]
        brainscapes=cli:brainscapes
    ''',
    packages=find_packages(include=['brainscapes']),
    package_data={'brainscapes.definitions': ['atlases/*.json','parcellations/*.json','spaces/*.json']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['anytree','nibabel','clint','appdirs'],
)

