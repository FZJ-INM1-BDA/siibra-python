from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="brainscapes",
    version="0.0.4.dev1",
    author="Vadim Marcenko",
    author_email="v.marcenko@fz-juelich.de",
    description="brainscapes client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://jugit.fz-juelich.de/v.marcenko/brainscapes.git",
    scripts=['brainscapes/cli.py'],
    entry_points='''
        [console_scripts]
        brainscapes=cli:brainscapes
    ''',
    packages=find_packages(include=['brainscapes', 'brainscapes.*']),
    # packages=find_packages(include=['.']),
    package_data={'brainscapes.configurations': ['atlases/*.json','parcellations/*.json','spaces/*.json']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['anytree','nibabel','click','clint','appdirs'],
)

