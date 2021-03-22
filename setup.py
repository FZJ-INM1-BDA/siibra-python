from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brainscapes",
    version="0.0.8.8",
    author="Forschungszentrum Juelich, Institute of Neuroscience and Medicine (INM-1) - V. Marcenko, T. Dickscheid",
    author_email="v.marcenko@fz-juelich.de",
    description="Brainscapes - Multilevel Human Brain Atlas",
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['anytree','statsmodels','nibabel','click','appdirs','scikit-image','scipy','statsmodels','requests','python-gitlab','memoization','cloud-volume'],
)

