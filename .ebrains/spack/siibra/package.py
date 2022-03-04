from spack import *


class Siibra(PythonPackage):
    """Software interfaces for interacting with brain atlases."""

    homepage = "https://readthedocs.org/projects/siibra-python/"
    git = "https://github.com/FZJ-INM1-BDA/siibra-python"
    pypi     = "siibra/siibra-0.3a11.tar.gz"

    maintainers = ['Timo Dickscheid', 'Xiao Gui', 'Vadim Marcenko']

    version('0.3a11', sha256='d4b6a1b9a49823403dc3d29b8a7aa38f2f832ec6a324d8ae39a68b082bdd50a6')
    version('0.3a10', sha256='f2994cc60baa7a0a5b54ce5cead2439a444d975b0e1d13b1c83754d97e5fc4a3')
    version('0.3a8', sha256='550658e16bc9ff169128217bcb7e3fc6e281f17dba11e168cf1cd756fdce298c')

    depends_on('python@3.6:', type=('build', 'run'))
    depends_on('py-setuptools', type='build')

    depends_on('anytree', type=('build', 'run'))
    depends_on('nibabel', type=('build', 'run'))
    depends_on('click', type=('build', 'run'))
    depends_on('appdirs', type=('build', 'run'))
    depends_on('scikit-image', type=('build', 'run'))
    depends_on('requests', type=('build', 'run'))
    depends_on('memoization', type=('build', 'run'))
    depends_on('neuroglancer-scripts', type=('build', 'run'))
    depends_on('nilearn', type=('build', 'run'))
    depends_on('simple-term-menu', type=('build', 'run'))
    depends_on('importlib-resources; python_version < "3.7"', type=('build', 'run'))