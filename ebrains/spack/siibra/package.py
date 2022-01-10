from spack import *

class Siibra(PythonPackage):
    """
    Software interfaces for interacting with brain atlases
    """

    homepage = 'https://readthedocs.org/projects/siibra-python/'
    git = 'https://github.com/FZJ-INM1-BDA/siibra-python'
    pypi = 'siibra/siibra-0.3a11.tar.gz'
    maintainers = ['Timo Dickscheid', 'Xiao Gui', 'Vadim Marcenko']

    version('0.3a11', '2e438db14f230cea99e290f36e9d0e61')
    version('0.3a10', 'fa7fba0b491f4b8655e2a0c1902b78ba')
    version('0.3a8', 'd9d7cf591c287ea09a562e9bf552506c')
    version('0.3a6', '80f30eace3520e702a26f4a9d5747b59')

    # python_requires
    depends_on('python@3.6:', type=('build', 'run'))

    # setup_requires
    depends_on('py-setuptools', type='build')

    # install_requires
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