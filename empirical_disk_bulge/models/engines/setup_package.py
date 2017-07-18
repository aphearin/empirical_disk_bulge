from distutils.extension import Extension
import os

PATH_TO_PKG = os.path.relpath(os.path.dirname(__file__))
SOURCES = ('disk_bulge_simple_disruption_engine.pyx', 'disk_in_situ_bulge_ex_situ.pyx')
THIS_PKG_NAME = '.'.join(__name__.split('.')[:-1])


def get_extensions():

    names = [THIS_PKG_NAME + "." + src.replace('.pyx', '') for src in SOURCES]
    sources = [os.path.join(PATH_TO_PKG, srcfn) for srcfn in SOURCES]
    include_dirs = ['numpy']
    libraries = []
    language = 'c'

    extensions = []
    for name, source in zip(names, sources):
        extensions.append(Extension(name=name,
            sources=[source],
            include_dirs=include_dirs,
            libraries=libraries,
            language=language))

    return extensions
