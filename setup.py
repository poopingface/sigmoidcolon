from setuptools import setup


def get_version() -> str:
    rel_path = "sigmoidcolon/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name='sigmoidcolon',
    version=get_version(),
    description='The biologically inspired sigmoid activation function.',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='http://github.com/poopingface/sigmoidcolon',
    author='Pooping Face',
    author_email='poopingface.co@gmail.com',
    license='MIT',
    packages=['sigmoidcolon'],
    install_requires=["torch"],
    zip_safe=False
)