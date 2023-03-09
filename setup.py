from setuptools import setup

setup(
    name='sigmoidcolon',
    version='0.0.1',
    description='The biologically inspired sigmoid activation function.',
    url='http://github.com/poopingface/sigmoidcolon',
    author='Pooping Face',
    author_email='poopingface.co@gmail.com',
    license='MIT',
    packages=['sigmoidcolon'],
    install_requires=["torch"],
    zip_safe=False
)