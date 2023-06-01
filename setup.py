from distutils.core import setup

setup(
    name='pud',
    version='0.1dev',
    packages=['pud'],
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "gym==0.15.7",
    ],
    license='Apache License 2.0',
)
