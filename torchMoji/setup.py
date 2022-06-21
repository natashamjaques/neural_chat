from setuptools import setup

setup(
    name='torchmoji',
    version='1.0.1',
    packages=['torchmoji'],
    description='torchMoji',
    include_package_data=True,
    install_requires=[
        'emoji==0.4.5',
        'numpy==1.22.0',
        'scipy==1.2.0',
        'scikit-learn==0.19.0',
        'text-unidecode==1.0',
    ],
)
