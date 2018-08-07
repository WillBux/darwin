from setuptools import setup
from setuptools import find_packages

setup(name='darwin',
      version='0.1',
      description='Machine Learning with Genetic Algorithms',
      author='Will Buxton',
      author_email='will.buxton88@gmail.com',
      url='https://github.com/WillBux/darwin',
      license='MIT',
      install_requires=['tqdm>=4.19.4',
                        'numpy>=1.13.3',
                        'pandas>=0.23.4',
                        'scikit-learn>=0.19.0'],
      packages=find_packages())
