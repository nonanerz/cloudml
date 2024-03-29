from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='example to run keras on gcloud ml-engine',
      author='Neeraj Natu',
      author_email='bulavaeduard@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'imutils',
          'opencv-python',
          'google-cloud-storage'
      ],
      zip_safe=False)