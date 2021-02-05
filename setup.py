from distutils.core import setup


if __name__ == '__main__':

    setup(
        name='PytorchTrainer',
        version='0.0.1',
        author='Andrew M. Webb',
        author_email='andrew@awebb.info',
        packages=['pytorch_trainer'],
        url='http://www.awebb.info',
        license='MIT License',
        description='A package for a simple high-level pytorch training interface',
        python_requires='>=3.4.3',
        install_requires=[
            'numpy >= 1.11.3',
            'torch',
            'tensorboardX',
            'tqdm',
            'gitpython'
        ],
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
    )
