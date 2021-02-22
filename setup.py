import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='dsmnet',
    version='1.0a0',
    author='Anselme Borgeaud',
    author_email='aborgeaud@gmail.com',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/afeborgeaud/dsmnet',
    packages=setuptools.find_packages(),
    classifiers=[],
    python_requires='>=3.7',
)