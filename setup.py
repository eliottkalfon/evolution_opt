import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evolution_opt", 
    version="0.0.2",
    author="El Kal",
    author_email="eliott.kalfon@gmail.com",
    description="Evolution inspired optimisation algorithms",
    long_description="Lorem Ipsum",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)