import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evolution_opt", 
    version="0.0.5",
    author="Eliot Kalfon",
    author_email="eliott.kalfon@gmail.com",
    description="Evolution inspired optimisation algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliottkalfon/evolution_opt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)