import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ladybug-geometry",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author="Ladybug Tools",
    author_email="info@ladybug.tools",
    description='Library housing basic geometry objects and computation methods '
    'needed for the Ladybug Tools core libraries.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ladybug-tools/ladybug-geometry",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: IronPython",
        "Operating System :: OS Independent"
    ],
    license="AGPL-3.0"
)
