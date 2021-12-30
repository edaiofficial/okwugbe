import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="okwugbe",
    version="0.1.7",
    author="Bonaventure F. P. Dossou - Chris C. Emezue",
    author_email="edai.official.edai@gmail.com",
    description="Automatic Speech Recognition Library for African Languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edaiofficial/asr-africa",
    project_urls={
        "Bug Tracker": "https://github.com/edaiofficial/asr-africa/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["numpy", "torch", "torchaudio", "datasets==1.12.1", "colorama",
                      "ipython==5.5.0", "livelossplot", "commonvoice-utils"]
)
