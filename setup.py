from setuptools import find_packages, setup

setup(name="malpolon",
      version="2.2.0",
      description="Malpolon v2.2.0",
      author="Theo Larcher, Titouan Lorieul, Benjamin Deneu, Lukas Picek",
      author_email="theo.larcher@inria.fr, titouan.lorieul@gmail.com, benjamin.deneu@wsl.ch, lukas.picek@inria.fr",
      url="https://github.com/plantnet/malpolon",
      classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Environment :: GPU :: NVIDIA CUDA",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Typing :: Typed",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS"
      ],
      packages=find_packages(
       where="./",
       include="malpolon*",
       exclude="malpolon.tests"),
      package_data={'malpolon.data.datasets._data': ['minigeolifeclef2022_species_details.csv']},
      python_requires=">=3.10, <4",
      install_requires=[
        "Cartopy>=0.22.0",
        "kaggle>=1.5.16",
        "matplotlib>=3.8.0",
        "numpy>=1.26.4",
        "omegaconf>=2.3.0",
        "pandas>=2.2.1",
        "Pillow>=10.3.0",
        "planetary_computer>=1.0.0",
        "pyproj>=3.6.1",
        "pystac>=1.6.1",
        "pytest>=7.2.2",
        "pytorch_lightning>=2.3.3",
        "rasterio>=1.3.8.post1",
        "scikit_learn>=1.5.0",
        "Shapely>=2.0.3",
        "tifffile>=2022.10.10",
        "timm>=0.9.2",
        "torch>=2.2.0",
        "torchgeo>=0.5.0",
        "torchmetrics>=1.2.0",
        "torchvision>=0.17.0",
        "tqdm>=4.66.3",
        "verde>=1.8.0"
      ],
      project_urls={
        "Bug Reports": "https://github.com/plantnet/malpolon/issues/new?assignees=aerodynamic-sauce-pan&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D",
        "Feature request": "https://github.com/plantnet/malpolon/issues/new?assignees=aerodynamic-sauce-pan&labels=enhancement&projects=&template=enhancement.md&title=%5BEnhancement%5D",
        "Host organizer": "https://plantnet.org/",
        "Source": "https://github.com/plantnet/malpolon",
    },
)
