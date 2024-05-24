"""Install PyDynamic in Python path and provide all packaging metadata."""

import codecs
import os
from os import path
from typing import List, Tuple

from setuptools import Command, find_packages, setup


def get_readme():
    """Get README.md's content"""
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as file:
        return file.read()


def get_version(rel_path: str) -> str:
    """Extract __version__ variable's value from a file's content"""
    for line in read_file_content(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def read_file_content(rel_path: str) -> str:
    """Extract a file's content and provide a variable to access it"""
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), "r") as file:
        return file.read()


class Tweet(Command):
    """Handle the tweeting on executing setup.py tweet"""

    _filename: str

    _consumer_key: str
    _consumer_secret: str
    _access_token: str
    _access_token_secret: str
    _twitter_api_auth_handle = None

    description: str = "Send new tweets to the Twitter API to announce releases"

    user_options: List[Tuple[str, str, str]] = [
        ("filename=", "f", "filename containing the tweet")
    ]

    def initialize_options(self):
        """Set filename to default value"""
        self._filename = "tweet.txt"

    def finalize_options(self):
        """React to invalid circumstance that no filename is set"""
        if self._filename is None:
            raise RuntimeError("Parameter --filename is missing")

    def run(self):
        """Actually organize and conduct the tweeting"""
        from tweepy import Client

        def set_twitter_api_secrets_from_environment():
            self._consumer_key = os.getenv("CONSUMER_KEY")
            self._consumer_secret = os.getenv("CONSUMER_SECRET")
            self._access_token = os.getenv("ACCESS_TOKEN")
            self._access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        def set_twitter_api_auth_handle():
            self._twitter_api_auth_handle = raise_error_or_retrieve_handle()

        def raise_error_or_retrieve_handle() -> Client:
            try:
                return initialize_twitter_api_auth_handle()
            except TypeError as type_error_message:
                if "must be string or bytes" in str(type_error_message):
                    raise ValueError(
                        "ValueError: Environment variables 'CONSUMER_KEY', "
                        "'CONSUMER_SECRET', 'ACCESS_TOKEN' and 'ACCESS_TOKEN_SECRET' "
                        "have to be set."
                    )
                raise TypeError from type_error_message

        def initialize_twitter_api_auth_handle() -> Client:
            return Client(
                consumer_key=self._consumer_key,
                consumer_secret=self._consumer_secret,
                access_token=self._access_token,
                access_token_secret=self._access_token_secret,
            )

        def tweet():
            self._twitter_api_auth_handle.create_tweet(text=read_tweet_from_file())

        def read_tweet_from_file() -> str:
            with open(self._filename, "r", encoding="utf-8") as file:
                content: str = file.read()
            return content

        set_twitter_api_secrets_from_environment()
        set_twitter_api_auth_handle()
        tweet()


current_release_version = get_version("src/PyDynamic/__init__.py")

setup(
    name="PyDynamic",
    version=current_release_version,
    description="A software package for the analysis of dynamic measurements",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://ptb-m4d.github.io/PyDynamic/",
    download_url=(
        f"https://github.com/PTB-M4D/PyDynamic/releases/download/"
        f"v{current_release_version}/PyDynamic-{current_release_version}.tar.gz"
    ),
    author=(
        "Sascha Eichstädt, Maximilian Gruber, Björn Ludwig, Thomas Bruns, Martin Weber"
    ),
    author_email="sascha.eichstaedt@ptb.de",
    keywords="measurement uncertainty, dynamic measurements, metrology, GUM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    project_urls={
        "Documentation": f"https://pydynamic.readthedocs.io/en/"
        f"v{current_release_version}/",
        "Source": f"https://github.com/PTB-M4D/PyDynamic/tree/"
        f"v{current_release_version}/",
        "Tracker": "https://github.com/PTB-M4D/PyDynamic/issues",
    },
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sympy",
        "PyWavelets",
        "time-series-buffer",
    ],
    extras_require={
        "examples": ["notebook"],
        "dev": [
            "black[jupyter]",
            "hypothesis",
            "ipykernel",
            "ipython",
            "myst-parser",
            "nbsphinx",
            "pytest",
            "pytest-cov",
            "pytest-custom-exit-code",
            "python-semantic-release<8",
            "sphinx",
            "sphinx-rtd-theme",
            "tox",
            "tweepy",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 7 - Inactive",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or "
        "later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    cmdclass={
        "tweet": Tweet,
    },
)
