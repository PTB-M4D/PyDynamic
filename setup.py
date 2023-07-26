"""Install PyDynamic in Python path and provide all packaging metadata."""
import codecs
import os
from os import path

from setuptools import Command, find_packages, setup


def get_readme():
    """Get README.md's content"""
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        return f.read()


def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), "r") as fp:
        return fp.read()


class Tweet(Command):
    filename: str

    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str

    description = "Send new tweets to the Twitter API to announce releases"

    user_options = [("filename=", "f", "filename containing the tweet")]

    def initialize_options(self):
        self.filename = "tweet.txt"

    def finalize_options(self):
        if self.filename is None:
            raise RuntimeError("Parameter --filename is missing")

    def run(self):
        import tweepy
        from tweepy import Client

        def _set_twitter_api_secrets():
            self.consumer_key = os.getenv("CONSUMER_KEY")
            self.consumer_secret = os.getenv("CONSUMER_SECRET")
            self.access_token = os.getenv("ACCESS_TOKEN")
            self.access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        def _tweet():
            # _get_twitter_api_handle().create_tweet(text=read_tweet_from_file())
            _get_twitter_api_auth_handle().search_recent_tweets(
                query="metrology", max_results=1
            )

        def _get_twitter_api_auth_handle() -> Client:
            try:
                auth = tweepy.Client(
                    consumer_key=self.consumer_key,
                    consumer_secret=self.consumer_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret,
                )
                return auth
            except TypeError as e:
                if "Consumer key must be" in str(e):
                    raise ValueError(
                        "ValueError: Environment variables 'CONSUMER_KEY', "
                        "'CONSUMER_SECRET', 'ACCESS_TOKEN' and 'ACCESS_TOKEN_SECRET' "
                        "have to be set."
                    )

                raise TypeError(str(e))

        def read_tweet_from_file() -> str:
            with open(self.filename, "r") as f:
                content: str = f.read()
            return content

        _set_twitter_api_secrets()
        _tweet()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


current_release_version = get_version("src/PyDynamic/__init__.py")

setup(
    name="PyDynamic",
    version=current_release_version,
    description="A software package for the analysis of dynamic measurements",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://ptb-m4d.github.io/PyDynamic/",
    download_url="https://github.com/PTB-M4D/PyDynamic/releases/download/v{0}/"
    "PyDynamic-{0}.tar.gz".format(current_release_version),
    author="Sascha Eichstädt, Maximilian Gruber, Björn Ludwig, Thomas Bruns, "
    "Martin Weber",
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
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or "
        "later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
    cmdclass={
        "tweet": Tweet,
    },
)
