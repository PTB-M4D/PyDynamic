"""Tweet a random sequence of eight letters."""

import random
import os
import re
import string

import tweepy as tweepy


def read_from_file(filename='tweet.txt'):
    f = open(filename, 'r')
    content:str = f.read()
    content = content.lstrip()
    content = re.sub(' +', ' ', content)

    return content


def convert_to_bold(a: str) -> str:
    converted = ''
    for c in a:
        print(c)
        if c in string.ascii_letters or c in string.digits:

            if c.isupper():
                c = chr(ord(c) + 119743)
            elif c.islower():
                c = chr(ord(c) + 119737)
            elif c.isdigit():
                c = chr(ord(c) + 120728)
        elif c in {' ', '\n', '\t', '\r'}:
            c = ' '
        elif c in {'#'}:
            c= ''
        else:
            c = c
        converted += c
        print(c)
    return converted


def format_md_to_unicode(to_format: str) -> str:
    string_list : [str] = to_format.split('\n')
    resulting_string : str = ''
    fragment: str
    for fragment in string_list:
        fragment = fragment.lstrip()
        to_add = fragment.replace('*', u'\U000025cf')
        if '#' in fragment:
            to_add = convert_to_bold(fragment)
        resulting_string += to_add.lstrip('') + '\n'
    return resulting_string


def remove_commit_hash(text: str) -> str:
    new_text = text
    new_text = re.sub('\(.*\)', '', new_text)
    return new_text


def _tweet():
    _get_twitter_api_handle().update_status(generate_tweet())


def _get_twitter_api_handle():
    return tweepy.API(_get_twitter_api_auth_handle())


def _get_twitter_api_auth_handle():
    auth = tweepy.OAuthHandler(os.getenv("public_key"), os.getenv("public_token"))
    auth.set_access_token(os.getenv("private_key"), os.getenv("private_token"))
    return auth


def generate_tweet() -> str:
    file_content = read_from_file()
    hashless_commits = remove_commit_hash(file_content)
    tweet = format_md_to_unicode(hashless_commits)
    return tweet  # + '\n'+generate_random_string()


if __name__ == "__main__":
    _tweet()
