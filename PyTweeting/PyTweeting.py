# -*- coding: utf-8 -*-
"""Tweet a random sequence of eight letters."""

import random
import os
import re
import string

import tweepy as tweepy


def generate_random_string():
    letters = string.ascii_lowercase
    res_string = ''.join(random.choice(letters) for i in range(8))
    return res_string


def read_from_file(filename='tweet.txt'):
    f = open(filename, 'r')
    content = f.read()
    return content


def convert_to_bold(a: str) -> str:
    converted = ''
    a.replace("\\#", " ", 10)
    for c in a:
        print(c)
        if c in {' ', '\n', '\t', '\r', '\\\\#'}:
            c = ' '
        elif c.isupper():
            c = chr(ord(c) + 119743)
        elif c.islower():
            c = chr(ord(c) + 119737)
        elif c.isdigit():
            c = chr(ord(c) + 120728)
        else:
            c = ''
        converted += c
        print(c)
    return converted


def format_md_to_unicode(to_format: str) -> str:
    string_list : [str] = to_format.split('\n')
    resulting_string : str = ''
    fragment: str
    for fragment in string_list:
        to_add = fragment
        if '\\#' in fragment:
            to_add=convert_to_bold(fragment)
        resulting_string += to_add + ' '
    return resulting_string


def remove_commit_hash(text : str) -> str:
    new_text = text
    new_text = re.sub('\(.*\)', '', new_text)
    return new_text


def generate_tweet()->str:
    file_content = read_from_file()
    hash_less_commits = remove_commit_hash(file_content)
    tweet = format_md_to_unicode(hash_less_commits)
    return tweet


auth = tweepy.OAuthHandler(os.getenv('public_key'), os.getenv('public_token'))

auth.set_access_token(os.getenv('private_key'),
                      os.getenv('private_token'))

api = tweepy.API(
    auth)  # , proxy='https://webproxy.bs.ptb.de:8080') #use when tweeting from inside
# PTB Network
print(format_md_to_unicode(remove_commit_hash(read_from_file())))
api.update_status(generate_tweet())
