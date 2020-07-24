# -*- coding: utf-8 -*-
"""Tweet a random sequence of eight letters."""

import random
import os
import string

import tweepy as tweepy


def generate_random_string():
    letters = string.ascii_lowercase
    res_string = ''.join(random.choice(letters) for i in range(8))
    return res_string

def read_from_file( filename = 'tweet.txt'):
    f=open(filename, 'r')
    content = f.read()
    return content

def generate_some_bold_letter():
    letters = string.ascii_lowercase
    letter = random.choice(letters)
    return '\u0067'

auth = tweepy.OAuthHandler(os.getenv('public_key'), os.getenv('public_token'))


auth.set_access_token(os.getenv('private_key'),
                      os.getenv('private_token'))

api = tweepy.API(
    auth)#, proxy='https://webproxy.bs.ptb.de:8080') #use when tweeting from inside
# PTB Network
print(read_from_file())
api.update_status(generate_some_bold_letter()+read_from_file()+generate_random_string())
