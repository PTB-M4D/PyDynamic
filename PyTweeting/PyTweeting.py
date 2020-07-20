# -*- coding: utf-8 -*-
"""Tweet a random sequence of eight letters."""

import random
import string

import tweepy as tweepy


def generate_random_string():
    letters = string.ascii_lowercase
    res_string = ''.join(random.choice(letters) for i in range(8))
    return res_string


print(generate_random_string())
auth = tweepy.OAuthHandler('DJ3yOjQhddvAMkhDffs85kmfR',
                           'yIw8OWpdoAOL167xpRBgpRtkBgHA7CuQQlI3M5DD2CFBNBnkuk')

auth.set_access_token('1166597179135471621-CtRH6kHoxYecyKlI0xOI7PeGV9lHAR',
                      'XaCGjngj88Ed6AZEnYh22mZLrFGCAI1RSaBV6XftO3rQG')

api = tweepy.API(
    auth)#, proxy='https://webproxy.bs.ptb.de:8080') #use when tweeting from inside
# PTB Network

api.update_status(generate_random_string())
