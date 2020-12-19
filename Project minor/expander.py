# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:24:08 2020

@author: PRATHAM
"""
import requests

url = 'http://shorturl.at/uzIR8'
resp = requests.head(url, allow_redirects=True) # so connections are recycled

print(resp.url)