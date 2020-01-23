from scipy.stats import norm
import numpy as np
import pickle
from pystan import StanModel
import falcon

class QuoteResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': (
                "I've always been more interested in "
                "the future than in the past."
            ),
            'author': 'Grace Hopper'
        }

        resp.media = quote

class TTest:
    def on_get(self, req, resp):
        resp.media = "Student"

api = falcon.API()
api.add_route('/quote', QuoteResource())
api.add_route('/two_sample_test', TTest())
