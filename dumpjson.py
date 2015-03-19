#!/usr/bin/env python

import json
import sqlite3
import sys

class Joke(object):
    def __init__(self, idx, text, rating, time):
        self._id = idx
        self.text = text
        self.rating = rating
        self.time = time

        self.categories = []
        self.subcategories = []

    def to_dict(self):
        """because using a dict in the first place was too mainstream
           long live JS!"""
        return {'text': self.text,
                'rating': self.rating,
                'categories': self.categories,
                'subcategories': self.subcategories}

def get_all_jokes():
    """Get all jokes in memory, 5M no worries here"""

    conn = sqlite3.connect('./data/jokes_sickipedia.sqlite3')
    c = conn.cursor()
    qry = "SELECT id_joke, text, rating, time FROM jokes"
    cat_qry = """SELECT distinct category from categories c,
                                               joke_has_category jhc
                 WHERE jhc.id_joke = %d AND jhc.id_category = c.id_category
              """
    subcat_qry = """SELECT distinct subcategory from subcategories s,
                                               joke_has_category jhc
                    WHERE jhc.id_joke = %d AND
                          jhc.id_subcategory = s.id_subcategory
                 """
    for row in c.execute(qry):
        j = Joke(*row)
        category_c = conn.cursor()
        for category, in category_c.execute(cat_qry % j._id):
            j.categories.append(category)
        subcategory_c = conn.cursor()
        for subcat, in subcategory_c.execute(subcat_qry % j._id):
            j.subcategories.append(subcat)
        yield j

def main(args):
    if len(args) != 2:
        print "Usage: %s dumfile" % args[0]
        return 42

    lst = [j.to_dict() for j in get_all_jokes()]

    print "Have %d objects" % len(lst)
    with open(args[1], 'w') as f:
        json.dump(lst, f)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
