class Joke(object):
    def __init__(self,
                 idx=0,
                 text="",
                 rating=0,
                 time=0,
                 categories=None,
                 subcategories=None):
        self._id = idx
        self.text = text
        self.rating = rating
        self.time = time
        self.categories = categories or []
        self.subcategories = subcategories or []

    def to_dict(self):
        """because using a dict in the first place was too mainstream
           long live JS!"""
        return {'text': self.text,
                'rating': self.rating,
                'categories': self.categories,
                'subcategories': self.subcategories}


