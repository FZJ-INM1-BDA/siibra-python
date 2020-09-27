
class Glossary:
    """
    A very simple class that provides enum-like siple autocompletion for an
    arbitrary list of names.
    """
    def __init__(self,words):
        self.words = list(words)

    def __dir__(self):
        return self.words

    def __str__(self):
        return "\n".join(self.words)

    def __iter__(self):
        return (w for w in self.words)

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.words:
            return name
        else:
            raise AttributeError("No such term: {}".format(name))

def create_key(name):
    """
    Creates an uppercase identifier string that includes only alphanumeric
    characters and underscore from a natural language name.
    """
    return "".join(e if e.isalnum() else '_' 
        for e in name.strip()).upper()

