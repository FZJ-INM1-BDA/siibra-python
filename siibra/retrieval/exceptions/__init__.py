class NoSiibraConfigMirrorsAvailableException(Exception): pass
class TagNotFoundException(Exception): pass

class RepositoryGetException(Exception): pass
class RepositoryGetNotFound(RepositoryGetException): pass
class RepositoryWriteException(Exception): pass
