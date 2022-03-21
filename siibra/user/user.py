from .drive import EbrainsDrive


class User:

    def __init__(self, auth_token=None, username=None, password=None):
        self.ebrains_drive = None
        if auth_token or (username and password):
            self.auth(auth_token, username, password)

    def auth(self, auth_token=None, username=None, password=None):
        assert auth_token or (username or password)
        self.ebrains_drive = EbrainsDrive(auth_token=auth_token, username=username, password=password)

    def annotations(self, anno_type=None):
        assert self.ebrains_drive, 'Please authenticate to access annotations'

        annotations = self.ebrains_drive.load(anno_type=anno_type)
        return annotations

    def annotation(self, anno_id: None):
        assert self.ebrains_drive, 'Please authenticate to access annotations'
        return self.ebrains_drive.load(anno_id=anno_id)

    def store(self, store_item, name=None):
        assert self.ebrains_drive, 'Please authenticate to access annotations'

        ##### ToDo it requires object to has .to_model().json()
        self.ebrains_drive.save(obj=store_item.to_model().json(), name=name)
        return

    def remove_annotation(self, anno_id):
        assert self.ebrains_drive, 'Please authenticate to access annotations'

        self.ebrains_drive.remove(anno_id=anno_id)

