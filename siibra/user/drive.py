import ebrains_drive
import json
import os
from io import StringIO
from uuid import UUID


class EbrainsDrive:

    drive_connected = False
    default_repo = None

    annotation_folder = 'Siibra annotations'
    annotation_drive_dir = None

    def __init__(self, auth_token=None, username=None, password=None):
        if auth_token or (username and password):
            self.connect_drive(auth_token=auth_token, username=username, password=username)

    def connect_drive(self, auth_token=None, username=None, password=None):
        assert auth_token or (username and password)

        if auth_token:
            client = ebrains_drive.connect(token=auth_token)
        if username and password:
            client = ebrains_drive.connect(username=username, password=password)

        if not client:
            print('Credentials invalid')

        list_repos = client.repos.list_repos()
        self.default_repo = client.repos.get_repo(list_repos[0].id)

        root_dir = self.default_repo.get_dir('/')
        if not root_dir.check_exists(self.annotation_folder):
            root_dir.mkdir(self.annotation_folder)

        self.annotation_drive_dir = self.default_repo.get_dir(f'/{self.annotation_folder}')
        self.drive_connected = True

    def load(self, anno_id=None, anno_type=None):
        """Load annotations from ebrains drive
        """
        if not self.drive_connected:
            self.connect_drive()

        stored_files = [f for f in self.annotation_drive_dir.ls()
                        if isinstance(f, ebrains_drive.files.SeafFile)]

        if not stored_files:
            return

        annotations = []
        for file in stored_files:
            file = self.default_repo.get_file(file.path)
            obj = json.loads(file.get_content())
            obj['name'] = os.path.splitext(file.name)[0]
            if anno_id:
                if obj['@id'] == anno_id:
                    return obj
            else:
                annotations.append(obj)

        if anno_type:
            annotations = list(filter(lambda c: c['@type'] == anno_type, annotations))

        return annotations

    def save(self, obj, name=None):
        if not self.drive_connected:
            self.connect_drive()

        ##### ToDo check if file exists with same id and remove?
        #      Or check if file exists with same id and name and then remove?
        #      Or do not remove at all?
        # if self.drive_id and self.file_name and (not name or self.file_name == name):
        #     self.remove_from_drive()

        if not name:
            name = 'Unnamed'

        annotation_file = StringIO(json.dumps(json.loads(obj), indent=4, sort_keys=True, cls=UUIDEncoder))
        self.annotation_drive_dir.upload(annotation_file, f'{name}.json')


    """Remove file from ebrains drive
    anno_id : str
        Id of the file
    """
    def remove(self, anno_id):
        assert anno_id
        if self.drive_connected is False:
            self.connect_drive()

        anno_files = [anno for anno in self.annotation_drive_dir.ls()
                      if isinstance(anno, ebrains_drive.files.SeafFile)]
        for file in anno_files:
            stored_file = self.default_repo.get_file(file.path)
            file_id = json.loads(stored_file.get_content())['@id']
            if file_id == anno_id:
                stored_file.delete()
                return

        print('File not found')


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return obj.hex
        return json.JSONEncoder.default(self, obj)
