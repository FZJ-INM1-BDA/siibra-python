from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
from dataclasses import is_dataclass, asdict as _asdict
from threading import Thread
from typing import Any, TYPE_CHECKING, Union
import sys
from uuid import uuid4

if TYPE_CHECKING:
    from siibra.locations import Point, PointSet

IDENTITY=[
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]

key_mapping = {
    "field_id": "@id"
}

def custom_dict_factory(list_of_tuples):
    updated = [(key_mapping.get(n, n), v) for n, v in list_of_tuples]
    return dict(updated)

def asdict(val_in: dict|list|str|int|float):
    if is_dataclass(val_in):
        val_in = _asdict(val_in, dict_factory=custom_dict_factory)
    if isinstance(val_in, (str, int, float)):
        return val_in
    if isinstance(val_in, (list, tuple)):
        return [asdict(v) for v in val_in if v is not None]
    if isinstance(val_in, dict):
        return {
            key: asdict(value)
            for key, value in val_in.items()
            if value is not None
        }
    raise Exception(f"Cannot deal with type {type(val_in)}")

TEMPLATE: str = None

LOG_FLAG = False

class ReqHndl(BaseHTTPRequestHandler):
    sxplr_requests = []

    @property
    def tmpl(self):
        with open(Path(__file__).parent / "template.html", "r") as fp:
            return fp.read()
    
    def queue_sxplr_request(self, req) -> None:
        self.sxplr_requests.append(req)
    
    def log_message(self, format: str, *args: Any) -> None:
        if LOG_FLAG:
            return super().log_message(format, *args)

    def do_GET(self):
        if self.path == "/template.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            self.wfile.write(bytes(self.tmpl, "utf-8"))
            return
        if self.path == "/ping":
            self.send_response(200)

            self.send_header("Content-Type", "application/json")
            self.end_headers()
            ReqHndl.sxplr_requests
            result = json.dumps([
                asdict(req)
                for req in ReqHndl.sxplr_requests
            ])
            ReqHndl.sxplr_requests = []
            self.wfile.write(bytes(result, "utf-8"))
            return
        self.send_error(404, f"{self.path} Not Found")
    
    def do_POST(self):
        if self.path.endswith("data"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes("OK", "utf-8"))
            return
        self.send_error(500, f"{self.path} is not supported")


class ThreadedController(Thread):

    # exits on main thread exits
    daemon = True

    def __init__(self, port=7099, *, debug=False):
        super().__init__()
        global LOG_FLAG
        if debug:
            LOG_FLAG = True
        self.port = port
        self.server = ThreadingHTTPServer(("localhost", self.port), ReqHndl)
        

    def run(self) -> None:
        print("!!!NOT FOR PRODUCTION USE!!!")
        print(f"Listening on {self.port}")
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()


class Explorer:
    """Start a controller for explorer.
    If used as a context manager, will start the server automatically.
    If used normally, user should call start() to start the server, and call stop() when done.
    
    Args:
        root_url: str The viewer URL that will be launched
        port: int Port plugin is running
    """

    def __init__(self, root_url="https://atlases.ebrains.eu/viewer/", port:int=7099) -> None:
        self.controller = ThreadedController(port)
        self.root_url = root_url

    def __enter__(self):
        self.start()
        return self

    def __exit__(self):
        self.stop()

    def start(self, *, atlas_spec: str="human", space_spec: str="mni 152", parcellation_spec: str="julich 3"):
        from . import encode_url
        import siibra

        atlas = siibra.atlases[atlas_spec]
        space = siibra.spaces[space_spec]
        parc = siibra.parcellations[parcellation_spec]
        self.goto_url = encode_url(atlas, space, parc, root_url=self.root_url, query_params={
            "pl": '["http://localhost:7099/template.html"]'
        })
        print(f"Go to {self.goto_url}", file=sys.stderr)
        self.controller.start()
        return self.goto_url

    def stop(self):
        self.controller.stop()

    def _check_alive(self):
        if not self.controller.is_alive():
            raise Exception(f"controller is not yet live")

    def navigate(self, *, position=None, orientation=None, zoom=None,
                 perspective_orientation=None, perspective_zoom=None):
        self._check_alive()

        from .api.request.navigateTo.request import Model, Params
        ReqHndl.sxplr_requests.append(
            Model(
                id=str(uuid4()),
                params=Params(
                    position=position,
                    orientation=orientation,
                    perspectiveOrientation=perspective_orientation,
                    perspectiveZoom=perspective_zoom,
                    zoom=zoom,
                    animate=True,
                )
            )
        )

    def overlay(self, *, url: str, transform=IDENTITY):
        self._check_alive()
        from .api.request.loadLayers.request import Model, Params, AddableLayer
        ReqHndl.sxplr_requests.append(
            Model(
                id=str(uuid4()),
                params=Params(layers=[
                    AddableLayer(source=url, transform=transform)
                ])
            )
        )

    @staticmethod
    def _point_to_coords(point: 'Point', name="Untitled", color: str="#ffffff"):
        from .api.request.addAnnotations.request import (
            SxplrCoordinatePointExtension, CoordinatePointModel,
            ApiModelsOpenmindsSANDSV3MiscellaneousCoordinatePointCoordinates as Coord
        )
        return SxplrCoordinatePointExtension(
            color=color,
            name=name,
            openminds=CoordinatePointModel(
                field_type="",
                field_id="",
                coordinateSpace={
                    "@id": point.space.id
                },
                coordinates=[Coord(value=v) for v in point]
            )
        )

    @staticmethod
    def _pointset_to_coords(pointset: 'PointSet', name="Untitled", color: str="#ff0000"):
        return [Explorer._point_to_coords(point, name, color)
                for point in pointset]

    def annotate(self, *, points: Union['Point', 'PointSet']):
        self._check_alive()
        from .api.request.addAnnotations.request import Model, Params
        from siibra.locations import Point, PointSet
        append_points = []

        if isinstance(points, PointSet):
            append_points.extend(
                Explorer._pointset_to_coords(points)
            )
        if isinstance(points, Point):
            append_points.append(
                Explorer._point_to_coords(points)
            )
        ReqHndl.sxplr_requests.append(
            Model(
                id=str(uuid4()),
                params=Params(
                    annotations=append_points
                )
            )
        )

    def select(self, *, atlas_spec: str=None, template_spec: str=None, parcellation_spec: str=None):
        assert (
            bool(atlas_spec) + bool(template_spec) + bool(parcellation_spec) == 1
        ), f"""Expected one and only one of {atlas_spec}, {template_spec}, and {parcellation_spec} to be set"""

        import siibra
        if bool(atlas_spec):
            from .api.request.selectAtlas.request import Model, AtId
            atlas = siibra.atlases[atlas_spec]
            if atlas is None:
                raise Exception(f"{atlas_spec} did not resolve to any atlas")
            ReqHndl.sxplr_requests.append(
                Model(
                    id=str(uuid4()),
                    params=AtId(field_id=atlas.id)
                )
            )
            return

        if bool(template_spec):
            from .api.request.selectTemplate.request import Model, AtId
            space = siibra.spaces[template_spec]
            if space is None:
                raise Exception(f"{template_spec} did not resolve to any space")
            ReqHndl.sxplr_requests.append(
                Model(
                    id=str(uuid4()),
                    params=AtId(field_id=space.id)
                )
            )
            return

        if bool(parcellation_spec):
            from .api.request.selectParcellation.request import Model, AtId
            parcellation = siibra.parcellations[parcellation_spec]
            if parcellation is None:
                raise Exception(f"{parcellation_spec} did not resolve to any parcellation")
            ReqHndl.sxplr_requests.append(
                Model(
                    id=str(uuid4()),
                    params=AtId(field_id=parcellation.id)
                )
            )
            return
