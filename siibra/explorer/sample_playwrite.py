import sys
from pathlib import Path
from typing import Callable

from siibra.locations import BoundingBox
from siibra.explorer.plugin import Explorer
from siibra.explorer.url import decode_url, DecodeNavigationException, DecodedUrl
from playwright.sync_api import sync_playwright

dismiss_preamble = [
    ("Welcome to ebrains siibra explorer", "Dismiss"),
    ("Privacy Policy", "Ok"),
]

wanted_dialog = [
    ("http://localhost:7099/template.html", "OK"),
]


def access_region(space_spec: str, parc_spec: str, region_spec: str, *, to_space_spec: str, record_video_dir: str = None):
    import siibra
    region = siibra.get_region(parc_spec, region_spec)
    bbox = region.get_bounding_box(space_spec)
    return assess_roundtrip(bbox, to_space_spec, record_video_dir)


VIEWPORT_SIZE_WIDTH = 1600
VIEWPORT_SIZE_HEIGHT = 800

def assess_roundtrip(bbox: BoundingBox, space_spec: str, record_video_dir: str = None):

    def try_decode_url(predicate: Callable[[DecodedUrl], bool]=lambda *args: True):
        while True:
            try:
                deocded_url = decode_url(page.url, VIEWPORT_SIZE_WIDTH)
                assert predicate(deocded_url)
                return deocded_url
                
            except (DecodeNavigationException, AssertionError):
                page.wait_for_timeout(1000)

    explorer = Explorer(root_url="https://atlases.ebrains.eu/viewer-staging/")
    url = explorer.start(space_spec=bbox.space)
    with sync_playwright() as playwrite:
        chromium = playwrite.chromium
        browser = chromium.launch()
        context = browser.new_context(record_video_dir=record_video_dir) if record_video_dir else browser.new_context()
        page = context.new_page()
        page.set_viewport_size({
            'width': VIEWPORT_SIZE_WIDTH,
            'height': VIEWPORT_SIZE_HEIGHT,
        })

        page.goto(url)
        page.wait_for_timeout(1000)

        for text, click in dismiss_preamble:
            dialog = page.get_by_role("dialog").filter(has_text=text)
            btn = dialog.locator("//button").filter(has_text=click)
            btn.dispatch_event("click")

        for text, click in wanted_dialog:
            dialog = page.get_by_role("dialog").filter(has_text=text)
            btn = dialog.locator("//button").filter(has_text=click)
            btn.click()

        bbox_size = bbox.maxpoint - bbox.minpoint
        max_dimen = max([p for p in bbox_size])
        max_viewport_dimen = max(VIEWPORT_SIZE_WIDTH, VIEWPORT_SIZE_HEIGHT) / 2  # viewport is quatered, so div by 2

        # convert to nm
        zoom = max_dimen * 1e6 / max_viewport_dimen

        start_decoded = try_decode_url()
        explorer.navigate(position=[coord * 1e6 for coord in bbox.center], zoom=zoom)  # in nm
        page.wait_for_timeout(10000)
        navigated_decoded = try_decode_url(lambda b: b.bounding_box.center != start_decoded.bounding_box.center)

        explorer.select(template_spec=space_spec)
        page.wait_for_timeout(10000)
        space_seld_decoded = try_decode_url(lambda b: b.bounding_box.space.id != navigated_decoded.bounding_box.space.id)

        explorer.select(template_spec=bbox.space)
        page.wait_for_timeout(10000)

        returned_decoded = try_decode_url(lambda b: b.bounding_box.space.id != space_seld_decoded.bounding_box.space.id)

        print_result = zip(
            ("start", "navigated", "space_specced", "returned"),
            (
                start_decoded.bounding_box,
                navigated_decoded.bounding_box,
                space_seld_decoded.bounding_box,
                returned_decoded.bounding_box),
        )

        for name, bbox in print_result:
            print(name, bbox)
        context.close()


if __name__ == "__main__":
    video_flag = "--video" in sys.argv[1:]
    if video_flag:
        Path("./video").mkdir(exist_ok=True, parents=True)
        access_region("mni 152",
                      "julich brain 3.1",
                      "hoc1 left",
                      to_space_spec="colin 27",
                      record_video_dir="./video")
    else:
        access_region("mni 152",
                      "julich brain 3.1",
                      "hoc1 left",
                      to_space_spec="colin 27")
