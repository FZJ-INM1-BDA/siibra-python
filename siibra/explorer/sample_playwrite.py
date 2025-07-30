from siibra.locations import Point
from siibra.explorer.plugin import Explorer
from siibra.explorer.url import decode_url
from playwright.sync_api import sync_playwright

dismiss_preamble = [
    ("Welcome to ebrains siibra explorer", "Dismiss"),
    ("Privacy Policy", "Ok"),
]

wanted_dialog = [
    ("http://localhost:7099/template.html", "OK"),
]


def assess_roundtrip(pt: Point, space_spec: str):
    
    explorer = Explorer(root_url="https://atlases.ebrains.eu/viewer-staging/")
    url = explorer.start(space_spec=pt.space)
    with sync_playwright() as playwrite:
        chromium = playwrite.chromium
        browser = chromium.launch()
        context = browser.new_context()
        page = context.new_page()

        page.goto(url)
        page.wait_for_timeout(1000)

        curr_url = page.url
        
        for text, click in dismiss_preamble:
            dialog = page.get_by_role("dialog").filter(has_text=text)
            btn = dialog.locator("//button").filter(has_text=click)
            btn.click(force=True)
        
        for text, click in wanted_dialog:
            dialog = page.get_by_role("dialog").filter(has_text=text)
            btn = dialog.locator("//button").filter(has_text=click)
            btn.click()

        explorer.navigate(position=[coord * 1e6 for coord in pt]) # in nm
        
        page.wait_for_timeout(5000)
        now_url = page.url

        explorer.select(template_spec=space_spec)
        page.wait_for_timeout(30000)
        space_specced_url = page.url

        
        explorer.select(template_spec=pt.space)
        page.wait_for_timeout(5000)
        return_url = page.url

        start = decode_url(curr_url)
        navigated = decode_url(now_url)
        space_specced = decode_url(space_specced_url)
        returned = decode_url(return_url)

        print_result = zip(
            ("start", "navigated", "space_specced", "returned"),
            (start, navigated, space_specced, returned), 
        )

        print("end")
        for name, bbox in print_result:
            print(name, bbox.bounding_box.center)

if __name__ == "__main__":
    pt = Point([10, 10, 10], "colin 27")
    assess_roundtrip(pt, "mni152")
