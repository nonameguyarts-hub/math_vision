"""
Visual smoke tests driven by a headless Chrome browser via CDP.
Verifies rendered UI states, raw HTML leaks, icon rendering, container health,
responsiveness, layout stability during reading, and review warning strings.
"""
import asyncio
import json
import os
import socket
import subprocess
import time
import unittest
import requests
import websockets
import base64

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots_visual")
ICON_NAMES = [
    "uploadupload", "uploadUpload", "arrow_right", "arrow_drop",
    "keyboard_arrow", "expand_more", "arrow_drop_down"
]

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

class TestVisualSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.port = find_free_port()
        cls.cdp_port = find_free_port()
        
        env = os.environ.copy()
        env["MATHVISION_DEMO"] = "1"
        env["MATHVISION_STUB_SLEEP"] = "5"
        env["MATHVISION_STUB_LATEX"] = "5 x + 50 = 160"
        
        # Start streamlit server
        cls.st_proc = subprocess.Popen([
            r".venv\Scripts\streamlit.exe", "run", "app.py",
            "--server.port", str(cls.port),
            "--server.headless", "true"
        ], env=env)
        
        # Start Chrome headless with remote debugging
        cls.ch_proc = subprocess.Popen([
            CHROME_PATH,
            "--headless=new",
            f"--remote-debugging-port={cls.cdp_port}",
            "--disable-gpu",
            "--no-sandbox",
            "--window-size=1280,900"
        ])
        
        # Wait for CDP endpoint
        cls.ws_url = None
        for _ in range(30):
            try:
                r = requests.get(f"http://localhost:{cls.cdp_port}/json", timeout=1)
                tabs = r.json()
                if tabs:
                    cls.ws_url = tabs[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.5)
        
        if not cls.ws_url:
            raise RuntimeError("Failed to connect to Chrome CDP")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "ch_proc"):
            cls.ch_proc.terminate()
            try:
                cls.ch_proc.wait(timeout=3)
            except Exception:
                cls.ch_proc.kill()
        if hasattr(cls, "st_proc"):
            cls.st_proc.terminate()
            try:
                cls.st_proc.wait(timeout=3)
            except Exception:
                cls.st_proc.kill()

    def run_async(self, coro):
        return asyncio.run(coro)

    async def _get_cdp_client(self):
        ws = await websockets.connect(self.ws_url, max_size=50*1024*1024)
        cid = [1]
        async def send(method, params=None):
            c = cid[0]
            cid[0] += 1
            msg = {"id": c, "method": method, "params": params or {}}
            await ws.send(json.dumps(msg))
            while True:
                resp = json.loads(await ws.recv())
                if resp.get("id") == c:
                    return resp.get("result", {})
        async def evaluate(expr):
            res = await send("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
            return res.get("result", {}).get("value")
        async def capture_shot(path, width=1280, height=900):
            await send("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": 1, "mobile": (width < 600)
            })
            await asyncio.sleep(0.4)
            res = await send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
            data = base64.b64decode(res["data"])
            with open(path, "wb") as f:
                f.write(data)
            if width != 1280:
                await send("Emulation.setDeviceMetricsOverride", {
                    "width": 1280, "height": 900, "deviceScaleFactor": 1, "mobile": False
                })
                await asyncio.sleep(0.2)

        async def click_element(expr):
            for _ in range(25):
                dis = await evaluate(f"""(() => {{
                    const el = {expr};
                    return el ? el.disabled : true;
                }})()""")
                if not dis:
                    break
                await asyncio.sleep(0.2)

            box = await evaluate(f"""(() => {{
                const el = {expr};
                if (!el) return null;
                el.scrollIntoView({{block: 'center'}});
                const rect = el.getBoundingClientRect();
                return {{x: rect.left + rect.width / 2, y: rect.top + rect.height / 2}};
            }})()""")
            if box:
                await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": box["x"], "y": box["y"], "button": "left", "clickCount": 1})
                await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": box["x"], "y": box["y"], "button": "left", "clickCount": 1})
            await evaluate(f"""(() => {{
                const el = {expr};
                if (el) el.click();
            }})()""")
            return True

        async def type_latex(val):
            box = await evaluate("""(() => {
                const inp = document.querySelector('input[aria-label*="Recognized LaTeX"]');
                if (!inp) return null;
                inp.scrollIntoView({block: 'center'});
                const r = inp.getBoundingClientRect();
                return {x: r.left + r.width / 2, y: r.top + r.height / 2};
            })()""")
            if not box:
                return False
            # Triple click to natively focus and select all text
            await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": box["x"], "y": box["y"], "button": "left", "clickCount": 3})
            await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": box["x"], "y": box["y"], "button": "left", "clickCount": 3})
            await asyncio.sleep(0.1)
            # Insert the new text over the selection
            await send("Input.insertText", {"text": val})
            await asyncio.sleep(0.2)
            # Dispatch Enter key to commit value to Streamlit
            await send("Input.dispatchKeyEvent", {"type": "rawKeyDown", "windowsVirtualKeyCode": 13, "key": "Enter", "code": "Enter"})
            await send("Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 13, "key": "Enter", "code": "Enter"})
            await asyncio.sleep(1.5)
            return True

        return ws, send, evaluate, capture_shot, click_element, type_latex

    def test_visual_checks_suite(self):
        """Run complete visual smoke verification."""
        async def flow():
            ws, send, evaluate, capture_shot, click_element, type_latex = await self._get_cdp_client()
            try:
                await send("Page.enable")
                await send("DOM.enable")
                await send("Runtime.enable")
                
                # Navigate to app
                await send("Page.navigate", {"url": f"http://localhost:{self.port}"})
                
                # Wait for app loaded
                for _ in range(40):
                    ready = await evaluate("document.querySelectorAll('input[type=\"file\"]').length > 0")
                    if ready:
                        break
                    await asyncio.sleep(0.5)
                await asyncio.sleep(1.5)
                
                failures = []
                
                # Verify no tabs exist
                tab_count = await evaluate("document.querySelectorAll('[data-testid=\"stTab\"], [data-testid=\"stTabs\"], [role=\"tablist\"]').length")
                if tab_count > 0:
                    failures.append(f"[Layout] Expected 0 tab elements after simplification, found: {tab_count}")
                
                # Helper for text leaks check
                async def check_text_leaks(state_name):
                    text = await evaluate("document.body.innerText")
                    for bad in ["<div", "</", "class=", "mv-"]:
                        if bad in text:
                            failures.append(f"[{state_name}] Raw HTML leak: text contains '{bad}'")
                    leaks = await evaluate("""(() => {
                        const leaks = [];
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('[data-testid="stIconMaterial"]').forEach(el => el.remove());
                        const t = clone.innerText;
                        if (t.includes("uploadupload") || t.includes("uploadUpload")) {
                            leaks.push("uploadupload");
                        }
                        for (const bad of ["arrow_right", "keyboard_arrow", "expand_more", "arrow_drop_down"]) {
                            if (t.includes(bad)) {
                                leaks.push(bad);
                            }
                        }
                        return leaks;
                    })()""")
                    for bad in leaks:
                        failures.append(f"[{state_name}] Raw icon name leak outside icon element: text contains '{bad}'")
                
                # Helper for horizontal overflow check
                async def check_overflow(state_name, width):
                    await send("Emulation.setDeviceMetricsOverride", {
                        "width": width, "height": 900, "deviceScaleFactor": 1, "mobile": (width < 600)
                    })
                    await asyncio.sleep(0.3)
                    overflow = await evaluate("""(() => {
                        const clientW = document.documentElement.clientWidth;
                        const scrollW = document.documentElement.scrollWidth;
                        return scrollW > clientW + 2; // small tolerance
                    })()""")
                    if overflow:
                        failures.append(f"[{state_name} @ {width}px] Horizontal overflow detected (scrollWidth > clientWidth)")

                # Helper for card container non-empty check
                async def check_card_containers(state_name):
                    empty_cards = await evaluate("""(() => {
                        const cards = Array.from(document.querySelectorAll('div[class*="st-key-card_"]'));
                        return cards.filter(c => {
                            if (c.offsetParent === null || window.getComputedStyle(c).display === 'none') {
                                return false;
                            }
                            const rect = c.getBoundingClientRect();
                            return rect.height > 0 && (c.children.length === 0 || c.innerText.trim() === '');
                        }).length;
                    })()""")
                    old_empty_cards = await evaluate("""(() => {
                        const cards = Array.from(document.querySelectorAll('div.mv-card'));
                        return cards.filter(c => {
                            if (c.offsetParent === null || window.getComputedStyle(c).display === 'none') {
                                return false;
                            }
                            return c.children.length === 0 || c.innerText.trim() === '';
                        }).length;
                    })()""")
                    if empty_cards > 0 or old_empty_cards > 0:
                        failures.append(f"[{state_name}] Found empty card containers (empty st-key-card: {empty_cards}, empty mv-card: {old_empty_cards})")

                # Helper for icon font check
                async def check_icon_fonts(state_name):
                    bad_icons = await evaluate("""(() => {
                        const icons = Array.from(document.querySelectorAll('[data-testid="stIconMaterial"]'));
                        return icons.filter(el => {
                            const fam = window.getComputedStyle(el).fontFamily;
                            return !fam.toLowerCase().includes('material') && !fam.toLowerCase().includes('symbols');
                        }).map(el => ({text: el.innerText, fam: window.getComputedStyle(el).fontFamily}));
                    })()""")
                    if bad_icons:
                        failures.append(f"[{state_name}] Icon elements do not use icon font: {bad_icons}")

                # -------------------------------------------------------------
                # 1. Solve Page: Empty State
                # -------------------------------------------------------------
                await check_text_leaks("Solve Empty")
                await check_icon_fonts("Solve Empty")
                await check_card_containers("Solve Empty")
                await check_overflow("Solve Empty", 1280)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_empty_1280.png"), 1280)
                await check_overflow("Solve Empty", 420)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_empty_420.png"), 420)
                
                # Check Solve button is disabled when LaTeX is empty
                solve_disabled = await evaluate("""(() => {
                    const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Solve');
                    return btn ? btn.disabled : null;
                })()""")
                if solve_disabled is not True:
                    failures.append(f"[Solve Empty] Expected Solve button disabled=True when LaTeX empty, got: {solve_disabled}")

                # Check palette button labels do not show raw LaTeX
                palette_raw = await evaluate("""(() => {
                    const btns = Array.from(document.querySelectorAll('button')).map(b => b.textContent.trim());
                    return btns.filter(t => t.includes('\\\\frac') || t.includes('\\\\sqrt'));
                })()""")
                if palette_raw:
                    failures.append(f"[Solve Empty] Palette buttons show raw LaTeX: {palette_raw}")

                # Check that full layout elements are present on the single-page Solve app
                layout_ok = await evaluate("""(() => {
                    const title = document.querySelector('.mv-title');
                    const stepper = document.querySelector('.mv-stepper');
                    const card1 = document.querySelector('div[class*=\"st-key-card_step1\"]');
                    const card2 = document.querySelector('div[class*=\"st-key-card_step2\"]');
                    return !!(title && stepper && card1 && card2);
                })()""")
                if not layout_ok:
                    failures.append("[Layout] Essential layout components missing on Solve page")

                # -------------------------------------------------------------
                # 5. Photo Uploaded State
                # -------------------------------------------------------------
                doc = await send("DOM.getDocument")
                root_id = doc["root"]["nodeId"]
                inp_node = await send("DOM.querySelector", {"nodeId": root_id, "selector": 'input[type="file"]'})
                test_photo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_photos", "WhatsApp Image 2026-10-15 at 2.16.35 PM.jpeg")
                await send("DOM.setFileInputFiles", {"files": [test_photo], "nodeId": inp_node["nodeId"]})
                await asyncio.sleep(2.5)

                await check_text_leaks("Photo Uploaded")
                await check_overflow("Photo Uploaded", 1280)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_photo_uploaded_1280.png"), 1280)
                await check_overflow("Photo Uploaded", 420)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_photo_uploaded_420.png"), 420)

                # -------------------------------------------------------------
                # 6. Reading in Progress & Layout Height Stability (5s Stub)
                # -------------------------------------------------------------
                # Trigger Read equation
                await click_element("Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Read equation'))")
                await asyncio.sleep(1.0)
                
                # Check spinner text verbatim
                sp_text = await evaluate("""(() => {
                    const sp = document.querySelector('.stSpinner');
                    return sp ? sp.innerText.trim() : '';
                })()""")
                if sp_text != "Recognising equation in image...":
                    failures.append(f"[Reading In Progress] Expected exact spinner text 'Recognising equation in image...', got: '{sp_text}'")

                h_during = await evaluate("document.documentElement.scrollHeight")
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_reading_1280.png"), 1280)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_reading_420.png"), 420)

                # Wait for 5s sleep stub to finish and spinner to disappear
                for _ in range(30):
                    is_spin = await evaluate("document.querySelectorAll('.stSpinner').length > 0")
                    if not is_spin:
                        break
                    await asyncio.sleep(0.5)

                h_after = await evaluate("document.documentElement.scrollHeight")
                diff = abs(h_after - h_during)
                print(f"Layout height change during reading: during={h_during}, after={h_after}, diff={diff}")
                if diff > 100:
                    failures.append(f"[Reading In Progress] Layout height jumped significantly: during={h_during}, after={h_after}, diff={diff}")

                # -------------------------------------------------------------
                # 7. Answer with Steps State ("5 x + 50 = 160")
                # -------------------------------------------------------------
                # Click Solve
                await click_element("Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Solve')")
                for _ in range(25):
                    page_txt = await evaluate("document.body.innerText")
                    if "x = 22" in page_txt or "22" in page_txt:
                        break
                    await asyncio.sleep(0.5)

                await check_text_leaks("Answer State")
                await check_overflow("Answer State", 1280)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_answered_1280.png"), 1280)
                await check_overflow("Answer State", 420)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_answered_420.png"), 420)

                page_txt = await evaluate("document.body.innerText")
                if "x = 22" not in page_txt and "22" not in page_txt:
                    failures.append("[Answer State] Expected 'x = 22' not found in solved output")

                # -------------------------------------------------------------
                # 8. Warning State with stub "500 - 50 = 160"
                # -------------------------------------------------------------
                await type_latex("500 - 50 = 160")
                from ui_checks import review
                expected_warn = review("500 - 50 = 160", "Arithmetic")["warnings"][0]
                for _ in range(25):
                    warn_page = await evaluate("document.body.innerText")
                    if expected_warn in warn_page:
                        break
                    await asyncio.sleep(0.5)

                if expected_warn not in warn_page:
                    failures.append(f"[Warning State] Review warning '{expected_warn}' not found verbatim in page")

                await capture_shot(os.path.join(OUTPUT_DIR, "solve_warning_1280.png"), 1280)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_warning_420.png"), 420)

                # -------------------------------------------------------------
                # 9. Error State with stub "2 x + 3 y = 74"
                # -------------------------------------------------------------
                await type_latex("2 x + 3 y = 74")
                for _ in range(20):
                    curr_val = await evaluate("(() => { const el = document.querySelector('input[aria-label*=\"Recognized LaTeX\"]'); return el ? el.value : ''; })()")
                    if "2 x + 3 y = 74" in curr_val:
                        break
                    await asyncio.sleep(0.5)
                # Click Solve
                await click_element("Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Solve')")
                
                expected_err = review("2 x + 3 y = 74", "Auto")["error"]
                for _ in range(25):
                    err_page = await evaluate("document.body.innerText")
                    if expected_err in err_page:
                        break
                    await asyncio.sleep(0.5)

                if expected_err not in err_page:
                    print(f"DEBUG ERROR STATE: expected='{expected_err}'")
                    print(f"DEBUG PAGE TEXT: {err_page}")
                    failures.append(f"[Error State] Review/solver error '{expected_err}' not found verbatim in page")

                await capture_shot(os.path.join(OUTPUT_DIR, "solve_error_1280.png"), 1280)
                await capture_shot(os.path.join(OUTPUT_DIR, "solve_error_420.png"), 420)

                # Report all failures
                if failures:
                    msg = f"Visual smoke test detected {len(failures)} failures:\n" + "\n".join(f"  - {f}" for f in failures)
                    self.fail(msg)

            finally:
                await ws.close()

        self.run_async(flow())

if __name__ == "__main__":
    unittest.main()
