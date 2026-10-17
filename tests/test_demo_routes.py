"""The 3D page is additive; only named static assets can be served."""

from io import BytesIO

import pytest

from voi_diffusion_ipp.viz.demo_server import WEB, handler


@pytest.mark.parametrize(
    "path,name,mime",
    [
        ("/", "index.html", "text/html; charset=utf-8"),
        ("/app.js", "app.js", "text/javascript"),
        ("/3d/", "3d/index.html", "text/html; charset=utf-8"),
        ("/3d?run=example", "3d/index.html", "text/html; charset=utf-8"),
        ("/3d/scene.js", "3d/scene.js", "text/javascript"),
        ("/3d/terrain.js", "3d/terrain.js", "text/javascript"),
        ("/3d/assets/helicopter.glb", "3d/assets/helicopter.glb", "model/gltf-binary"),
        ("/3d/assets/aircraft.glb", "3d/assets/aircraft.glb", "model/gltf-binary"),
        (
            "/3d/assets/pine_tree_01_twig_diff_1k.jpg",
            "3d/assets/pine_tree_01_twig_diff_1k.jpg",
            "image/jpeg",
        ),
        (
            "/3d/vendor/three/OrbitControls.js",
            "3d/vendor/three/OrbitControls.js",
            "text/javascript",
        ),
    ],
)
def test_named_static_assets_preserve_original_demo(path, name, mime):
    request = object.__new__(handler(None))
    request.path = path
    responses = []
    request.send_bytes = lambda body, content_type, code=200: responses.append(
        (body, content_type, code)
    )
    request.do_GET()
    assert responses == [((WEB / name).read_bytes(), mime, 200)]


@pytest.mark.parametrize(
    "path",
    [
        "/3d/../../pyproject.toml",
        "/3d/vendor/three/../../../README.md",
        "/3d/%2e%2e/README.md",
        "/3d/unknown.js",
    ],
)
def test_static_routes_do_not_expose_arbitrary_files(path):
    request = object.__new__(handler(None))
    request.path = path
    responses = []
    request.send_json = lambda body, code=200: responses.append((body, code))
    request.wfile = BytesIO()
    request.do_GET()
    assert responses == [({"error": "Not found"}, 404)]
