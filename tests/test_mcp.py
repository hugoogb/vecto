import pytest

mcp_server = pytest.importorskip("vecto.mcp_server")


def test_tool_converts_and_reports(ring_png, tmp_path):
    out = tmp_path / "x.svg"
    data = mcp_server.vectorize_image(str(ring_png), str(out), preset="logo", return_svg=True)
    assert out.exists() and data["output_path"] == str(out)
    assert data["svg"].startswith("<svg") and data["fidelity"] > 0.9


def test_list_presets():
    assert "cut" in mcp_server.list_presets()
