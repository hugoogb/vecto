import json

from typer.testing import CliRunner

from vecto.cli import app

runner = CliRunner()


def test_convert_default_output_and_summary(ring_png):
    res = runner.invoke(app, ["convert", str(ring_png)])
    assert res.exit_code == 0, res.output
    assert ring_png.with_suffix(".svg").read_text().startswith("<svg")
    assert "fidelity" in res.output


def test_legacy_single_command(ring_png, tmp_path):
    out = tmp_path / "o.svg"
    res = runner.invoke(app, ["single", str(ring_png), str(out), "--threshold", "128", "--preview"])
    assert res.exit_code == 0, res.output
    assert out.exists() and out.with_suffix(".preview.png").exists()


def test_stdin_to_stdout(ring_png):
    res = runner.invoke(app, ["convert", "-", "-q"], input=ring_png.read_bytes())
    assert res.exit_code == 0, res.output
    assert res.stdout.strip().startswith("<svg") and res.stdout.strip().endswith("</svg>")


def test_json_output(ring_png, tmp_path):
    out = tmp_path / "o.svg"
    res = runner.invoke(app, ["convert", str(ring_png), str(out), "--json", "-p", "logo", "--auto"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["output"] == str(out)
    assert data["fidelity"] > 0.9 and data["candidates_tried"] > 1
    assert data["options"]["crop"] is True


def test_json_to_stdout_embeds_svg(ring_png):
    res = runner.invoke(app, ["convert", str(ring_png), "-", "--json"])
    assert json.loads(res.stdout)["svg"].startswith("<svg")


def test_missing_file_and_bad_mode(tmp_path):
    assert runner.invoke(app, ["convert", str(tmp_path / "nope.png")]).exit_code != 0
    res = runner.invoke(app, ["convert", "-", "--mode", "rainbow"], input=b"x")
    assert res.exit_code != 0


def test_batch_applies_options_recurses_and_continues_on_error(ring_png, color_png, tmp_path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    (src / "a.png").write_bytes(ring_png.read_bytes())
    (src / "sub" / "b.png").write_bytes(color_png.read_bytes())
    (src / "broken.png").write_bytes(b"garbage")
    out = tmp_path / "out"
    res = runner.invoke(app, [
        "batch", str(src), str(out), "-r", "-j", "1", "--color", "#123456", "--json",
    ])
    assert res.exit_code == 1  # one failure
    data = json.loads(res.stdout)
    assert data["converted"] == 2 and data["failed"] == 1
    assert "#123456" in (out / "a.svg").read_text()
    assert (out / "sub" / "b.svg").exists()

    res2 = runner.invoke(app, ["batch", str(src), str(out), "-r", "--skip-existing", "--json"])
    assert json.loads(res2.stdout)["skipped"] == 2


def test_batch_parallel(ring_png, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):
        (src / f"{i}.png").write_bytes(ring_png.read_bytes())
    res = runner.invoke(app, ["batch", str(src), str(tmp_path / "out"), "-j", "2", "-q"])
    assert res.exit_code == 0, res.output
    assert len(list((tmp_path / "out").glob("*.svg"))) == 3


def test_presets_command():
    res = runner.invoke(app, ["presets", "--json"])
    assert "signature" in json.loads(res.stdout)
