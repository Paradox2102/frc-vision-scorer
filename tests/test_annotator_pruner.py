import scripts.annotator_server as server


def _use_crop_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "BOXES_DIR", tmp_path)


def test_prune_list_filters_and_normalizes_labels(tmp_path, monkeypatch):
    _use_crop_dir(tmp_path, monkeypatch)
    equipment = tmp_path / "equipment_2026hop_qm30" / "tablet2"
    legacy = tmp_path / "2026hop_qm30" / "hub3"
    equipment.mkdir(parents=True)
    legacy.mkdir(parents=True)
    (equipment / "frame00000100.jpg").write_bytes(b"equipment")
    (legacy / "frame00000200.jpg").write_bytes(b"legacy")

    result = server.prune_images("equipment")
    assert result["count"] == 1
    assert result["images"][0]["label"] == "tablet"
    assert result["images"][0]["event"] == "2026hop_qm30"

    result = server.prune_images("all")
    assert result["count"] == 2


def test_prune_serves_and_deletes_crop(tmp_path, monkeypatch):
    _use_crop_dir(tmp_path, monkeypatch)
    crop = tmp_path / "equipment_event" / "radio" / "frame00000001.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"jpeg-data")
    rel = "equipment_event/radio/frame00000001.jpg"

    served = server.prune_image(rel)
    assert served.status_code == 200
    assert served.path == crop

    deleted = server.delete_prune_image(rel)
    assert deleted == {"deleted": rel}
    assert not crop.exists()


def test_prune_rejects_path_traversal(tmp_path, monkeypatch):
    _use_crop_dir(tmp_path, monkeypatch)
    assert server.prune_image("../outside.jpg").status_code == 404
    assert server.delete_prune_image("../outside.jpg").status_code == 404
