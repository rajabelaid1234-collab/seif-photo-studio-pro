from __future__ import annotations

import storage


def test_expired_items_are_deleted_but_fresh_items_remain(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "data")
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(storage.time, "time", lambda: clock["now"])

    expired_id = storage.save_item("owner-a", "old.png", b"old")
    # The second item is created 20 days later, so it remains fresh at day 31.
    clock["now"] += 20 * storage.DAY
    fresh_id = storage.save_item("owner-a", "fresh.png", b"fresh")
    clock["now"] += 11 * storage.DAY

    assert storage.cleanup_expired() == 1
    assert storage.get_bytes("owner-a", expired_id) is None
    assert storage.get_bytes("owner-a", fresh_id) == b"fresh"
    assert [row[0] for row in storage.list_items("owner-a")] == [fresh_id]


def test_users_cannot_read_or_delete_other_users_files(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "data")
    first = storage.save_item("owner-a", "a.png", b"private-a")
    second = storage.save_item("owner-b", "b.png", b"private-b")

    assert storage.get_bytes("owner-a", first) == b"private-a"
    assert storage.get_bytes("owner-a", second) is None
    storage.delete_item("owner-a", second)
    assert storage.get_bytes("owner-b", second) == b"private-b"
    assert len(storage.list_items("owner-a")) == 1
    assert len(storage.list_items("owner-b")) == 1
