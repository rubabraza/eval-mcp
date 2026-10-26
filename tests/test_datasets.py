from evalcore.datasets import list_datasets, load_dataset


def test_list_datasets_includes_tickets():
    names = {d["name"] for d in list_datasets()}
    assert "tickets" in names


def test_tickets_case_count_in_range():
    tickets = next(d for d in list_datasets() if d["name"] == "tickets")
    assert 30 <= tickets["cases"] <= 40


def test_load_dataset_parses_all_cases():
    cases = load_dataset("tickets")
    assert len(cases) >= 30
    assert cases[0].id == "t001"
    # Every case has valid enum labels (pydantic would have raised otherwise).
    assert all(c.expected.category and c.expected.priority for c in cases)


def test_load_missing_dataset_raises():
    try:
        load_dataset("does-not-exist")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
