from networkgeometry.stimuli.definitions import load_structures, load_templates, prompts_for

def test_day_and_month_states():
    s = load_structures()
    assert [st.label for st in s["day"].states][:2] == ["Monday", "Tuesday"]
    assert s["day"].n_states == 7 if hasattr(s["day"], "n_states") else len(s["day"].states) == 7
    assert len(s["month"].states) == 12
    assert "May" in s["month"].excluded

def test_templates_have_slot_and_pools():
    t = load_templates()
    assert all("{X}" in tpl for tpl in t["shared"])
    assert "day" in t["specific"] and "month" in t["specific"]

def test_prompts_fill_slot_in_canonical_order():
    s = load_structures(); t = load_templates()
    runs = prompts_for(s["day"], t["shared"])
    assert runs[0][0].endswith("Monday") or "Monday" in runs[0][0]
    assert len(runs[0]) == 7
