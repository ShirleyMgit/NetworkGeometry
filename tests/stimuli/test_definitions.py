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

def test_hierarchy_is_three_animal_families():
    s = load_structures()
    labels = [st.label for st in s["hierarchy"].states]
    assert labels == ["dog", "cat", "horse", "crow", "eagle", "owl", "salmon", "trout", "tuna"]

def test_probe_pool_covers_every_structure_state_final():
    t = load_templates()
    probe = t["probe"]
    assert set(probe) == {"day", "month", "years", "hierarchy", "flat"}
    for frames in probe.values():
        assert len(frames) == 2
        assert all(frame.endswith("{X}") for frame in frames)

def test_probe_prompts_end_in_the_state_word():
    s = load_structures(); t = load_templates()
    runs = prompts_for(s["hierarchy"], t["probe"]["hierarchy"])
    assert runs[0][0] == "The animal is dog"
    assert runs[0][-1] == "The animal is tuna"
