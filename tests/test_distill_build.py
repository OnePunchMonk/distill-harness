from harness.logging_ import EscalationRecord
from distill.build import dedupe_records, filter_records, rebalance_records


def make_record(input="q", teacher="a", student="wrong", task_class="default", verdict=None):
    return EscalationRecord(
        schema_version=1,
        timestamp=0.0,
        task_class=task_class,
        input=input,
        student_attempt=student,
        teacher_output=teacher,
        verifier_verdict=verdict,
        signal_kind="structural",
    )


def test_filter_drops_empty_teacher_output():
    records = [make_record(teacher=""), make_record(teacher="real answer")]
    out = filter_records(records)
    assert len(out) == 1
    assert out[0].teacher_output == "real answer"


def test_filter_drops_identical_pairs():
    records = [make_record(teacher="same", student="same"), make_record(teacher="a", student="b")]
    out = filter_records(records)
    assert len(out) == 1


def test_filter_drops_teacher_failed_verdict():
    records = [make_record(verdict="teacher_failed"), make_record(verdict="ok")]
    out = filter_records(records)
    assert len(out) == 1
    assert out[0].verifier_verdict == "ok"


def test_dedupe_normalizes_whitespace_and_case():
    records = [make_record(input="Hello World"), make_record(input="hello   world")]
    out = dedupe_records(records)
    assert len(out) == 1


def test_rebalance_caps_per_class():
    records = [make_record(input=str(i), task_class="a") for i in range(5)]
    records += [make_record(input=str(i), task_class="b") for i in range(2)]
    out = rebalance_records(records, max_per_class=3)
    a_count = sum(1 for r in out if r.task_class == "a")
    b_count = sum(1 for r in out if r.task_class == "b")
    assert a_count == 3
    assert b_count == 2
