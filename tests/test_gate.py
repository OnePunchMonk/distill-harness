from harness.gate import Budget, GateConfig, should_escalate
from harness.signals import self_reported_signal, structural_signal


def test_low_confidence_escalates():
    signal = self_reported_signal(0.2)
    budget = Budget(max_per_session=10)
    config = GateConfig(default_confidence_threshold=0.5)
    assert should_escalate(signal, "default", budget, config) is True


def test_high_confidence_does_not_escalate():
    signal = self_reported_signal(0.9)
    budget = Budget(max_per_session=10)
    config = GateConfig(default_confidence_threshold=0.5)
    assert should_escalate(signal, "default", budget, config) is False


def test_structural_failure_escalates():
    signal = structural_signal(passed=False)
    budget = Budget(max_per_session=10)
    config = GateConfig()
    assert should_escalate(signal, "code_gen", budget, config) is True


def test_budget_exhaustion_blocks_escalation():
    signal = structural_signal(passed=False)
    budget = Budget(max_per_session=1)
    config = GateConfig()
    assert should_escalate(signal, "default", budget, config) is True
    budget.spend("default")
    assert should_escalate(signal, "default", budget, config) is False


def test_per_class_budget_cap():
    signal = structural_signal(passed=False)
    budget = Budget(max_per_session=10, max_per_task_class={"code_gen": 1})
    config = GateConfig()
    assert should_escalate(signal, "code_gen", budget, config) is True
    budget.spend("code_gen")
    assert should_escalate(signal, "code_gen", budget, config) is False
    # other classes unaffected
    assert should_escalate(signal, "summarization", budget, config) is True
