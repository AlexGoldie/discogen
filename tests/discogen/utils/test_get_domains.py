from discogen.utils import get_domains


def test_all_domains_valid_config() -> None:
    """Ensure all expected domains in DiscoGen are returned by get_domains()."""
    domains_list = get_domains()

    expected_domains = [
        "BayesianOptimisation",
        "BrainSpeechDetection",
        "ComputerVisionClassification",
        "ContinualLearning",
        "GreenhouseGasPrediction",
        "LanguageModelling",
        "ModelUnlearning",
        "OffPolicyRL",
        "OnPolicyMARL",
        "OnPolicyRL",
        "UnsupervisedEnvironmentDesign",
    ]

    assert sorted(domains_list) == sorted(expected_domains)
