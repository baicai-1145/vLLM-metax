from tools.debug.characterize_c500_p2p import _repetitions


def test_repetitions_limits_large_transfers():
    assert _repetitions(4096, 100) == 100
    assert _repetitions(1024 * 1024, 100) == 50
    assert _repetitions(64 * 1024 * 1024, 100) == 20
