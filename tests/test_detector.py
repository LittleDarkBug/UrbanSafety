import torch

from urbansafety.models.detector import AnomalyDetector, PGDAttack, ranking_hinge_loss


def test_detector_scores_every_segment_in_unit_interval():
    model = AnomalyDetector(input_dim=16, hidden_dim=8)
    scores = model(torch.randn(3, 7, 16))
    assert scores.shape == (3, 7, 1)
    assert torch.all((scores >= 0) & (scores <= 1))


def test_ranking_loss_is_zero_when_anomalous_bags_outrank_normal_ones_by_the_margin():
    scores = torch.tensor([[[0.9], [0.1]], [[0.2], [0.1]]])  # sac anormal max 0,9 ; sac normal max 0,2
    assert ranking_hinge_loss(scores, torch.tensor([1.0, 0.0]), margin=0.1).item() == 0.0


def test_ranking_loss_penalizes_a_normal_bag_scoring_above_an_anomalous_one():
    scores = torch.tensor([[[0.3]], [[0.8]]])
    loss = ranking_hinge_loss(scores, torch.tensor([1.0, 0.0]), margin=0.1)
    assert abs(loss.item() - (0.1 - 0.3 + 0.8)) < 1e-6


def test_ranking_loss_on_single_class_batch_is_zero_but_differentiable():
    model = AnomalyDetector(input_dim=4, hidden_dim=4)
    loss = ranking_hinge_loss(model(torch.randn(2, 3, 4)), torch.tensor([1.0, 1.0]))
    loss.backward()
    assert loss.item() == 0.0


def test_pgd_perturbation_stays_within_epsilon_and_leaves_model_gradients_untouched():
    torch.manual_seed(0)
    model = AnomalyDetector(input_dim=8, hidden_dim=4)
    x = torch.randn(2, 5, 8)
    delta = PGDAttack(model, epsilon=0.01, steps=3, alpha=0.004).perturb(x, torch.ones(2, 5))
    assert delta.shape == x.shape
    assert delta.abs().max().item() <= 0.01 + 1e-7
    assert all(p.grad is None for p in model.parameters())


def test_pgd_perturbation_increases_the_loss():
    torch.manual_seed(0)
    model = AnomalyDetector(input_dim=8, hidden_dim=4, dropout=0.0).eval()
    x, y = torch.randn(4, 6, 8), torch.ones(4, 6)
    bce = lambda inp: torch.nn.functional.binary_cross_entropy(model(inp).view(-1), y.view(-1)).item()
    delta = PGDAttack(model, epsilon=0.05, steps=5, alpha=0.02).perturb(x, y)
    assert bce(x + delta) > bce(x)
