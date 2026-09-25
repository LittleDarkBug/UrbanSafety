import json

import numpy as np
import torch

from urbansafety.data.annotations import clip_labels, is_anomaly, load_all, load_cadp, load_dota
from urbansafety.data.bdd100k import format_for_slm, parse_metadata
from urbansafety.data.features import FeatureSequenceDataset, collate_padded
from urbansafety.evaluation import metrics


def test_clip_labels_mark_only_segments_overlapping_the_anomaly():
    labels = clip_labels({"Robbery001_x264": {"duration": 30, "timestamps": [[12, 17]]}})
    assert labels["Robbery001_x264"].tolist() == [0, 0, 1, 1, 0, 0]


def test_clip_labels_normal_video_is_all_zero_and_unstamped_anomaly_all_one():
    labels = clip_labels({
        "Normal_Videos001_x264": {"duration": 20, "timestamps": [[0, 5]]},
        "Fighting002_x264": {"duration": 10, "timestamps": []},
    })
    assert labels["Normal_Videos001_x264"].sum() == 0
    assert labels["Fighting002_x264"].tolist() == [1, 1]


def test_is_anomaly_follows_each_dataset_convention():
    assert is_anomaly("Abuse003_x264", {"dataset": "ucf_crime"})
    assert not is_anomaly("Normal_Videos050_x264", {"dataset": "ucf_crime"})
    assert is_anomaly("0001", {"dataset": "dota"})
    assert not is_anomaly("cadp_x", {"dataset": "cadp", "is_anomaly": False})


def test_cadp_frames_are_grouped_into_ordered_pseudo_videos(tmp_path):
    d = tmp_path / "train" / "Accident"
    d.mkdir(parents=True)
    for n in (10, 2, 1):
        (d / f"clip7 ({n}).jpg").write_bytes(b"")
    anns = load_cadp(tmp_path)
    (vid, ann), = anns.items()
    assert vid == "cadp_train_Accident_clip7"
    assert [p.split("(")[1] for p in ann["frames"]] == ["1).jpg", "2).jpg", "10).jpg"]
    assert ann["is_anomaly"] and ann["timestamps"] == [[0, 3 / 30]]


def test_dota_metadata_becomes_timestamped_anomaly(tmp_path):
    (tmp_path / "metadata_train.json").write_text(json.dumps(
        {"v1": {"num_frames": 150, "anomaly_start": 60, "anomaly_end": 90, "anomaly_class": "LA"}}))
    ann = load_dota(tmp_path)["v1"]
    assert ann["duration"] == 5 and ann["timestamps"] == [[2, 3]]
    assert ann["sentences"] == ["Traffic surveillance showing lateral collision"]


def test_load_all_splits_by_dataset(tmp_path):
    ucf = tmp_path / "ucf"
    ucf.mkdir()
    (ucf / "UCFCrime_Train.json").write_text(json.dumps({"Abuse001_x264": {"duration": 10}}))
    (ucf / "UCFCrime_Test.json").write_text(json.dumps({"Normal_Videos001_x264": {"duration": 10}}))

    class P:
        enabled = {"ucf_crime": True, "dota": False, "cadp": False}

    P.ucf = ucf
    splits = load_all(P)
    assert list(splits["train"]) == ["Abuse001_x264"] and list(splits["test"]) == ["Normal_Videos001_x264"]


def test_bdd100k_metadata_is_summarized_for_the_language_model():
    image = {"name": "a.jpg", "attributes": {"weather": "rainy", "scene": "highway", "timeofday": "night"},
             "labels": [{"category": "car"}, {"category": "car"}, {"category": "pedestrian"},
                        {"category": "traffic light", "attributes": {"trafficLightColor": "red"}},
                        {"category": "lane", "attributes": {"laneType": "crosswalk"}}]}
    text = format_for_slm(parse_metadata(image))
    assert text.splitlines() == ["Weather: rainy", "Scene: highway", "Time: night",
                                 "Objects: 2 cars, 1 pedestrian", "Traffic lights: 1 red", "Lane markings: crosswalk"]


def test_feature_dataset_filters_split_and_dataset_and_pads_labels(tmp_path):
    torch.save(torch.randn(4, 3), tmp_path / "Abuse001_x264.pt")
    torch.save(torch.randn(2, 3), tmp_path / "Normal_Videos001_x264.pt")
    torch.save(torch.randn(2, 3), tmp_path / "cadp_train_Accident_c.pt")
    anns = {"Abuse001_x264": {"split": "train", "dataset": "ucf_crime"},
            "Normal_Videos001_x264": {"split": "train", "dataset": "ucf_crime"},
            "cadp_train_Accident_c": {"split": "train", "dataset": "cadp", "is_anomaly": True}}
    ds = FeatureSequenceDataset(tmp_path, anns, "train", ("ucf_crime",),
                                clip_labels={"Abuse001_x264": torch.tensor([0.0, 1.0])})
    assert len(ds) == 2
    x, y = collate_padded([ds[0], ds[1]])
    assert x.shape == (2, 4, 3) and y.tolist() == [[0, 1, 0, 0], [0, 0, 0, 0]]


def test_metrics_match_a_hand_computed_confusion_matrix():
    m = metrics(np.array([0.9, 0.6, 0.3, 0.1]), np.array([1, 0, 1, 0]), threshold=0.5)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == m["recall"] == 0.5 and m["auc"] == 0.75
