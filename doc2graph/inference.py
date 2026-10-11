import torch
from torch.nn import functional as F
import os
from pathlib import Path
from PIL import Image, ImageDraw
import json

from .data.feature_builder import FeatureBuilder
from .data.graph_builder import GraphBuilder
from .data.multipage import build_cross_page_links
from .data.preprocessing import center
from .models.graphs import SetModel
from .paths import CHECKPOINTS, INFERENCE
from .training.utils import get_device

pretrain = {
    "funsd": {
        "node_num_classes": 4,
        "edge_num_classes": 2,
        # Document2Graph sorts label names with numpy.unique before training.
        "node_labels": ("answer", "header", "other", "question"),
    },
    "pau": {"node_num_classes": 5, "edge_num_classes": 2},
}


def inference(weights, paths, device=-1, multipage=False):
    INFERENCE.mkdir(parents=True, exist_ok=True)
    if not weights:
        raise ValueError("Provide a checkpoint with --weights.")
    if not paths:
        raise ValueError("Provide one or more image files with --docs.")
    if multipage and len(paths) < 2:
        raise ValueError("Multipage inference requires at least two page images.")

    missing_paths = [str(path) for path in paths if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Input image file(s) not found: " + ", ".join(missing_paths)
        )

    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read '{path}' as an image file.") from exc

    checkpoint = CHECKPOINTS / weights[0]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        model_name, dataset_name, *_ = weights[0].split("-")
        info = pretrain[dataset_name]
    except (ValueError, KeyError) as exc:
        raise ValueError(
            "Checkpoint name must identify a supported model and dataset, "
            "for example e2e-funsd-best.pt."
        ) from exc

    # create a graph per each file
    print("Doc2Graph Inference:")
    device = get_device(device)

    print("-> Creating graphs ...")
    gb = GraphBuilder()
    graphs, _, _, features = gb.get_graph(paths, "CUSTOM")
    empty_pages = [
        str(path) for path, page_texts in zip(paths, features["texts"]) if not page_texts
    ]
    if empty_pages:
        raise ValueError(
            "OCR found no text in image(s): " + ", ".join(empty_pages)
        )

    # add embedded visual, text, layout etc. features to the graphs
    print("-> Creating features ...")
    fb = FeatureBuilder(d=device)
    chunks, _ = fb.add_features(graphs, features)

    document_graph = None
    page_links = []
    if multipage:
        document_graph = {
            "pages": [os.path.basename(path) for path in paths],
            "nodes": [
                {
                    "page": page_id,
                    "node": node_id,
                    "text": text,
                    "box": box,
                }
                for page_id, page_texts in enumerate(features["texts"])
                for node_id, (text, box) in enumerate(
                    zip(page_texts, features["boxs"][page_id])
                )
            ],
            "page_links": page_links,
            "cross_page_links": build_cross_page_links(features["texts"]),
        }

    # create the model
    print("-> Creating model ...")
    pre = dataset_name
    sm = SetModel(name=model_name, device=device)
    model = sm.get_model(
        info["node_num_classes"], info["edge_num_classes"], chunks, False
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # predict on graphs
    print("Predicting:")
    with torch.no_grad():
        for num, graph in enumerate(graphs):
            _, name = os.path.split(paths[num])
            name = name.split(".")[0]
            print(f" -> {name}")

            n, e = model(graph.to(device), graph.ndata["feat"].to(device))
            node_probs = F.softmax(n, dim=1)
            edge_probs = F.softmax(e, dim=1)
            npreds = node_probs.argmax(dim=1)
            epreds = edge_probs.argmax(dim=1)

            # save results
            links = (epreds == 1).nonzero(as_tuple=True)[0].tolist()
            u, v = graph.edges()
            entities = features["boxs"][num]
            contents = features["texts"][num]

            graph_img = Image.open(paths[num]).convert("RGB")
            graph_draw = ImageDraw.Draw(graph_img)

            result = []
            for i, idx in enumerate(links):
                source_id = int(u[idx])
                target_id = int(v[idx])
                if pre == "funsd":
                    source_label = info["node_labels"][int(npreds[source_id])]
                    target_label = info["node_labels"][int(npreds[target_id])]
                    # FUNSD key-value links are directed question -> answer.
                    # The pretrained model also predicts node roles; use them
                    # to discard edge positives with incompatible endpoints.
                    if source_label != "question" or target_label != "answer":
                        continue
                    pair = {
                        "key": {"text": contents[source_id], "box": entities[source_id]},
                        "value": {"text": contents[target_id], "box": entities[target_id]},
                        "confidence": float(edge_probs[idx, 1]),
                        "key_confidence": float(node_probs[source_id, npreds[source_id]]),
                        "value_confidence": float(node_probs[target_id, npreds[target_id]]),
                    }
                else:
                    pair = {
                        "source": {"text": contents[source_id], "box": entities[source_id]},
                        "target": {"text": contents[target_id], "box": entities[target_id]},
                        "confidence": float(edge_probs[idx, 1]),
                    }
                result.append(pair)
                if multipage:
                    page_links.append(
                        {
                            "source": {"page": num, "node": source_id},
                            "target": {"page": num, "node": target_id},
                            "type": "key_value" if pre == "funsd" else "model_edge",
                            "confidence": float(edge_probs[idx, 1]),
                        }
                    )

                key_center = center(entities[source_id])
                value_center = center(entities[target_id])
                graph_draw.line((key_center, value_center), fill="violet", width=3)
                graph_draw.ellipse(
                    [
                        (key_center[0] - 4, key_center[1] - 4),
                        (key_center[0] + 4, key_center[1] + 4),
                    ],
                    fill="green",
                    outline="black",
                )
                graph_draw.ellipse(
                    [
                        (value_center[0] - 4, value_center[1] - 4),
                        (value_center[0] + 4, value_center[1] + 4),
                    ],
                    fill="red",
                    outline="black",
                )

            graph_img.save(INFERENCE / f"{name}.png")
            with open(INFERENCE / f"{name}.json", "w") as outfile:
                json.dump(result, outfile)

    if document_graph is not None:
        with open(INFERENCE / "document_graph.json", "w") as outfile:
            json.dump(document_graph, outfile, indent=2)
        print(
            f"-> Wrote {len(page_links)} page-level predictions and "
            f"{len(document_graph['cross_page_links'])} cross-page text matches "
            "to document_graph.json"
        )
