from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18


def build_feature_extractor() -> tuple[torch.nn.Module, callable, torch.device]:
  weights = ResNet18_Weights.DEFAULT
  model = resnet18(weights=weights)
  model.fc = torch.nn.Identity()
  model.eval()
  if torch.backends.mps.is_available():
    device = torch.device("mps")    # Apple
  elif torch.cuda.is_available():
    device = torch.device("cuda")   # Nvidia
  else:
    device = torch.device("cpu")
  model.to(device)
  preprocess = weights.transforms()
  return model, preprocess, device


def extract_embeddings(
  image_paths: list[Path],
  model: torch.nn.Module,
  preprocess: callable,
  device: torch.device,
  batch_size: int = 32,
) -> np.ndarray:
  embeddings = []
  with torch.no_grad():
    for start in range(0, len(image_paths), batch_size):
      batch_paths = image_paths[start:start + batch_size]
      batch_images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
      batch_tensor = torch.stack(batch_images).to(device)
      batch_features = model(batch_tensor).cpu().numpy()
      embeddings.append(batch_features)
  return np.concatenate(embeddings, axis=0)


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int = 3) -> np.ndarray:
  predictions = []
  for vector in test_x:
    distances = np.linalg.norm(train_x - vector, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = train_y[nearest_idx]
    values, counts = np.unique(nearest_labels, return_counts=True)
    predictions.append(values[np.argmax(counts)])
  return np.array(predictions)


def malignant_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, malignant_labels: set[str], k: int = 3) -> np.ndarray:
  scores = []
  for vector in test_x:
    distances = np.linalg.norm(train_x - vector, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = train_y[nearest_idx]
    malignant_count = sum(label in malignant_labels for label in nearest_labels)
    scores.append(malignant_count / k)
  return np.array(scores, dtype=np.float32)


def build_few_shot_split(df_scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  few_shot_rows = []
  for dx, group in df_scope.groupby("dx"):
    group_sorted = group.sort_values("image_id")
    if len(group_sorted) <= 1:
      few_shot_rows.append(group_sorted.head(1))
    else:
      train_count = min(5, len(group_sorted) - 1)
      few_shot_rows.append(group_sorted.head(train_count))
  if not few_shot_rows:
    return pd.DataFrame(columns=df_scope.columns), pd.DataFrame(columns=df_scope.columns)
  df_few_shot = pd.concat(few_shot_rows).drop_duplicates(subset=["image_id"])
  remaining_ids = set(df_scope["image_id"]) - set(df_few_shot["image_id"])
  df_remaining = df_scope[df_scope["image_id"].isin(remaining_ids)].copy()
  return df_few_shot, df_remaining


def main() -> None:
  project_dir = Path(__file__).resolve().parent
  dataset_dir = project_dir / "skin-cancer-mnist-ham10000"
  selected_dir = project_dir / "selected_cropped"
  metadata_csv = dataset_dir / "HAM10000_metadata.csv"
  selected_metadata_csv = dataset_dir / "selected_metadata.csv"
  selected_used_csv = dataset_dir / "SELECTED_used.csv"
  selected_test_csv = dataset_dir / "SELECTED_test.csv"
  selected_metrics_txt = dataset_dir / "SELECTED_metrics.txt"
  selected_results_txt = dataset_dir / "SELECTED_results.txt"
  #few_shot_used_csv = dataset_dir / "few_shot_used.csv"
  #few_shot_used_localisation_csv = dataset_dir / "few_shot_used_localisation.csv"
  #few_shot_predictions_csv = dataset_dir / "few_shot_predictions.csv"

  if not metadata_csv.exists():
    raise FileNotFoundError(
      "Expected dataset file not found: "
      f"{metadata_csv}\n"
      "Make sure the dataset folder 'skin-cancer-mnist-ham10000' exists next to main.py."
    )

  if not selected_dir.exists():
    raise FileNotFoundError(
      "Expected folder not found: "
      f"{selected_dir}\n"
      "Make sure 'selected_cropped' exists next to main.py."
    )

  image_ids = sorted({p.stem for p in selected_dir.iterdir() if p.is_file()})
  if not image_ids:
    raise ValueError("No images found in selected_cropped.")

  df = pd.read_csv(metadata_csv)
  df_selected = df[df["image_id"].isin(image_ids)].copy()
  if df_selected.empty:
    raise ValueError("No matching metadata rows found for selected_cropped images.")

  df_selected["image_id"] = df_selected["image_id"].astype(str)
  df_selected = df_selected.sort_values("image_id")
  df_selected.to_csv(selected_metadata_csv, index=False)

  focus_localizations = {"hand", "upper extremity", "abdomen"}
  df_focus = df_selected[df_selected["localization"].isin(focus_localizations)].copy()
  if df_focus.empty:
    raise ValueError("No selected images found for requested localizations.")

  focus_train_rows = []
  focus_test_rows = []
  metrics_lines = ["Focused split for hand / upper extremity / abdomen"]

  for (localization, dx), group in df_focus.groupby(["localization", "dx"]):
    group_sorted = group.sort_values("image_id")
    if len(group_sorted) <= 1:
      train_count = 1
    else:
      train_count = min(3, len(group_sorted) - 1)
    train_part = group_sorted.head(train_count)
    test_part = group_sorted.iloc[train_count:]
    focus_train_rows.append(train_part)
    if not test_part.empty:
      focus_test_rows.append(test_part)
    metrics_lines.append(
      f"{localization} / {dx}: total={len(group_sorted)}, train={len(train_part)}, test={len(test_part)}"
    )

  df_selected_used = pd.concat(focus_train_rows).drop_duplicates(subset=["image_id"]) if focus_train_rows else pd.DataFrame(columns=df_selected.columns)
  df_selected_test = pd.concat(focus_test_rows).drop_duplicates(subset=["image_id"]) if focus_test_rows else pd.DataFrame(columns=df_selected.columns)
  df_selected_used.to_csv(selected_used_csv, index=False)
  df_selected_test.to_csv(selected_test_csv, index=False)
  selected_metrics_txt.write_text("\n".join(metrics_lines), encoding="utf-8")

  if df_selected_used.empty or df_selected_test.empty:
    selected_results_txt.write_text("No test data available for focused localizations.", encoding="utf-8")
    return

  def resolve_image_path(image_id: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
      candidate = selected_dir / f"{image_id}{ext}"
      if candidate.exists():
        return candidate
    candidate = selected_dir / image_id
    if candidate.exists():
      return candidate
    raise FileNotFoundError(f"Image file not found for image_id: {image_id}")

  model, preprocess, device = build_feature_extractor()

  focus_paths = [resolve_image_path(image_id) for image_id in df_focus["image_id"].tolist()]
  focus_embeddings = extract_embeddings(focus_paths, model, preprocess, device)
  embedding_map = {
    image_id: embedding
    for image_id, embedding in zip(df_focus["image_id"].tolist(), focus_embeddings)
  }

  train_features = np.stack([embedding_map[image_id] for image_id in df_selected_used["image_id"].tolist()])
  train_labels = df_selected_used["dx"].to_numpy()

  test_features = np.stack([embedding_map[image_id] for image_id in df_selected_test["image_id"].tolist()])
  test_labels = df_selected_test["dx"].to_numpy()

  k_neighbors = 3
  predictions = predict_knn(train_features, train_labels, test_features, k=k_neighbors)
  correct = predictions == test_labels
  accuracy = float(np.mean(correct)) if len(correct) else 0.0

  malignant_labels = {"mel", "bcc", "akiec"}
  malignant_true = pd.Series(test_labels).isin(malignant_labels)
  malignant_score = malignant_scores(train_features, train_labels, test_features, malignant_labels, k=k_neighbors)
  if malignant_true.nunique() > 1:
    auc = roc_auc_score(malignant_true, malignant_score)
    auc_text = f"{auc:.4f}"
  else:
    auc_text = "n/a"

  results_lines = []
  results_lines.append("Focused evaluation for hand / upper extremity / abdomen")
  results_lines.append(f"Train images: {len(df_selected_used)}")
  results_lines.append(f"Test images: {len(df_selected_test)}")
  results_lines.append(f"Accuracy: {accuracy:.4f}")
  results_lines.append(f"AUC malignant vs benign: {auc_text}")
  results_lines.append("Per-localization: ")
  for localization, group in df_focus.groupby("localization"):
    loc_train = df_selected_used[df_selected_used["localization"] == localization]
    loc_test = df_selected_test[df_selected_test["localization"] == localization]
    if loc_train.empty or loc_test.empty:
      results_lines.append(f"  {localization}: train={len(loc_train)}, test={len(loc_test)}, accuracy=n/a, auc=n/a")
      continue
    loc_train_features = np.stack([embedding_map[image_id] for image_id in loc_train["image_id"].tolist()])
    loc_train_labels = loc_train["dx"].to_numpy()
    loc_test_features = np.stack([embedding_map[image_id] for image_id in loc_test["image_id"].tolist()])
    loc_test_labels = loc_test["dx"].to_numpy()
    loc_predictions = predict_knn(loc_train_features, loc_train_labels, loc_test_features, k=k_neighbors)
    loc_accuracy = float(np.mean(loc_predictions == loc_test_labels)) if len(loc_test_labels) else 0.0
    loc_malignant_true = pd.Series(loc_test_labels).isin(malignant_labels)
    loc_malignant_score = malignant_scores(
      loc_train_features,
      loc_train_labels,
      loc_test_features,
      malignant_labels,
      k=k_neighbors,
    )
    if loc_malignant_true.nunique() > 1:
      loc_auc = roc_auc_score(loc_malignant_true, loc_malignant_score)
      loc_auc_text = f"{loc_auc:.4f}"
    else:
      loc_auc_text = "n/a"
    results_lines.append(
      f"  {localization}: train={len(loc_train)}, test={len(loc_test)}, accuracy={loc_accuracy:.4f}, auc={loc_auc_text}"
    )

  results_lines.append("Split counts per localization and dx:")
  results_lines.extend(metrics_lines[1:])

  selected_results_txt.write_text("\n".join(results_lines), encoding="utf-8")
  for line in results_lines:
    print(line)


if __name__ == "__main__":
  main()