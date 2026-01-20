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
  few_shot_used_csv = dataset_dir / "few_shot_used.csv"
  few_shot_used_localisation_csv = dataset_dir / "few_shot_used_localisation.csv"
  few_shot_predictions_csv = dataset_dir / "few_shot_predictions.csv"

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

  df_few_shot, df_remaining = build_few_shot_split(df_selected)
  df_few_shot.to_csv(few_shot_used_csv, index=False)

  localisation_few_shot_rows = []
  for _, group in df_selected.groupby("localization"):
    df_loc_few_shot, _ = build_few_shot_split(group)
    if not df_loc_few_shot.empty:
      localisation_few_shot_rows.append(df_loc_few_shot)
  if localisation_few_shot_rows:
    df_few_shot_localisation = pd.concat(localisation_few_shot_rows).drop_duplicates(subset=["image_id"])
  else:
    df_few_shot_localisation = pd.DataFrame(columns=df_selected.columns)
  df_few_shot_localisation.to_csv(few_shot_used_localisation_csv, index=False)

  if df_remaining.empty:
    print("Few-shot set covers all selected images. No remaining images to predict.")
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

  all_paths = [resolve_image_path(image_id) for image_id in df_selected["image_id"].tolist()]
  all_embeddings = extract_embeddings(all_paths, model, preprocess, device)
  embedding_map = {
    image_id: embedding
    for image_id, embedding in zip(df_selected["image_id"].tolist(), all_embeddings)
  }

  train_features = np.stack([embedding_map[image_id] for image_id in df_few_shot["image_id"].tolist()])
  train_labels = df_few_shot["dx"].to_numpy()

  test_features = np.stack([embedding_map[image_id] for image_id in df_remaining["image_id"].tolist()])
  test_labels = df_remaining["dx"].to_numpy()

  k_neighbors = 3
  predictions = predict_knn(train_features, train_labels, test_features, k=k_neighbors)
  df_predictions = df_remaining.copy()
  df_predictions["y_true"] = test_labels
  df_predictions["y_pred"] = predictions
  df_predictions["correct"] = df_predictions["y_true"] == df_predictions["y_pred"]

  malignant_labels = {"mel", "bcc", "akiec"}
  df_predictions["malignant_true"] = df_predictions["y_true"].isin(malignant_labels)
  malignant_score = malignant_scores(train_features, train_labels, test_features, malignant_labels, k=k_neighbors)
  df_predictions["malignant_score"] = malignant_score
  if df_predictions["malignant_true"].nunique() > 1:
    auc = roc_auc_score(df_predictions["malignant_true"], df_predictions["malignant_score"])
  else:
    auc = float("nan")
  df_predictions.to_csv(few_shot_predictions_csv, index=False)

  accuracy = float(np.mean(df_predictions["correct"])) if len(df_predictions) else 0.0
  print("Few-shot samples:", len(df_few_shot))
  print("Predictions generated:", len(df_predictions))
  print("Accuracy:", round(accuracy, 4))
  print("AUC malignant vs benign:", round(auc, 4) if not np.isnan(auc) else "n/a")
  metrics_lines = []
  metrics_lines.append(f"Overall train images: {len(df_few_shot)}")
  metrics_lines.append(f"Overall test images: {len(df_predictions)}")
  metrics_lines.append(f"Overall accuracy: {accuracy:.4f}")
  metrics_lines.append(f"Overall AUC malignant vs benign: {auc:.4f}" if not np.isnan(auc) else "Overall AUC malignant vs benign: n/a")
  metrics_lines.append("Per-localization metrics:")

  print("Per-localization metrics:")
  for localization, group in df_selected.groupby("localization"):
    df_loc_few_shot, df_loc_remaining = build_few_shot_split(group)
    if df_loc_few_shot.empty or df_loc_remaining.empty:
      line = f"  {localization}: train={len(df_loc_few_shot)}, test={len(df_loc_remaining)}, accuracy=n/a, auc=n/a"
      print(line)
      metrics_lines.append(line)
      continue
    loc_train_features = np.stack([embedding_map[image_id] for image_id in df_loc_few_shot["image_id"].tolist()])
    loc_train_labels = df_loc_few_shot["dx"].to_numpy()
    loc_test_features = np.stack([embedding_map[image_id] for image_id in df_loc_remaining["image_id"].tolist()])
    loc_test_labels = df_loc_remaining["dx"].to_numpy()

    loc_predictions = predict_knn(loc_train_features, loc_train_labels, loc_test_features, k=k_neighbors)
    loc_correct = loc_predictions == loc_test_labels
    loc_accuracy = float(np.mean(loc_correct)) if len(loc_correct) else 0.0

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

    line = (
      f"  {localization}: train={len(df_loc_few_shot)}, "
      f"test={len(df_loc_remaining)}, "
      f"accuracy={loc_accuracy:.4f}, auc={loc_auc_text}"
    )
    print(line)
    metrics_lines.append(line)

  metrics_path = dataset_dir / "few_shot_metrics.txt"
  metrics_path.write_text("\n".join(metrics_lines), encoding="utf-8")


if __name__ == "__main__":
  main()