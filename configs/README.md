# Configuration and manifests

`example.json` is a starting point for one source-domain training run. All relative paths in a config are resolved from the directory containing that config. The example therefore expects private images and masks under `data/` at the repository root and writes checkpoints under `runs/example/`.

The repository retains 21 CSV indexes but no medical images, pixel masks, or pretrained checkpoints. For explicit-manifest mode, prepare your own data and create separate CSV files for training, validation, and testing. For an unseen-domain evaluation, point `test_manifest` at the target domain and keep it separate from the source-domain training and validation sets.

## Using the repository's indexed CSVs

`python -m orthoseg` remains the public training and evaluation entry point. `orthoseg/indexed_data.py` implements the index adapter using the retained CSVs in `Datasets/BioMedicalDataset/`; no earlier Python dataset loaders are retained there. This mode requires a source checkout with those CSVs. A wheel installation alone does not include the indexes.

`example_indexed.json` shows an alternative to writing new manifests. It selects the existing ISIC2018 training and validation indexes and the PH2 test index:

```json
"indexed_datasets": {"train": "ISIC2018", "val": "ISIC2018", "test": "PH2"}
```

Set `data.indexed_metadata_root` to this repository's `Datasets/BioMedicalDataset` directory, which contains the retained CSV indexes. Set `data.indexed_data_root` to **your own** `BioMedicalDataset` image tree with the corresponding dataset subdirectories. The example image-tree path is illustrative; the repository does not contain the image files. Paths in both settings are resolved from the config file's directory.

When `python -m orthoseg train --config configs/example_indexed.json` is invoked, the adapter checks each selected index and its image/mask files, then writes normalized CSV sidecars under `train.output_dir/indexed_manifests/<dataset>_<split>/<split>_manifest.csv`. DSB2018 instance masks, when selected, are combined into derived binary masks in that sidecar directory. The source indexes, images, and masks are left unchanged. The generated manifests still pass through the same `ManifestDataset` loader and train/validation/test overlap checks as explicit manifests.

Indexed mode accepts a subset of `train`, `val`, and `test` entries, so a target-only evaluation config may name just `test`. For training, `train` and `val` must name the same source dataset. Do not combine `data.indexed_datasets` with `data.root` or `train_manifest`, `val_manifest`, or `test_manifest`. The indexed roots are meaningful only with `indexed_datasets`. For a new dataset or a custom split, use explicit manifests instead.

The adapter accepts these exact dataset names and preserved splits:

| Dataset name | Available splits |
| --- | --- |
| `BUSI`, `DSB2018`, `ISIC2018` | train, val, test |
| `COVID19` | train, test |
| `CVC-ClinicDB` | train, test |
| `Kvasir-SEG`, `PolypSegData-pooled` | train |
| `COVID19_2`, `MonuSeg2018`, `PH2`, `STU`, `CVC-300`, `CVC-ColonDB`, `ETIS-LaribPolypDB`, `Kvasir` | test |

The preserved BUSI validation and BUSI test indexes overlap. The BUSI train/val → STU test cross-domain configuration uses no BUSI test index. If you also evaluate BUSI test in the same run, the overlap check rejects it; prepare a separate source-domain holdout for that experiment.

The retained `PolypSegData/train_frame.csv` pools 550 CVC-ClinicDB and 900 Kvasir images (1,450 rows). The adapter exposes the source subsets as `CVC-ClinicDB`/`train` and `Kvasir-SEG`/`train`, while the explicit `PolypSegData-pooled`/`train` name selects the pooled index. None has a preserved polyp validation split. The paper does not clarify the `SD6 + SD7` source protocol for Table IV, so selecting the pooled index does not establish that the paper used it. COVID19 also has no preserved source validation split. To train either source, make disjoint train/validation manifests from source data and use explicit-manifest mode; never choose checkpoints on a target test set. The retained indexes have no REFUGE or Drishti-GS metadata, so fundus OD/OC experiments require explicit manifests. See errors from the adapter for unsupported dataset/split combinations.

## Manifest format

For `data.task: "binary"`, each CSV has these columns:

```csv
image_path,mask_path
images/case_001.png,masks/case_001.png
```

For `data.task: "fundus"`, provide separate optic-disc and optic-cup masks. The two classes are represented by two binary channels; the optic-disc mask must include the cup region:

```csv
image_path,od_mask_path,oc_mask_path
images/case_001.png,od_masks/case_001.png,oc_masks/case_001.png
```

Manifest image and mask paths may be absolute or relative to `data.root`. Each mask must align spatially with its image. Masks are loaded as grayscale; by default, original pixel values greater than zero are foreground. Both `0/1` and `0/255` masks are supported. The paths above illustrate the format; no sample images are bundled.

## Source-to-target protocol in the manuscript

Tables I and II define the source and unseen datasets. Table IV reports these corresponding cross-dataset evaluations:

| Source dataset | Unseen test dataset | Modality |
| --- | --- | --- |
| DSB-2018 (SD1) | MonuSeg2018 (UD1) | Microscopy |
| BUSI (SD2) | STU (UD2) | Ultrasound |
| ISIC2018 (SD3) | PH2 (UD3) | Dermoscopy |
| COVID19-1 (SD4) | COVID19-2 (UD4) | Radiology |
| REFUGE (SD5) | Drishti-GS (UD5), optic disc and cup | Fundus |
| CVC-ClinicDB (SD6), Kvasir-SEG (SD7) | CVC-300 (UD6), CVC-ColonDB (UD7), ETIS (UD8) | Colonoscopy |

For the colonoscopy targets, Table IV labels the source `SD6 + SD7`, while the method and implementation text describes independent training for each source domain. The manuscript does not resolve whether those two sources were pooled or evaluated through another protocol. Reproducing that row requires confirmation from the original experiment records; this reference implementation does not merge them automatically.

## Main settings

| Key | Meaning |
| --- | --- |
| `data.root` | Base directory for relative paths in the CSV manifests. |
| `data.train_manifest`, `val_manifest`, `test_manifest` | CSV files for the three splits. Set each run to one source dataset; change the test CSV for a corresponding unseen dataset. |
| `data.indexed_datasets` | Alternative split-to-dataset mapping for the retained CSV indexes, for example `{"train":"ISIC2018","val":"ISIC2018","test":"PH2"}`. |
| `data.indexed_data_root`, `indexed_metadata_root` | Private image-tree root and repository CSV-index root, respectively; used only in indexed mode. |
| `data.task` | `binary` for one target class or `fundus` for optic disc and cup. |
| `data.image_size` | An integer for square input or `[height, width]`. The paper uses 352 × 352. |
| `data.spacing` | `null` reports HD95 in pixels. A non-null `[row_spacing, column_spacing]` is a constant mm/pixel spacing on the resized model-output grid. Do not set it together with per-image manifest spacing. |
| `model.in_channels` | Keep at `3`: the current loader converts all images to RGB. |
| `model.num_classes` | `1` for binary or `2` for fundus. |
| `model.widths` | Structural and texture encoder feature widths for this reference implementation. |
| `model.latent_dim` | MID projection dimension; the paper specifies 256. |
| `model.num_experts` | CSTA kernel expert count; the paper specifies 4. |
| `train.epochs`, `batch_size`, `lr` | Main training schedule. The example uses the paper's 200 epochs, batch size 16, and initial learning rate 0.0001. |
| `train.aux_lr` | Learning rate for the MID auxiliary conditional model. |
| `train.alpha`, `beta` | Structural equivariance and texture invariance loss weights. The paper's Fig. 9 reports the selected values 0.3 and 1.0. |
| `train.tau`, `gamma` | MID weight schedule parameters. The paper gives 1.0 and 0.9, respectively. |
| `train.seed`, `num_workers` | Reproducibility seed and data-loader worker count. |
| `train.output_dir` | Checkpoint, history, and evaluation output directory. |

Change `model.num_classes` to `2` when using `data.task: "fundus"`. The example widths, optimizer weight decay, and auxiliary learning rate are reference defaults; they should not be read as undocumented settings recovered from the paper.

## HD95 spacing

With `data.spacing: null` and no manifest spacing columns, HD95 is reported in pixels. For calibrated images, choose one of these inputs:

- Add both `spacing_y_mm,spacing_x_mm` columns to each manifest. They describe the original image in mm/pixel. For an image resized from `H × W` to `h × w`, the loader uses `[spacing_y_mm × H / h, spacing_x_mm × W / w]` on the output grid.
- Set `data.spacing: [row_mm_per_pixel, column_mm_per_pixel]` only when the same known spacing already applies to every resized model output. These values are **not** original-image spacing.

The code rejects configurations that supply both sources of spacing. Leave both unspecified when trustworthy physical calibration is unavailable; image file dimensions alone do not establish mm/pixel.
