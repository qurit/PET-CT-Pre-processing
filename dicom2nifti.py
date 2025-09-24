from pathlib import Path
import os, platform, dateutil.parser
import pydicom, SimpleITK as sitk
import pandas as pd
import json
import csv, datetime
import numpy as np
from dotenv import load_dotenv

load_dotenv()

target_spacing = (4.07283, 4.07283, 4.07283)

ROOT_DIR = Path(os.environ["ROOT_DIR"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
MASK_DIR = Path(os.environ["MASK_DIR"])
MASK_SAVING_DIR = Path(os.environ["MASK_SAVING_DIR"])

LOG_PATH = OUTPUT_DIR / "conversion_log.csv"

if not LOG_PATH.exists():
    with LOG_PATH.open("w", newline="") as f:
        csv.writer(f).writerow(
            [
                "timestamp",
                "patient",
                "modality",
                "orig_size",
                "new_size",
                "orig_spacing",
                "new_spacing",
                "orig_origin",
                "new_origin",
                "file_path",
            ]
        )


def _v(vec):
    """helper - JSON-encode a tuple so it sits in one cell"""
    return json.dumps(list(vec))  # e.g. "[512,512,1056]"


def log_image(
    patient: str, modality: str, img_in: sitk.Image, img_out: sitk.Image, out_path: Path
):
    with LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.datetime.now().isoformat(timespec="seconds"),
                patient,
                modality,
                _v(img_in.GetSize()),
                _v(img_out.GetSize()),
                _v(img_in.GetSpacing()),
                _v(img_out.GetSpacing()),
                _v(img_in.GetOrigin()),
                _v(img_out.GetOrigin()),
                str(out_path),
            ]
        )


SEGS_LOG_PATH = OUTPUT_DIR / "segments_log.csv"
if not SEGS_LOG_PATH.exists():
    with SEGS_LOG_PATH.open("w", newline="") as f:
        csv.writer(f).writerow(
            [
                "timestamp",
                "patient",
                "seg_path",
                "num_components",
                "component_names",
                "component_nonzero_voxels",
            ]
        )


def _seg_names(seg_img: sitk.Image):
    names = []
    C = seg_img.GetNumberOfComponentsPerPixel()
    for i in range(C):
        key = f"Segment{i}_Name"
        names.append(
            seg_img.GetMetaData(key) if seg_img.HasMetaDataKey(key) else f"seg_{i+1}"
        )
    return names


def _log_segments(patient: str, seg_path: Path, C: int, names, counts):
    with SEGS_LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.datetime.now().isoformat(timespec="seconds"),
                patient,
                str(seg_path),
                C,
                json.dumps(names),
                json.dumps(counts),
            ]
        )


########################################################################
# ----------  helpers --------------------------------------------------
########################################################################


def clean_series(
    series_dir: Path,
):  # helper function to remove duplicate slices and address the warning "Non uniform sampling or missing slices detected"
    rows = []
    for f in sorted(series_dir.glob("*.IMA")):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        z = round(ds.ImagePositionPatient[2], 3)  # mm, round to avoid FP noise
        rows.append((f, ds.InstanceNumber, z))
    df = pd.DataFrame(rows, columns=["file", "instance", "z"])
    df.sort_values("z", ascending=True, inplace=True)

    # keep the *first* slice at each z-location
    dup_mask = df.duplicated(subset="z", keep="first")
    dups = df[dup_mask]
    if len(dups):
        print("Removing duplicates...")
        return list(str(path) for path in df.loc[~dup_mask, "file"].tolist())
    else:
        print("No duplicates found.")
        return list(str(path) for path in df["file"].tolist())


def enable_long_path(p: Path) -> Path:
    """Add the \\\\?\\ prefix on Windows so >260-char paths work."""
    p = p.resolve()
    if platform.system() == "Windows" and not str(p).startswith(r"\\?\\"):
        return Path(rf"\\?\{p}")
    return p


def convert_mask(seg_path: Path, pet_iso: sitk.Image, out_dir: Path, patient_name: str):
    """
    Turn a .seg.nrrd into a 0/1 mask aligned with the PET grid.
    """
    seg_img = sitk.ReadImage(str(seg_path))  # preserves geometry
    C = seg_img.GetNumberOfComponentsPerPixel()

    # Collect per-component masks (in SEG space) for logging and union
    masks = []
    if C > 1:
        for i in range(C):
            m = sitk.Cast(
                sitk.VectorIndexSelectionCast(seg_img, i, sitk.sitkUInt8) > 0,
                sitk.sitkUInt8,
            )
            masks.append(m)
    else:
        masks = [sitk.Cast(seg_img > 0, sitk.sitkUInt8)]

    # Log component metadata (names + voxel counts)
    names = _seg_names(seg_img) if C > 1 else ["seg_1"]
    counts = []
    for m in masks:
        # Fast count without full numpy copy:
        counts.append(int(sitk.GetArrayViewFromImage(m).sum()))
    _log_segments(patient_name, seg_path, C, names, counts)

    # Union in SEG space
    union = masks[0]
    for m in masks[1:]:
        union = sitk.Or(union, m)

    # --- resample onto PET grid --------------------------------------------
    mask_on_pet = sitk.Resample(
        union,
        pet_iso,  # reference = PET SUV grid
        sitk.Transform(),  # identity (frames already match)
        sitk.sitkNearestNeighbor,  # <-- keep labels crisp
        0,  # background value
    )

    # --- save ---------------------------------------------------------------
    out_path = out_dir / f"{patient_name}_MASK.nii.gz"
    sitk.WriteImage(
        mask_on_pet, str(out_path), imageIO="NiftiImageIO", useCompression=True
    )
    log_image(patient_name, "MASK", union, mask_on_pet, out_path)
    print("✓ MASK →", out_path)


def suv_conversion_factors(ds: pydicom.FileDataset):
    """Return (SUVfactor, slope, intercept) from the PET header."""
    nuclide = ds[(0x0054, 0x0016)][0]
    inj_dose = nuclide[
        (0x0018, 0x1074)
    ].value  # Bq  :contentReference[oaicite:0]{index=0}
    half_life = float(nuclide[(0x0018, 0x1075)].value)  # s
    weight_kg = ds[(0x0010, 0x1030)].value  # kg

    study_date = str(ds[(0x0008, 0x0021)].value)
    series_time = str(ds[(0x0008, 0x0031)].value)
    inj_time = str(nuclide[(0x0018, 0x1072)].value)
    parse = dateutil.parser.parse

    series_dt = parse(f"{study_date} {series_time}")
    inj_dt = parse(f"{study_date} {inj_time}")
    decay = 2 ** (-(series_dt - inj_dt).total_seconds() / half_life)
    suv_fact = (weight_kg * 1_000) / (decay * inj_dose)
    slope = getattr(ds, "RescaleSlope", 1)  # use defaults if missing
    intercept = getattr(ds, "RescaleIntercept", 0)
    return suv_fact, slope, intercept


def sitk_from_series(series_dir: Path):
    """Read a DICOM series; fall back to glob if SeriesUIDs are missing."""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(series_dir))  # most likely empty for .IMA format
    if ids:
        files = reader.GetGDCMSeriesFileNames(str(series_dir), ids[0])
    else:
        print("No series IDs found.")
        files = clean_series(series_dir)  ## removing duplicate slices
    reader.SetFileNames(files)
    return reader.Execute(), files[0]


########################################################################
# ----------  main loop ------------------------------------------------
########################################################################
def convert_patient(patient_root: Path, out_dir: Path):
    # PET_CT_68GA sits two levels down
    level1 = next(patient_root.iterdir())
    petct_dir = next(level1.iterdir())
    ct_dir = next(petct_dir.glob("CT_*"), None)  # CT_WB_2_0_B26F_3_2MM_0004
    pet_dir = next(petct_dir.glob("TOF_PET_*"), None)  # TOF_PET_WB_CORRECTED_0003
    if not ct_dir or not pet_dir:
        print("✗ Missing CT or PET folder in", patient_root.name)
        return

    # Extracting patient name
    p = (
        str(patient_root).split("\\")[-1].split("-")[0]
    )  # e.g John_Smith_01-11-26-10 ---> John_Smith_01
    # There is only one example that goes like John_Smith and not like John_Smith_01
    if len(p.split("_")) == 2:
        patient_name = p
        print("patient: ", patient_name)
    else:
        patient_name = p[:-3]
        print("patient: ", patient_name)

    # ----------  PET (SUV) --------------------------------------------
    pet_img, pet_first = sitk_from_series(pet_dir)
    pet_ds = pydicom.dcmread(pet_first, force=True)
    suv, slope, inter = suv_conversion_factors(pet_ds)

    pet_img = sitk.Cast(pet_img, sitk.sitkFloat32)
    pet_img = (pet_img * slope + inter) * suv  # formula
    pet_out = out_dir / f"{patient_name}_PT.nii.gz"
    ## Make PET isotropic
    new_size = [
        int(round(sz * sp / nsp))
        for sz, sp, nsp in zip(pet_img.GetSize(), pet_img.GetSpacing(), target_spacing)
    ]
    pet_iso = sitk.Resample(
        pet_img,
        new_size,
        transform=sitk.Transform(),  # identity
        interpolator=sitk.sitkLinear,
        outputOrigin=pet_img.GetOrigin(),
        outputSpacing=target_spacing,
        outputDirection=pet_img.GetDirection(),
        defaultPixelValue=0.0,
    )
    sitk.WriteImage(pet_iso, pet_out, imageIO="NiftiImageIO", useCompression=True)
    log_image(patient_name, "PET_SUV", pet_img, pet_iso, pet_out)
    print("✓ PET →", pet_out)

    # ----------  CT  ---------------------------------------------------
    ct_img, ct_first = sitk_from_series(ct_dir)
    ct_ds = pydicom.dcmread(ct_first, force=True)  # .IMA need force
    ct_out = out_dir / f"{patient_name}_CT.nii.gz"
    ct_blur = sitk.SmoothingRecursiveGaussian(
        ct_img,
        sigma=[
            2.0,
            2.0,
            2.0,
        ],  # sigma ~ almost half of the spacing along each axis
    )  # blur to avoid aliasing. Source: https://discourse.itk.org/t/resampling-to-volume-to-smaller-size-and-smaller-voxel-spacing/4984/11 and https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/70_Data_Augmentation.ipynb
    ct_on_pet = sitk.Resample(
        ct_blur,
        pet_iso,  # reference image = PET grid
        sitk.Transform(),  # identity – physical coordinates already align
        sitk.sitkLinear,
        defaultPixelValue=-1000,  # air outside CT field of view
    )
    sitk.WriteImage(ct_on_pet, ct_out, imageIO="NiftiImageIO", useCompression=True)
    log_image(patient_name, "CT", ct_img, ct_on_pet, ct_out)
    print("✓ CT →", ct_out)

    # ----------  MASK  ---------------------------------------------------
    seg_path = next(MASK_DIR.glob(f"{patient_name}"))
    seg_path = next(seg_path.glob("*.seg.nrrd"))
    print("seg_path = ", seg_path)
    convert_mask(enable_long_path(seg_path), pet_iso, MASK_SAVING_DIR, patient_name)


def walk_dataset(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for patient in root.iterdir():
        if patient.is_dir():
            convert_patient(enable_long_path(patient), output)


########################################################################
if __name__ == "__main__":
    walk_dataset(ROOT_DIR, OUTPUT_DIR)
