#!/usr/bin/python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, sys, errno, shutil, uuid,subprocess,csv,json
import math,inspect
import glob,xnat
import re,time
import requests
import pandas as pd
import nibabel as nib
import numpy as np
import pathlib
import os
import traceback
# import xnat
import argparse,xmltodict
from redcapapi_functions import *
catalogXmlRegex = re.compile(r'.*\.xml$')
XNAT_HOST_URL=os.environ['XNAT_HOST']  #'http://snipr02.nrg.wustl.edu:8080' #'https://snipr02.nrg.wustl.edu' #'https://snipr.wustl.edu'
XNAT_HOST = XNAT_HOST_URL # os.environ['XNAT_HOST'] #
XNAT_USER = os.environ['XNAT_USER']#
XNAT_PASS =os.environ['XNAT_PASS'] #
api_token=os.environ['REDCAP_API']

# import xnat
import inspect
import traceback
from datetime import datetime
from railway_fill_database import apply_single_row_csv_to_table,railway_table_exists_for_project,load_csv_to_mysql,drop_column_from_table,apply_single_row_csv_to_table_1
LOG_FILE = "./xnat_session_errors.log"
import inspect
import traceback
from datetime import datetime

LOG_FILE = "/software/railway_db_errors.log"

ERROR_FILE = "/software/error.log"
def log_error_2(msg,func_name):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # err = (
    #     f"[{ts}]\n"
    #     f"{msg}\n"
    #     f"Traceback:\n{traceback.format_exc()}\n"
    #     f"{'-' * 80}\n"
    # )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err = (
        f"[{ts}] Function: {func_name}\n"
        f"[{ts}] Function: {msg}\n"
        # f"Session ID: {session_id}\n"
        f"{msg}\n"
        f"Traceback:\n{traceback.format_exc()}\n"
        f"{'-' * 80}\n"
    )
    with open(ERROR_FILE, "w") as f:
        f.write(err)
import json
# import traceback
# from datetime import datetime


def log_error(msg, func_name):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Allow structured logging
    if isinstance(msg, (dict, list)):
        msg_str = json.dumps(msg, indent=2, default=str)
    else:
        msg_str = str(msg)

    tb = traceback.format_exc()
    tb_str = tb if "Traceback" in tb else "No traceback"

    err = (
        f"[{ts}]\n"
        f"Function: {func_name}\n"
        f"Message:\n{msg_str}\n"
        f"Traceback:\n{tb_str}\n"
        f"{'-' * 80}\n"
    )

    with open(ERROR_FILE, "a") as f:   # APPEND, not overwrite
        f.write(err)

import re
from datetime import datetime

def _extract_mmddyyyy_from_tail(filename: str, ext_lower: str):
    """
    Extract date from tail pattern: _MM_DD_YYYY.<ext>
    Example: ..._09_26_2023.pdf  -> datetime.date(2023, 9, 26)
    Returns None if pattern not found.
    """
    # Ensure we match the date immediately before the extension
    # e.g. "_09_26_2023.pdf" or "_09_26_2023.nii.gz" (ext_lower passed in)
    pattern = re.compile(rf"_(\d{{2}})_(\d{{2}})_(\d{{4}}){re.escape(ext_lower)}$", re.IGNORECASE)
    m = pattern.search(filename)
    if not m:
        return None
    mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
    try:
        return datetime.strptime(f"{mm}_{dd}_{yyyy}", "%m_%d_%Y").date()
    except Exception:
        return None


# # ---- replace your old sort with this ----
# ext_lower = ext_lower  # already computed earlier (like ".pdf")
# dated = []
# undated = []
#
# for fn in matches:
#     d = _extract_mmddyyyy_from_tail(fn, ext_lower)
#     if d is None:
#         undated.append(fn)
#     else:
#         dated.append((d, fn))
#
# if dated:
#     # sort by actual date, then filename as tie-breaker
#     dated.sort(key=lambda x: (x[0], x[1]))
#     latest_name = dated[-1][1]
# else:
#     # fallback if none have the date pattern
#     matches_sorted = sorted(matches)
#     latest_name = matches_sorted[-1]

def download_file_from_xnat_uri(uri: str, out_path: str, verify: bool = True):
    """
    Download a file from XNAT given a REST URI like:
      /data/projects/.../files/<filename>

    Saves to out_path and returns out_path on success, else None.
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        if not uri or not uri.startswith("/data/"):
            log_error(f"Invalid uri: {uri}", func_name)
            return None

        if not out_path:
            log_error("out_path is empty", func_name)
            return None

        # Build full URL
        base = XNAT_HOST.rstrip("/")
        url = f"{base}{uri}"

        # Ensure parent dir exists
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Stream download
        with requests.get(url, auth=(XNAT_USER, XNAT_PASS), stream=True, verify=verify) as r:
            if r.status_code != 200:
                log_error(f"Download failed: HTTP {r.status_code} url={url} text={r.text[:200]}", func_name)
                return None

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return out_path

    except Exception:
        log_error("Unhandled exception during download_file_from_xnat_uri", func_name)
        return None

def get_latest_file_uri_from_scan_resource(
    session_id: str,
    scan_id: str,
    scan_resource_dir_name: str,
    file_extension: str,
):
    """
    Given:
      - session_id (experiment id)
      - scan_id (scan identifier)
      - scan_resource_dir_name (resource folder name under the scan)
      - file_extension (e.g. ".pdf" or "pdf")

    Do:
      - list files in scan resource folder that end with the extension
      - sort filenames
      - pick the last one (latest by sort order)
      - return the URI of that file

    Returns:
      - uri (str) OR None on failure
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        # Normalize extension
        ext = file_extension.strip()
        if not ext:
            log_error("Empty file_extension provided", func_name)
            return None
        if not ext.startswith("."):
            ext = "." + ext
        ext_lower = ext.lower()

        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as conn:
            if session_id not in conn.experiments:
                log_error(f"Session ID not found on XNAT: {session_id}", func_name)
                return None

            exp = conn.experiments[session_id]

            # Find scan
            if scan_id not in exp.scans:
                log_error(f"Scan ID not found in session: {scan_id}", func_name)
                return None

            scan = exp.scans[scan_id]

            # Find scan resource directory
            if scan_resource_dir_name not in scan.resources:
                log_error(
                    f'Scan resource "{scan_resource_dir_name}" not found for scan_id={scan_id}',
                    func_name,
                )
                return None

            res = scan.resources[scan_resource_dir_name]

            # List files with the requested extension
            file_names = [str(fn) for fn in res.files.keys()]
            matches = [fn for fn in file_names if fn.lower().endswith(ext_lower)]

            if not matches:
                log_error(
                    f'No files ending with "{ext}" found in scan_resource="{scan_resource_dir_name}" '
                    f'for scan_id={scan_id}',
                    func_name,
                )
                return None

            # Sort and pick "latest" by sort order
            # matches_sorted = sorted(matches)
            # latest_name = matches_sorted[-1]
            ext_lower = ext_lower  # already computed earlier (like ".pdf")
            dated = []
            undated = []

            for fn in matches:
                d = _extract_mmddyyyy_from_tail(fn, ext_lower)
                if d is None:
                    undated.append(fn)
                else:
                    dated.append((d, fn))

            if dated:
                # sort by actual date, then filename as tie-breaker
                dated.sort(key=lambda x: (x[0], x[1]))
                latest_name = dated[-1][1]
            else:
                # fallback if none have the date pattern
                matches_sorted = sorted(matches)
                latest_name = matches_sorted[-1]

            fobj = res.files[latest_name]

            # Best-effort URI extraction across xnatpy versions
            uri = (
                getattr(fobj, "uri", None)
                or getattr(fobj, "_uri", None)
                or getattr(fobj, "data_uri", None)
                or getattr(fobj, "URI", None)
            )

            if uri is None:
                # Fallback: try to expose something still useful for debugging
                try:
                    uri = str(getattr(fobj, "xnat_uri"))
                except Exception:
                    uri = None

            if uri is None:
                log_error(
                    f"Could not determine URI for file: {latest_name} (scan_id={scan_id}, resource={scan_resource_dir_name})",
                    func_name,
                )
                return None

            return uri

    except Exception:
        log_error("Unhandled exception during execution", func_name)
        return None


def get_id_from_nifti_location_csv(
    session_id: str,
    resource_name: str = "NIFTI_LOCATION",
    csv_suffix: str = "NIFTILOCATION.csv",
    id_col: str = "ID",
):
    """
    Returns:
      - scalar ID value (first unique value) OR None on failure
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        msg = " I AM HERE!!!!!!!!!!!!!!!!!"
        log_error(msg, func_name)

        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as conn:

            if session_id not in conn.experiments:
                log_error("Session ID not found on XNAT", func_name)
                return None

            exp = conn.experiments[session_id]

            if resource_name not in exp.resources:
                log_error(f'Resource "{resource_name}" not found', func_name)
                return None

            res = exp.resources[resource_name]

            csv_files = [
                fname for fname in res.files.keys()
                if str(fname).endswith(csv_suffix)
            ]

            if not csv_files:
                log_error(f'No file ending with "{csv_suffix}" found', func_name)
                return None

            csv_name = sorted(csv_files)[0]

            # --- FIX: download using xnatpy download() ---
            import tempfile, os

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp_path = tmp.name

                # xnatpy file object supports download(path)
                res.files[csv_name].download(tmp_path)

                df = pd.read_csv(tmp_path)

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            # --- extract ID ---
            if id_col not in df.columns:
                log_error(f'Column "{id_col}" not found in {csv_name}. ', func_name)
                return None

            series = df[id_col].dropna()
            if series.empty:
                log_error(f'Column "{id_col}" exists but has no valid values', func_name)
                return None

            return series.unique()[0]

    except Exception:
        log_error("Unhandled exception during execution", func_name)
        return None



def call_apply_single_row_csv_to_table(session_id, csv_file):
    """
    Wrapper to:
    - resolve project/subject from session_id
    - use project_id as table name
    - apply single-row CSV to DB table
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        # Resolve project & subject
        project_id, subject_label = given_sessionid_get_project_n_subjectids(session_id)

        if not project_id:
            raise ValueError(f"Could not resolve project for session_id={session_id}")

        table_name = project_id  # as per your design

        result = apply_single_row_csv_to_table(
            # engine=ENGINE,                    # global/shared engine
            csv_file=csv_file,
            table_name=table_name,
            session_id=session_id,      # identifier column in CSV & DB
        )

        return result

    except Exception as e:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"[{ts}] Function: {func_name}\n"
            f"session_id={session_id}\n"
            f"csv_file={csv_file}\n"
            f"Error: {e}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
            f"{'-'*80}\n"
        )
        with open(LOG_FILE, "a") as f:
            f.write(msg)

        # Propagate or return sentinel (choose one)
        raise
        # or: return None

def given_sessionid_get_project_n_subjectids(session_id):
    """
    Given an XNAT experiment/session ID, return (project_id, subject_id).
    Logs any exception with function name and timestamp.
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as connection:
            exp = connection.experiments[session_id]
            subj = exp.subject
            # print("Project ID:", exp.project)  # project identifier
            # print("Subject ID:", subj.id)  # system-wide subject id
            # print("Subject Label:", subj.label)
            return exp.project, subj.label

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        error_msg = (
            f"[{timestamp}] "
            f"Function: {func_name}\n"
            f"Session ID: {session_id}\n"
            f"Error: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
            f"{'-'*80}\n"
        )

        with open(LOG_FILE, "a") as f:
            f.write(error_msg)

        # Safe failure return (important for batch pipelines)
        return None, None



def given_csvfile_proj_subjids_append(csvfile, session_id):
    """
    Read CSV with session IDs, append/overwrite PROJECT_ID and SUBJECT_ID columns.
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        df = pd.read_csv(csvfile)

        # Ensure columns exist
        df["PROJECT_ID"] = ""
        df["SUBJECT_ID"] = ""

        for idx, row in df.iterrows():
            # session_id = str(row.get(session_col, "")).strip()
            if not session_id:
                continue

            try:
                proj, subj = given_sessionid_get_project_n_subjectids(session_id)
                df.at[idx, "PROJECT_ID"] = "" if proj is None else proj
                df.at[idx, "SUBJECT_ID"] = "" if subj is None else subj
            except Exception:
                df.at[idx, "PROJECT_ID"] = ""
                df.at[idx, "SUBJECT_ID"] = ""

        out_csv = csvfile #.replace(".csv", "_with_proj_subj.csv")
        df.to_csv(out_csv, index=False)
        return out_csv

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        err = (
            f"[{timestamp}] Function: {func_name}\n"
            f"CSV file: {csvfile}\n"
            f"Error: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
            f"{'-' * 80}\n"
        )
        with open(LOG_FILE, "a") as f:
            f.write(err)
        return None
def get_scan_dicom_metadata_from_first_dicom(
    session_id: str,
    scan_id: str,
    dicom_resource_name: str = "DICOM",
    bash_safe: bool = True,
):
    """
    Given session_id and scan_id:
      - find scan's resource folder (default: "DICOM")
      - download the first DICOM file from that resource
      - read DICOM header tags using SimpleITK (GDCM)
      - return tuple:
          (acquisition_site, acquisition_datetime, scanner, body_part, kvp)

    Returns:
      (site, acq_dt, scanner, body_part, kvp) OR (None, None, None, None, None) on failure
    """
    func_name = inspect.currentframe().f_code.co_name

    def _clean(v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        # Collapse whitespace
        s = " ".join(s.split())
        if bash_safe:
            s = s.replace(" ", "_")
        return s

    def _get_tag(reader, tag, default=None):
        try:
            if reader.HasMetaDataKey(tag):
                return reader.GetMetaData(tag)
        except Exception:
            pass
        return default

    try:
        if not session_id or not str(session_id).strip():
            log_error("session_id is empty", func_name)
            return None, None, None, None, None
        if not scan_id or not str(scan_id).strip():
            log_error("scan_id is empty", func_name)
            return None, None, None, None, None

        import tempfile
        import os
        import SimpleITK as sitk

        # -------------------------
        # 1) Find first DICOM file in scan resource
        # -------------------------
        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as conn:
            if session_id not in conn.experiments:
                log_error(f"Session ID not found on XNAT: {session_id}", func_name)
                return None, None, None, None, None

            exp = conn.experiments[session_id]

            if scan_id not in exp.scans:
                log_error(f"Scan ID not found in session: {scan_id}", func_name)
                return None, None, None, None, None

            scan = exp.scans[scan_id]

            if dicom_resource_name not in scan.resources:
                log_error(
                    f'Scan resource "{dicom_resource_name}" not found for scan_id={scan_id}',
                    func_name,
                )
                return None, None, None, None, None

            res = scan.resources[dicom_resource_name]

            # list all files (keys may include paths)
            all_files = [str(k) for k in res.files.keys()]

            # Prefer .dcm (case-insensitive), otherwise take the first file
            dcm_files = [f for f in all_files if f.lower().endswith(".dcm")]
            candidates = sorted(dcm_files) if dcm_files else sorted(all_files)

            if not candidates:
                log_error(
                    f'No files found in scan resource "{dicom_resource_name}" (scan_id={scan_id})',
                    func_name,
                )
                return None, None, None, None, None

            first_name = candidates[0]

            # download to temp file
            tmp_path = None
            try:
                suffix = ".dcm" if first_name.lower().endswith(".dcm") else ".bin"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp_path = tmp.name

                res.files[first_name].download(tmp_path)

            except Exception:
                log_error(
                    f"Failed to download first DICOM file: {first_name} (scan_id={scan_id})",
                    func_name,
                )
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                return None, None, None, None, None

        # -------------------------
        # 2) Read DICOM header tags using SimpleITK (no pixel load)
        # -------------------------
        try:
            # Use ImageFileReader for metadata
            reader = sitk.ImageFileReader()
            reader.SetFileName(tmp_path)

            # Ensure GDCM for DICOM
            image_io = sitk.GDCMImageIO()
            reader.SetImageIO(image_io)

            # Only read header info (fast)
            reader.ReadImageInformation()

            # DICOM tags (group|element)
            institution = _get_tag(reader, "0008|0080")  # InstitutionName
            acq_dt = _get_tag(reader, "0008|002a")       # AcquisitionDateTime

            # Fallback if AcquisitionDateTime missing:
            if not acq_dt:
                acq_date = _get_tag(reader, "0008|0022")  # AcquisitionDate
                acq_time = _get_tag(reader, "0008|0032")  # AcquisitionTime
                if acq_date and acq_time:
                    acq_dt = f"{acq_date}_{acq_time}"
                else:
                    # Additional fallbacks (often present)
                    study_date = _get_tag(reader, "0008|0020")  # StudyDate
                    study_time = _get_tag(reader, "0008|0030")  # StudyTime
                    if study_date and study_time:
                        acq_dt = f"{study_date}_{study_time}"

            manufacturer = _get_tag(reader, "0008|0070")  # Manufacturer
            model = _get_tag(reader, "0008|1090")         # ManufacturerModelName
            body_part = _get_tag(reader, "0018|0015")     # BodyPartExamined
            kvp = _get_tag(reader, "0018|0060")           # KVP

            scanner = None
            if manufacturer and model:
                scanner = f"{manufacturer}_{model}"
            else:
                scanner = manufacturer or model

            # Clean for Bash if requested
            institution = _clean(institution)
            acq_dt = _clean(acq_dt)
            scanner = _clean(scanner)
            body_part = _clean(body_part)
            kvp = _clean(kvp)

            return institution, acq_dt, scanner, body_part, kvp

        finally:
            # cleanup temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    except Exception:
        log_error("Unhandled exception during get_scan_dicom_metadata_from_first_dicom", func_name)
        return None, None, None, None, None


def get_session_label_from_session_id(session_id: str):
    """
    Given an XNAT experiment/session ID, return the session label.

    Returns:
      - session_label (str) OR None on failure
    """
    func_name = inspect.currentframe().f_code.co_name

    try:
        if not session_id or not str(session_id).strip():
            log_error("session_id is empty", func_name)
            return None

        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as conn:
            if session_id not in conn.experiments:
                log_error(f"Session ID not found on XNAT: {session_id}", func_name)
                return None

            exp = conn.experiments[session_id]
            return exp.label

    except Exception:
        log_error("Unhandled exception during get_session_label_from_session_id", func_name)
        return None


def upload_file_to_project_resource(
    project_id: str,
    resource_label: str,
    local_file_path: str,
    remote_filename: str , #| None = None,
    verify: bool = True,
    use_multipart: bool = True,
    log_file: str = "xnat_upload_errors.log"
) -> dict:
    """
    Upload a local file into a PROJECT-level resource folder in XNAT.

    Target REST shape (project resources):
      - Create resource folder:
          PUT  /data/projects/{project-id}/resources/{resource-label}
      - Upload file:
          PUT  /data/projects/{project-id}/resources/{resource-label}/files/{filename}
        (either multipart 'file=@...' OR raw body with ?inbody=true)
    See XNAT API docs. :contentReference[oaicite:0]{index=0}

    Returns:
      {"ok": True, "project_id": ..., "resource_label": ..., "filename": ..., "uploaded_to": ...}
      or {"ok": False, "error": "...", "where": "..."} on failure (and logs traceback).
    """
    xnat_host=XNAT_HOST
    username = XNAT_USER
    password = XNAT_PASS
    try:
        if not os.path.isfile(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        filename = remote_filename or os.path.basename(local_file_path)

        # XNAT expects paths under /data/...
        resource_base = f"/data/projects/{project_id}/resources/{resource_label}"
        upload_endpoint = f"{resource_base}/files/{filename}"

        with xnat.connect(xnat_host, user=username, password=password, verify=verify) as sess:
            # xnatpy session exposes requests-like methods; use interface.* if present
            http = getattr(sess, "interface", sess)

            # 1) Ensure the project resource folder exists (idempotent PUT)
            r1 = http.put(resource_base)
            if hasattr(r1, "ok") and not r1.ok:
                raise RuntimeError(f"Failed creating/ensuring resource folder: {r1.status_code} {getattr(r1, 'text', '')}")

            # 2) Upload the file
            with open(local_file_path, "rb") as f:
                if use_multipart:
                    # multipart form upload (like: curl -F "file=@x")
                    r2 = http.put(upload_endpoint, files={"file": f})
                else:
                    # raw body upload requires inbody=true
                    r2 = http.put(upload_endpoint, params={"inbody": "true"}, data=f)

            if hasattr(r2, "ok") and not r2.ok:
                raise RuntimeError(f"Upload failed: {r2.status_code} {getattr(r2, 'text', '')}")

        return {
            "ok": True,
            "project_id": project_id,
            "resource_label": resource_label,
            "filename": filename,
            "uploaded_to": upload_endpoint,
        }

    except Exception as e:
        # Write full traceback to a log file (so you can debug auth/permissions/paths)
        try:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write("\n" + "=" * 80 + "\n")
                lf.write(f"Error uploading to XNAT project resource\n")
                lf.write(f"xnat_host={xnat_host}\nproject_id={project_id}\nresource_label={resource_label}\nfile={local_file_path}\n")
                lf.write(f"Exception: {repr(e)}\n")
                lf.write(traceback.format_exc())
        except Exception:
            pass

        return {"ok": False, "where": "upload_file_to_project_resource", "error": str(e)}
from railway_fill_database import  railway_drop_table

def create_new_sessionlist_table_in_railway(project_id: str) -> str:

    table_present=railway_table_exists_for_project(project_id)
    func_name = inspect.currentframe().f_code.co_name
    if table_present == 0 :
        xnat_download_project_sessions_csv(project_id,f'/software/{project_id}_copy.csv')
        make_csv_columns_railway_compatible(f'/software/{project_id}_copy.csv',f'/software/{project_id}.csv')
        log_error(f"table created in the railway: {project_id}", func_name)
        load_csv_to_mysql(f'/software/{project_id}.csv') #,key_col="SESSION_ID")

    log_error(f"table present in the railway: {table_present}", func_name)
def create_new_sessionlist_table_in_railway_with_session_id(session_id: str) -> str:

    proj_id,subj_id=given_sessionid_get_project_n_subjectids(session_id)
    create_new_sessionlist_table_in_railway(proj_id)

def xnat_download_project_sessions_csv(
    project_id: str,
    out_csv_path: str,
    *,
    # xnat_host: str,
    # username: str,
    # password: str,
    verify: bool = True,
) -> str:
    """
    Step-2:
    Download XNAT sessions (experiments) list for a project as CSV
    using xnatpy session + native XNAT CSV endpoint.

    Endpoint used:
      /data/projects/{project_id}/experiments?format=csv

    Returns
    -------
    str
        Path to saved CSV file
    """
    xnat_host= XNAT_HOST #: str,
    username=XNAT_USER #str,
    password=XNAT_PASS #: str,

    if not project_id or not project_id.strip():
        raise ValueError("project_id is required")

    project_id = project_id.strip()
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    with xnat.connect(
        xnat_host,
        user=username,
        password=password,
        verify=verify,
    ) as sess:

        if project_id not in sess.projects:
            raise ValueError(f"Project '{project_id}' not found in XNAT")

        # base = sess.host.rstrip("/")
        base = XNAT_HOST.rstrip("/")
        csv_url = f"{base}/data/projects/{project_id}/experiments?format=csv"

        # Use authenticated session from xnatpy
        response = sess._interface.get(csv_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch sessions CSV for project '{project_id}': "
                f"HTTP {response.status_code}"
            )

        with open(out_csv_path, "wb") as f:
            f.write(response.content)

    return out_csv_path

def make_csv_columns_railway_compatible(
    in_csv_path: str,
    out_csv_path: str,
) -> str:
    """
    Step-3:
    Take an input CSV (XNAT sessions list),
    make column names compatible with Railway/MySQL,
    and write a cleaned CSV.

    Assumptions (explicit by design):
    - The column `ID` ALWAYS exists and is the session identifier.
    - `ID` will be renamed to `SESSION_ID`.
    """

    if not in_csv_path or not str(in_csv_path).strip():
        raise ValueError("in_csv_path is required")
    if not out_csv_path or not str(out_csv_path).strip():
        raise ValueError("out_csv_path is required")

    if not os.path.exists(in_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {in_csv_path}")

    def clean_col(col: str) -> str:
        c = str(col).strip()
        c = re.sub(r"\s+", "_", c)              # spaces → _
        c = re.sub(r"[^0-9A-Za-z_]+", "", c)    # drop special chars
        c = re.sub(r"_+", "_", c).strip("_")    # collapse _
        if not c:
            c = "COL"
        if re.match(r"^[0-9]", c):
            c = f"C_{c}"
        return c.upper()

    # Read CSV
    df = pd.read_csv(in_csv_path)

    # Clean column names
    df.columns = [clean_col(c) for c in df.columns]

    # Enforce session ID rule
    if "ID" not in df.columns:
        raise RuntimeError("Expected column 'ID' not found in sessions CSV")

    df.rename(columns={"ID": "SESSION_ID"}, inplace=True)

    # Drop duplicate columns after cleaning
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Drop fully-null columns (safe for DB creation)
    df.dropna(axis=1, how="all", inplace=True)

    # Write cleaned CSV
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    return out_csv_path
# Z-Brain-Thin  Z-Axial-Brain Z-Axial-Brain_usable Z-Brain-Thin_usable Z-Brain-Thin_scan_ids Z-Axial-Brain_scan_ids
# import xnat

# import xnat

# import xnat

def analyze_scans_in_session(session_id: str):
    """
    Analyze scans in an XNAT session.

    - Counts Z-Axial-Brain and Z-Brain-Thin
    - Counts scan quality (usable / questionable / unusable) ONLY for those types
    - Separately counts:
        * axial_usable
        * thin_usable
    - Stores scan details ONLY for Z-Axial-Brain and Z-Brain-Thin
    """

    TARGET_TYPES = {"Z-Axial-Brain", "Z-Brain-Thin"}

    type_counts = {
        "Z-Axial-Brain": 0,
        "Z-Brain-Thin": 0
    }

    quality_counts = {
        "usable": 0,
        "questionable": 0,
        "unusable": 0
    }

    # 🔹 NEW: explicit usable-by-type counters
    usable_by_type = {
        "axial_usable": 0,
        "thin_usable": 0
    }

    scan_details = []

    with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as xnat_session:

        experiment = xnat_session.experiments[session_id]

        for scan_id, scan in experiment.scans.items():

            scan_type = getattr(scan, "type", None)
            scan_quality = getattr(scan, "quality", None)

            # TYPE COUNTS
            if scan_type in type_counts:
                type_counts[scan_type] += 1

            # Only operate on target types
            if scan_type in TARGET_TYPES:

                # QUALITY COUNTS
                if scan_quality == "usable":
                    quality_counts["usable"] += 1

                    # 🔹 usable separated by type
                    if scan_type == "Z-Axial-Brain":
                        usable_by_type["axial_usable"] += 1
                    elif scan_type == "Z-Brain-Thin":
                        usable_by_type["thin_usable"] += 1

                elif scan_quality == "questionable":
                    quality_counts["questionable"] += 1
                else:
                    quality_counts["unusable"] += 1

                # STORE DETAILS
                scan_details.append({
                    "scan_id": scan_id,
                    "type": scan_type,
                    "series_description": getattr(scan, "series_description", None),
                    "quality": scan_quality,
                    "frames": getattr(scan, "frames", None)
                })

    return {
        "scan_details": scan_details,
        "type_counts": type_counts,
        "quality_counts": quality_counts,
        "usable_by_type": usable_by_type
    }

# import xnat
def get_nifti_filenames_from_scan_details_uri(session_id: str, scan_details: list):
    """
    Takes scan_details (from analyze_scans_in_session) and returns NIFTI FILE URLs
    found in each scan's 'NIFTI' resource.

    Output structure:
    {
      "axial": [ {"scan_id":..., "files":[<url>, ...]} , ...],
      "thin":  [ {"scan_id":..., "files":[<url>, ...]} , ...],
      "missing_nifti_resource": [scan_id, ...],
      "errors": [ {"scan_id":..., "error":...}, ...]
    }

    NOTE:
    - Uses global XNAT_HOST / XNAT_USER / XNAT_PASS
    - URLs require XNAT authentication when accessed
    """

    out = {
        "axial": [],
        "thin": [],
        "missing_nifti_resource": [],
        "errors": []
    }

    with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as xnat_session:
        experiment = xnat_session.experiments[session_id]

        for item in scan_details:
            scan_id = str(item.get("scan_id"))
            scan_type = item.get("type")

            try:
                scan = experiment.scans[scan_id]

                # Check NIFTI resource
                if "NIFTI" not in scan.resources:
                    out["missing_nifti_resource"].append(scan_id)
                    continue

                nifti_res = scan.resources["NIFTI"]

                file_urls = []
                for f in nifti_res.files.values():
                    # f.uri is the REST endpoint for the file
                    uri = getattr(f, "uri", None)
                    if uri:
                        # Ensure no double slash
                        file_urls.append(XNAT_HOST.rstrip("/") + uri)

                # Put into correct bucket
                payload = {
                    "scan_id": scan_id,
                    "files": file_urls
                }

                if scan_type == "Z-Axial-Brain":
                    out["axial"].append(payload)
                elif scan_type == "Z-Brain-Thin":
                    out["thin"].append(payload)

            except Exception as e:
                out["errors"].append({
                    "scan_id": scan_id,
                    "error": str(e)
                })

    return out

def get_nifti_filenames_from_scan_details(session_id: str, scan_details: list):
    """
    Takes scan_details (from analyze_scans_in_session) and returns NIFTI filenames
    found in each scan's 'NIFTI' resource.

    Output structure:
    {
      "axial": [ {"scan_id":..., "files":[...]} , ...],
      "thin":  [ {"scan_id":..., "files":[...]} , ...],
      "missing_nifti_resource": [scan_id, ...],
      "errors": [ {"scan_id":..., "error":...}, ...]
    }

    NOTE: Uses global XNAT_HOST / XNAT_USER / XNAT_PASS like your previous function.
    """

    out = {
        "axial": [],
        "thin": [],
        "missing_nifti_resource": [],
        "errors": []
    }

    with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as xnat_session:
        experiment = xnat_session.experiments[session_id]

        for item in scan_details:
            scan_id = str(item.get("scan_id"))
            scan_type = item.get("type")

            try:
                scan = experiment.scans[scan_id]

                # Get NIFTI resource (common label is "NIFTI")
                if "NIFTI" not in scan.resources:
                    out["missing_nifti_resource"].append(scan_id)
                    continue

                nifti_res = scan.resources["NIFTI"]

                # List filenames in that resource
                filenames = []
                for f in nifti_res.files.values():
                    # xnatpy file objects usually have .name
                    filenames.append(getattr(f, "name", None))

                # clean Nones
                filenames = [n for n in filenames if n]

                # Put into the correct bucket based on scan_type
                payload = {"scan_id": scan_id, "files": filenames}

                if scan_type == "Z-Axial-Brain":
                    out["axial"].append(payload)
                elif scan_type == "Z-Brain-Thin":
                    out["thin"].append(payload)
                else:
                    # ignore non-target types silently (scan_details should already be filtered)
                    pass

            except Exception as e:
                out["errors"].append({"scan_id": scan_id, "error": str(e)})

    return out

def format_nifti_files_for_log(nifti_files: dict, session_id: str = None) -> str:
    """
    Convert nifti_files dict into a compact, readable multi-line string for logging.
    """
    lines = []
    if session_id:
        lines.append(f"Session: {session_id}")

    # Axial
    lines.append("NIFTI files (AXIAL):")
    if nifti_files.get("axial"):
        for rec in nifti_files["axial"]:
            scan_id = rec.get("scan_id")
            files = rec.get("files", [])
            lines.append(f"  - scan {scan_id}: " + (", ".join(files) if files else "(no files)"))
    else:
        lines.append("  (none)")

    # Thin
    lines.append("NIFTI files (THIN):")
    if nifti_files.get("thin"):
        for rec in nifti_files["thin"]:
            scan_id = rec.get("scan_id")
            files = rec.get("files", [])
            lines.append(f"  - scan {scan_id}: " + (", ".join(files) if files else "(no files)"))
    else:
        lines.append("  (none)")

    # Missing resource
    missing = nifti_files.get("missing_nifti_resource", [])
    if missing:
        lines.append("Missing NIFTI resource scan_ids: " + ", ".join(map(str, missing)))

    # Errors
    errs = nifti_files.get("errors", [])
    if errs:
        lines.append("Errors:")
        for e in errs:
            lines.append(f"  - scan {e.get('scan_id')}: {e.get('error')}")

    return "\n".join(lines)


def log_step2_nifti_files(nifti_files: dict, session_id: str = None):
    """
    Logs the NIFTI file listing using your existing log_error(msg, func_name).
    """
    step2 = format_nifti_files_for_log(nifti_files, session_id=session_id)

    log_error(
        step2,
        func_name="fill_after_dicom2nifti"
    )
# import csv
def combine_analysis_with_nifti_files(analysis: dict, nifti_files: dict) -> dict:
    """
    Merge nifti_files into the analysis dict so downstream (CSV/DB/logging) can use one object.
    """
    analysis["nifti_files"] = nifti_files

    # Optional convenience fields (ready for CSV/db)
    analysis["nifti_files_axial"] = [
        {"scan_id": r.get("scan_id"), "files": r.get("files", [])}
        for r in nifti_files.get("axial", [])
    ]
    analysis["nifti_files_thin"] = [
        {"scan_id": r.get("scan_id"), "files": r.get("files", [])}
        for r in nifti_files.get("thin", [])
    ]
    analysis["missing_nifti_resource_scan_ids"] = nifti_files.get("missing_nifti_resource", [])
    analysis["nifti_errors"] = nifti_files.get("errors", [])

    return analysis

import csv

def _flatten_nifti_list(nifti_list):
    """
    nifti_list format:
      [ {"scan_id": "...", "files": ["a.nii.gz","b.nii.gz"]}, ... ]

    Returns a single string safe for CSV/DB:
      "scanID:file1|file2;scanID:file1"
    """
    parts = []
    for rec in (nifti_list or []):
        scan_id = str(rec.get("scan_id"))
        files = rec.get("files", []) or []
        files = [str(f) for f in files if f]
        if files:
            parts.append(f"|".join(files)) #{scan_id}:" +
        else:
            parts.append(f"{scan_id}:(no_files)")
    return ";".join(parts)


def write_session_scan_summary_csv(
    session_id: str,
    analysis: dict,
    out_csv: str,
):
    """
    Writes a 1-row CSV using the COMBINED output (analysis + nifti_files).

    Columns:
      session_id_this,
      Z-Brain-Thin_count,
      Z-Axial-Brain_count,
      Z-Axial-Brain_usable_count,
      Z-Brain-Thin_usable_count,
      Z-Brain-Thin_scan_ids,
      Z-Axial-Brain_scan_ids,
      Z-Brain-Thin_nifti_files,
      Z-Axial-Brain_nifti_files,
      missing_nifti_resource_scan_ids
    """

    # ---- counts ----
    type_counts = analysis.get("type_counts", {}) or {}
    usable_by_type = analysis.get("usable_by_type", {}) or {}
    scan_details = analysis.get("scan_details", []) or []

    z_thin_count = int(type_counts.get("Z-Brain-Thin", 0))
    z_axial_count = int(type_counts.get("Z-Axial-Brain", 0))

    z_axial_usable = int(usable_by_type.get("axial_usable", 0))
    z_thin_usable = int(usable_by_type.get("thin_usable", 0))

    # ---- scan_ids ----
    z_thin_ids = []
    z_axial_ids = []

    for rec in scan_details:
        sid = str(rec.get("scan_id"))
        stype = rec.get("type")
        if stype == "Z-Brain-Thin":
            z_thin_ids.append(sid)
        elif stype == "Z-Axial-Brain":
            z_axial_ids.append(sid)

    z_thin_ids_str = "|".join(z_thin_ids)
    z_axial_ids_str = "|".join(z_axial_ids)

    # ---- nifti filenames (combined output) ----
    # Prefer the convenience lists if you added them via combine_analysis_with_nifti_files()
    thin_nifti_list = analysis.get("nifti_files_thin")
    axial_nifti_list = analysis.get("nifti_files_axial")

    # Fallback if only raw nifti_files dict exists
    if thin_nifti_list is None or axial_nifti_list is None:
        nf = analysis.get("nifti_files", {}) or {}
        thin_nifti_list = thin_nifti_list if thin_nifti_list is not None else nf.get("thin", [])
        axial_nifti_list = axial_nifti_list if axial_nifti_list is not None else nf.get("axial", [])

    z_thin_nifti_str = _flatten_nifti_list(thin_nifti_list)
    z_axial_nifti_str = _flatten_nifti_list(axial_nifti_list)

    missing_ids = analysis.get("missing_nifti_resource_scan_ids")
    if missing_ids is None:
        missing_ids = (analysis.get("nifti_files", {}) or {}).get("missing_nifti_resource", [])
    missing_ids_str = "|".join([str(x) for x in (missing_ids or [])])

    # ---- write CSV ----
    headers = [
        "session_id_this",
        "Z-Brain-Thin_count",
        "Z-Axial-Brain_count",
        "Z-Axial-Brain_usable_count",
        "Z-Brain-Thin_usable_count",
        "Z-Brain-Thin_scan_ids",
        "Z-Axial-Brain_scan_ids",
        "Z-Brain-Thin_nifti_files",
        "Z-Axial-Brain_nifti_files",
        "missing_nifti_resource_scan_ids",
    ]

    row = {
        "session_id_this": session_id,
        "Z-Brain-Thin_count": z_thin_count,
        "Z-Axial-Brain_count": z_axial_count,
        "Z-Axial-Brain_usable_count": z_axial_usable,
        "Z-Brain-Thin_usable_count": z_thin_usable,
        "Z-Brain-Thin_scan_ids": z_thin_ids_str,
        "Z-Axial-Brain_scan_ids": z_axial_ids_str,
        "Z-Brain-Thin_nifti_files": z_thin_nifti_str,
        "Z-Axial-Brain_nifti_files": z_axial_nifti_str,
        "missing_nifti_resource_scan_ids": missing_ids_str,
    }

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)

    return out_csv

def fill_after_dicom2nifti(session_id):
    # step1=analyze_scans_in_session(session_id)
    # log_error(step1,
    #     func_name="fill_after_dicom2nifti",
    # )
    # nifti_files=get_nifti_filenames_from_scan_details(session_id, step1["scan_details"])
    analysis = analyze_scans_in_session(session_id)
    nifti_files = get_nifti_filenames_from_scan_details(session_id, analysis["scan_details"])
    step1=combine_analysis_with_nifti_files(analysis, nifti_files)
    log_step2_nifti_files(nifti_files, session_id=session_id)
    log_error(step1,
        func_name="fill_after_dicom2nifti",
    )
    csv_file=f'{session_id}_values_after_dicom2nifti.csv'
    write_session_scan_summary_csv(session_id,
    step1,csv_file

    )
    project_id,subject_id=given_sessionid_get_project_n_subjectids(session_id)
    table_name = project_id  # as per your design

    drop_column_from_table(
            table_name,
            'session_id_this'
    )
    result = apply_single_row_csv_to_table_1(
        # engine=ENGINE,                    # global/shared engine
        csv_file=csv_file,
        table_name=table_name,
        session_id_col='session_id_this',  # identifier column in CSV & DB
    )
    # # nifti_files=get_nifti_filenames_from_scan_details(session_id, step1["scan_details"])
    # # log_step2_nifti_files(nifti_files, session_id=session_id)
    # # step2 = count_usability_for_z_axial_scans(session_id, step1["scan_ids"])
    # # log_error(step2,
    # #     func_name="fill_after_dicom2nifti",
    # # )
def get_candidate_scans_for_dicom2nifti(
    session_id: str,
    target_types=("Z-Axial-Brain", "Z-Brain-Thin"),
    ok_qualities=("usable", "questionable"),
):
    """
    (1) Given a session_id, return scan_ids that match type + quality criteria.

    Returns: list[str]
    """
    func_name = "get_candidate_scans_for_dicom2nifti"
    try:
        target_types_set = set(target_types)
        ok_qualities_set = set(q.lower() for q in ok_qualities)

        scan_ids = []
        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            exp = sess.experiments[session_id]

            for scan_id, scan in exp.scans.items():
                stype = getattr(scan, "type", None)
                squal = getattr(scan, "quality", None)

                if stype in target_types_set and (squal or "").lower() in ok_qualities_set:
                    scan_ids.append(str(scan_id))

        return scan_ids

    except Exception:
        log_error({"session_id": session_id}, func_name)
        return []

# import os
# import xnat

# expects these to exist in your utilities file (as they already do)
# XNAT_HOST, XNAT_USER, XNAT_PASS
# log_error(msg, func_name)

def xnat_download_scan_resource_zip(
    session_id: str,
    scan_id: str,
    resource_name: str,
    zip_path: str,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Download *any* scan resource folder as a ZIP:
      /data/experiments/{session_id}/scans/{scan_id}/resources/{resource_name}/files?format=zip

    Returns zip_path if successful; raises on failure.
    """
    func_name = "xnat_download_scan_resource_zip"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)

        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            base = XNAT_HOST.rstrip("/")
            url = (
                f"{base}/data/experiments/{session_id}/scans/{scan_id}/resources/{resource_name}/files"
                f"?format=zip"
            )

            http = getattr(sess, "interface", sess)
            r = http.get(url, stream=True)
            if getattr(r, "status_code", 999) != 200:
                raise RuntimeError(
                    f"Download failed: HTTP {r.status_code} :: {getattr(r, 'text', '')[:300]}"
                )

            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        return zip_path

    except Exception as e:
        log_error(
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "resource_name": resource_name,
                "zip_path": zip_path,
                "error": str(e),
            },
            func_name,
        )
        raise


def xnat_ensure_scan_resource_exists(session_id: str, scan_id: str, resource_name: str) -> None:
    """
    Ensure scan resource folder exists (idempotent-ish).
    Uses REST PUT, which is commonly accepted by XNAT for resource creation.
    """
    func_name = "xnat_ensure_scan_resource_exists"
    try:
        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            base = XNAT_HOST.rstrip("/")
            http = getattr(sess, "interface", sess)
            resource_uri = f"/data/experiments/{session_id}/scans/{scan_id}/resources/{resource_name}"
            http.put(base + resource_uri)
    except Exception as e:
        log_error(
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "resource_name": resource_name,
                "error": str(e),
            },
            func_name,
        )
        raise
from typing import Optional

def xnat_upload_file_to_scan_resource(
    session_id: str,
    scan_id: str,
    resource_name: str,
    local_path: str,
    remote_filename: Optional[str] = None,
    ensure_resource: bool = True,
) -> None:
#     ...
#
#
# def xnat_upload_file_to_scan_resource(
#     session_id: str,
#     scan_id: str,
#     resource_name: str,
#     local_path: str,
#     remote_filename: str | None = None,
#     ensure_resource: bool = True,
# ) -> None:
    """
    Upload *any* local file into *any* scan resource folder.

    Upload target:
      /data/experiments/{session_id}/scans/{scan_id}/resources/{resource_name}/files/{remote_filename}

    - If remote_filename is None, uses basename(local_path)
    - If ensure_resource True, creates the resource folder if needed.
    - Raises on failure.
    """
    func_name = "xnat_upload_file_to_scan_resource"
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"local_path does not exist: {local_path}")

        if remote_filename is None:
            remote_filename = os.path.basename(local_path)

        if ensure_resource:
            xnat_ensure_scan_resource_exists(session_id, scan_id, resource_name)

        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            base = XNAT_HOST.rstrip("/")
            http = getattr(sess, "interface", sess)

            upload_uri = (
                f"/data/experiments/{session_id}/scans/{scan_id}/resources/{resource_name}/files/{remote_filename}"
            )

            with open(local_path, "rb") as f:
                r = http.put(base + upload_uri, files={"file": f})

            if getattr(r, "status_code", 999) not in (200, 201):
                raise RuntimeError(
                    f"Upload failed: HTTP {r.status_code} :: {getattr(r, 'text', '')[:300]}"
                )

    except Exception as e:
        log_error(
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "resource_name": resource_name,
                "local_path": local_path,
                "remote_filename": remote_filename,
                "error": str(e),
            },
            func_name,
        )
        raise
import inspect

def xnat_get_scans_list_in_session(session_id: str):
    """
    (1) Given an XNAT session/experiment ID, return list of scans in that session.

    Returns: list[dict]
      [
        {"scan_id": "1", "type": "...", "quality": "...", "series_description": "...", "frames": ...},
        ...
      ]
    """
    func_name = inspect.currentframe().f_code.co_name
    try:
        out = []
        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            if session_id not in sess.experiments:
                log_error(f"Session ID not found on XNAT: {session_id}", func_name)
                return []

            exp = sess.experiments[session_id]
            for scan_id, scan in exp.scans.items():
                out.append({
                    "scan_id": str(scan_id),
                    "type": getattr(scan, "type", None),
                    "quality": getattr(scan, "quality", None),
                    "series_description": getattr(scan, "series_description", None),
                    "frames": getattr(scan, "frames", None),
                })
        return out

    except Exception:
        log_error({"session_id": session_id}, func_name)
        return []


def xnat_set_all_scan_types_in_session(session_id: str, new_type: str = "Z-Axial-Brain", dry_run: bool = False):
    """
    (2) For each scan in the session, update scan type to `new_type`.

    Returns dict summary:
      {
        "session_id": ...,
        "new_type": ...,
        "updated": [{"scan_id":..., "old_type":..., "method":...}, ...],
        "skipped": [{"scan_id":..., "reason":...}, ...],
        "errors":  [{"scan_id":..., "error":...}, ...]
      }

    Implementation notes:
    - Uses REST PUT via xnatpy interface.
    - Tries multiple common XNAT field keys for 'type' to handle different scan datatypes.
    """
    func_name = inspect.currentframe().f_code.co_name
    summary = {"session_id": session_id, "new_type": new_type, "updated": [], "skipped": [], "errors": []}

    try:
        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            if session_id not in sess.experiments:
                msg = f"Session ID not found on XNAT: {session_id}"
                log_error(msg, func_name)
                summary["errors"].append({"scan_id": None, "error": msg})
                return summary

            exp = sess.experiments[session_id]
            base = XNAT_HOST.rstrip("/")
            http = getattr(sess, "interface", sess)

            # Iterate scans
            for scan_id, scan in exp.scans.items():
                scan_id = str(scan_id)
                old_type = getattr(scan, "type", None)

                if old_type == new_type:
                    summary["skipped"].append({"scan_id": scan_id, "reason": "already_target_type"})
                    continue

                if dry_run:
                    summary["skipped"].append({"scan_id": scan_id, "reason": f"dry_run(old_type={old_type})"})
                    continue

                # Try different attribute keys (XNAT differs by scan datatype)
                # Most common is xnat:imagescandata/type
                candidates = [
                    "xnat:ctScanData/type",
                    "type",
                ]

                updated = False
                last_err = None

                for key in candidates:
                    try:
                        # PUT /data/experiments/{session_id}/scans/{scan_id}?{key}={new_type}
                        url = f"{base}/data/experiments/{session_id}/scans/{scan_id}"
                        r = http.put(url, params={key: new_type})

                        code = getattr(r, "status_code", None)
                        ok = (code in (200, 201)) or getattr(r, "ok", False)

                        if ok:
                            summary["updated"].append({
                                "scan_id": scan_id,
                                "old_type": old_type,
                                "method": f"PUT params {key}={new_type}",
                                "http_status": code,
                            })
                            updated = True
                            break
                        else:
                            last_err = f"HTTP {code} {getattr(r, 'text', '')[:200]}"
                    except Exception as e:
                        last_err = str(e)

                if not updated:
                    summary["errors"].append({"scan_id": scan_id, "error": last_err or "unknown_error"})

        return summary

    except Exception:
        log_error({"session_id": session_id, "new_type": new_type}, func_name)
        summary["errors"].append({"scan_id": None, "error": "Unhandled exception (see error log)"})
        return summary
def xnat_download_file_from_project_resource(
        project_id: str,
        resource_name: str,
        filename: str,
        local_directory: str,
        chunk_size: int = 1024 * 1024,
) -> str:
    """
    Download a specific file from a PROJECT-level resource folder.

    Parameters
    ----------
    project_id : str
        XNAT project ID
    resource_name : str
        Resource folder name under the project
    filename : str
        Exact filename inside the resource folder
    local_directory : str
        Local directory where file will be saved
    chunk_size : int
        Streaming chunk size (default 1MB)

    Returns
    -------
    str
        Full local file path if successful

    Raises
    ------
    Exception on failure
    """

    func_name = "xnat_download_file_from_project_resource"

    try:
        if not project_id or not resource_name or not filename:
            raise ValueError("project_id, resource_name, and filename are required")

        os.makedirs(local_directory, exist_ok=True)

        local_path = os.path.join(local_directory, filename)

        with xnat.connect(XNAT_HOST, user=XNAT_USER, password=XNAT_PASS) as sess:
            base = XNAT_HOST.rstrip("/")
            http = getattr(sess, "interface", sess)

            download_url = (
                f"{base}/data/projects/{project_id}/resources/"
                f"{resource_name}/files/{filename}"
            )

            r = http.get(download_url, stream=True)

            if getattr(r, "status_code", 999) != 200:
                raise RuntimeError(
                    f"Download failed: HTTP {r.status_code} :: "
                    f"{getattr(r, 'text', '')[:300]}"
                )

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        return local_path

    except Exception as e:
        log_error(
            {
                "project_id": project_id,
                "resource_name": resource_name,
                "filename": filename,
                "local_directory": local_directory,
                "error": str(e),
            },
            func_name,
        )
        raise