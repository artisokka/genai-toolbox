import os
import subprocess
import time
import urllib.request
from typing import Optional


def _ollama_is_up(url: str = "http://127.0.0.1:11434/api/tags", timeout_sec: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec):
            return True
    except Exception:
        return False


def ensure_ollama_running(status_placeholder: Optional[object] = None, wait_seconds: int = 20) -> bool:
    """Ensure Ollama server is running locally. Attempts to start it if not.
    Returns True if running, False otherwise.
    """
    if _ollama_is_up():
        return True

    def _spawn(cmd_list):
        creationflags_local = 0
        if os.name == 'nt':
            creationflags_local = getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'DETACHED_PROCESS', 0)
        subprocess.Popen(
            cmd_list,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name != 'nt'),
            creationflags=creationflags_local,
        )

    def _wait_until_up(max_wait: int) -> bool:
        start = time.time()
        while time.time() - start < max_wait:
            if status_placeholder:
                status_placeholder.info("Waiting for Ollama to start...")
            if _ollama_is_up():
                if status_placeholder:
                    status_placeholder.success("Ollama is running.")
                return True
            time.sleep(1)
        return False

    # 1) Plain spawn (assumes ollama is in PATH)
    try:
        _spawn(["ollama", "serve"])
        if _wait_until_up(wait_seconds):
            return True
    except Exception:
        pass

    # 2) Platform-specific fallbacks
    if os.name == 'nt':
        # 2a) Use cmd start to launch in a new window
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if _wait_until_up(wait_seconds):
                return True
        except Exception:
            pass
        # 2b) Use PowerShell Start-Process hidden
        try:
            subprocess.Popen([
                "powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command",
                "Start-Process -FilePath ollama -ArgumentList 'serve' -WindowStyle Hidden"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if _wait_until_up(wait_seconds):
                return True
        except Exception:
            pass
        # 2c) Bundled batch (last resort)
        try:
            script_dir_local = os.path.dirname(os.path.abspath(__file__))
            bat_path = os.path.join(script_dir_local, "rag", "ollama.bat")
            if os.path.exists(bat_path):
                subprocess.Popen([bat_path, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if _wait_until_up(wait_seconds):
                    return True
        except Exception:
            pass
    else:
        # POSIX: try nohup
        try:
            subprocess.Popen(["nohup", "ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
            if _wait_until_up(wait_seconds):
                return True
        except Exception:
            pass

    if status_placeholder:
        status_placeholder.error("Ollama did not become ready. Please run 'ollama serve' manually and reload.")
    return False


def _build_index_to_name(mapping, expected_len=None):
    """Return a list mapping index->name from various mapping formats.
    Supports: list/tuple, {int: str}, {str: int} (inverted), or {str: sequence-of-names} like {'label_diag': [...]}.
    Chooses the sequence whose length matches expected_len when available. Guarantees length==expected_len if provided (pads with None).
    """
    def _fit(seq):
        if expected_len is None:
            return list(seq)
        seq = list(seq)
        if len(seq) >= expected_len:
            return seq[:expected_len]
        return seq + [None] * (expected_len - len(seq))

    if mapping is None:
        return None
    if isinstance(mapping, (list, tuple)):
        return _fit(mapping)
    if isinstance(mapping, dict):
        try:
            keys = list(mapping.keys())
            if keys and all(isinstance(k, int) for k in keys):
                max_idx = int(max(keys))
                size = max_idx + 1
                if expected_len is not None:
                    size = max(size, expected_len)
                idx_to_name = [None] * size
                for k, v in mapping.items():
                    idx = int(k)
                    if 0 <= idx < size:
                        idx_to_name[idx] = str(v)
                return _fit(idx_to_name)
            if keys and all(isinstance(v, int) for v in mapping.values()):
                max_idx = int(max(mapping.values())) if mapping else -1
                size = max_idx + 1
                if expected_len is not None:
                    size = max(size, expected_len)
                idx_to_name = [None] * size
                for name, idx in mapping.items():
                    idx = int(idx)
                    if 0 <= idx < size:
                        idx_to_name[idx] = str(name)
                return _fit(idx_to_name)
            seq_candidates = []
            for k, v in mapping.items():
                if isinstance(v, (list, tuple)) and (len(v) == 0 or isinstance(v[0], str)):
                    seq_candidates.append((k, list(v)))
            if seq_candidates:
                if expected_len is not None:
                    for k, seq in seq_candidates:
                        if len(seq) == expected_len:
                            return _fit(seq)
                seq_candidates.sort(key=lambda kv: (("diag" in kv[0].lower()), len(kv[1])), reverse=True)
                return _fit(seq_candidates[0][1])
        except Exception:
            return None
    return None


def _humanize_diag(code: str) -> str:
    mapping = {
        'NORM': 'Normal ECG',
        '1AVB': 'First-degree AV block',
        '2AVB': 'Second-degree AV block',
        '3AVB': 'Third-degree AV block',
        'RBBB': 'Right bundle branch block',
        'CRBBB': 'Complete right bundle branch block',
        'IRBBB': 'Incomplete right bundle branch block',
        'LBBB': 'Left bundle branch block',
        'CLBBB': 'Complete left bundle branch block',
        'ILBBB': 'Incomplete left bundle branch block',
        'AFIB': 'Atrial fibrillation',
        'AFL': 'Atrial flutter',
        'STACH': 'Sinus tachycardia',
        'SBRAD': 'Sinus bradycardia',
        'LVH': 'Left ventricular hypertrophy',
        'WPW': 'Wolff–Parkinson–White pattern',
        'AMI': 'Acute myocardial infarction',
        'ALMI': 'Anterolateral myocardial infarction',
        'ASMI': 'Anteroseptal myocardial infarction',
        'ILMI': 'Inferolateral myocardial infarction',
        'IMI': 'Inferior myocardial infarction',
        'IPMI': 'Inferoposterior myocardial infarction',
        'IPLMI': 'Inferoposterolateral myocardial infarction',
        'SEHYP': 'Secondary ST-T abnormality',
        'ISCAL': 'Ischemia anterolateral',
        'ISCAN': 'Ischemia anterior',
        'ISCAS': 'Ischemia anteroseptal',
        'ISCIL': 'Ischemia inferolateral',
        'ISCIN': 'Ischemia inferior',
        'ISCLA': 'Ischemia lateral',
        'ISC_': 'Ischemia (unspecified)',
        'IVCD': 'Intraventricular conduction delay',
        'DIG': 'Digitalis effect',
        'LNGQT': 'Prolonged QT interval',
        'RAO/RAE': 'Right atrial enlargement',
        'LAO/LAE': 'Left atrial enlargement',
        'RVH': 'Right ventricular hypertrophy',
        'PMI': 'Pacemaker influence',
        'NST_': 'Non-specific ST changes',
    }
    c = (code or '').strip().upper()
    return mapping.get(c, code)

