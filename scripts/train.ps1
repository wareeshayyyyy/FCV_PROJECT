# PowerShell helper to run training inside a virtual env
param(
    [string]$DataCsv = "data/labels.csv",
    [string]$DataDir = "data/images",
    [int]$Epochs = 10,
    [int]$BatchSize = 16
)

if (-not (Test-Path -Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
python -m bonefracture.train --data_csv $DataCsv --data_dir $DataDir --epochs $Epochs --batch_size $BatchSize
