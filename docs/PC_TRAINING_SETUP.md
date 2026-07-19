# Training on the PC (RX 9070 XT) — Complete Beginner Setup Guide

Goal: turn your Windows gaming PC into a GPU training server that the Mac
(and Claude, via SSH) can send TCN training jobs to, and get trained model
artifacts back. No dual-boot required — your RX 9070 XT is **officially
supported by AMD ROCm inside WSL2** (verified July 2026 against AMD's live
compatibility matrix, ROCm 7.2.1 Radeon track).

```
┌─────────────── Mac (ScalpTrader home) ───────────────┐
│ live engine · corpus · orchestration · sim gates     │
│          │  ssh / rsync over Tailscale               │
└──────────┼───────────────────────────────────────────┘
           ▼
┌─────────────── Windows PC (RX 9070 XT) ──────────────┐
│ WSL2 Ubuntu 24.04 → ROCm → PyTorch → train_tcn.py    │
│ artifacts (model.pt, scaler.json) rsync'd back       │
└──────────────────────────────────────────────────────┘
```

Do the parts in order. Each has: what you're doing, exact steps, and the
source to follow if stuck. Budget ~2–3 hours total, mostly waiting on
installers.

---

## Part A — Prepare Windows (15 min)

**What**: WSL2 GPU compute requires a recent AMD driver on the Windows side.
AMD's WSL support for the RX 9000 series requires **Adrenalin 26.1.1 or
newer; 26.2.2+ is recommended** (it ships the new ROCDXG library that makes
WSL compute independent of display-driver updates).

1. Press `Win`, type "AMD Software: Adrenalin Edition", open it.
2. Top-right gear → System → check the driver version. If older than 26.2.2,
   download the latest from https://www.amd.com/en/support (pick Radeon RX
   9070 XT) and install, then reboot.
3. Make sure Windows 11 is updated: Settings → Windows Update.

Source: AMD WSL compatibility matrix —
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html

## Part B — Install WSL2 + Ubuntu 24.04 (20 min)

**What**: WSL2 runs a real Ubuntu Linux inside Windows — no partitioning, no
risk to your Windows install, uninstallable any time.

1. Press `Win`, type "PowerShell", right-click → **Run as administrator**.
2. Run:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. Reboot when prompted. After reboot a black terminal window opens and
   finishes the install; it asks you to create a **Linux username and
   password** (this is separate from Windows — remember it; the password
   prompt shows nothing while you type, that's normal).
4. You now have a Linux prompt like `you@PCNAME:~$`. Later you can reopen it
   any time: press `Win`, type "Ubuntu", open the app.
5. Update it (Linux's equivalent of Windows Update; run in the Ubuntu
   window):
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

Sources:
- Official Microsoft doc (updated 2026): https://learn.microsoft.com/en-us/windows/wsl/install
- Video walkthrough: NetworkChuck, "Linux on Windows......Windows on Linux" —
  https://www.youtube.com/watch?v=vxTW22y8zV8
- New to the Linux terminal? 30-min primer (official Ubuntu):
  https://documentation.ubuntu.com/desktop/en/latest/tutorial/the-linux-command-line-for-beginners/

## Part C — Install ROCm inside WSL2 (30 min)

**What**: ROCm is AMD's CUDA-equivalent — the layer that lets PyTorch use
the 9070 XT. Follow AMD's official WSL how-to **exactly as written on the
live page** (installer URLs change with each release; the page is the source
of truth):

**→ https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html**

The shape of it (run inside the Ubuntu window):
1. Download AMD's installer package (`amdgpu-install` .deb for Ubuntu 24.04)
   with the `wget` command the page gives you.
2. Install it: `sudo apt install ./amdgpu-install_*.deb`
3. Run the WSL-specific usecase (note: **no display driver inside WSL**, the
   page's command handles this — typically
   `amdgpu-install -y --usecase=wsl,rocm --no-dkms`).
4. Verify the GPU is visible from Linux:
   ```bash
   rocminfo | grep -i "gfx"
   ```
   You should see `gfx1201` (that's the 9070 XT). If `rocminfo` prints
   nothing or errors, see Troubleshooting below.

## Part D — Python + PyTorch for ROCm (20 min)

**What**: install the AMD-built PyTorch that talks to ROCm. AMD's WSL
PyTorch page is the authoritative command source (currently PyTorch 2.9.1 on
ROCm 7.2.1, **requires Python 3.12**):

**→ follow the "Install PyTorch for WSL" section of the same how-to guide
above** (it installs `torch` wheels from `repo.radeon.com`).

Then verify — this is the moment of truth:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```
Expected output: `True 7.2...` — on AMD, PyTorch deliberately reuses the
`torch.cuda` API, so our `--device auto` flag (which checks
`torch.cuda.is_available()`) picks the 9070 XT with zero code changes.
`torch.version.hip` being non-None is how you know it's the ROCm build.

Source: PyTorch HIP semantics — https://docs.pytorch.org/docs/stable/notes/hip.html
(Note: pytorch.org's own Linux ROCm wheels — `pip3 install torch
--index-url https://download.pytorch.org/whl/rocm7.2` — are the alternative
if AMD's wheels give trouble; both are legitimate.)

## Part E — Tailscale: connect Mac and PC (15 min)

**What**: Tailscale is a zero-config private network ("VPN between your own
devices"). It gives both machines stable names so SSH works from anywhere,
even off your home network. Free for personal use.

1. On the **PC (Windows side, not Ubuntu)**: download and run the installer
   from https://tailscale.com/download — then right-click the new tray icon →
   Log in (create the account with Google/GitHub/etc.).
2. On the **Mac**: install from https://tailscale.com/download/mac, launch,
   Allow the VPN prompt, log into the SAME account.
3. Both machines now appear at https://login.tailscale.com/admin with names
   like `gaming-pc` and `sreenaths-macbook`. MagicDNS is on by default, so
   the Mac can reach the PC as just `gaming-pc`.

Sources:
- Official quickstart (validated 2026): https://tailscale.com/docs/how-to/quickstart
- Video: "How to get started with Tailscale in under 10 minutes" —
  https://www.youtube.com/watch?v=sPdvyR7bLqI

## Part F — SSH: let the Mac open a terminal on the PC (30 min)

**What**: SSH is remote terminal access. The current best-practice route for
"Mac → Windows PC → land directly in WSL Ubuntu" is: enable Windows'
**built-in OpenSSH Server** as the front door and make it drop the session
straight into Ubuntu.

On the **PC (Windows)**:
1. Settings → System → Optional features → "Add an optional feature" →
   install **OpenSSH Server**.
2. Admin PowerShell:
   ```powershell
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   ```
3. Make SSH land in Ubuntu instead of Windows CMD:
   ```powershell
   New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
     -Value "C:\Windows\System32\wsl.exe" -PropertyType String -Force
   ```

On the **Mac**:
4. Create your key (one time; press Enter at every prompt):
   ```bash
   ssh-keygen -t ed25519
   ```
5. Show the public key: `cat ~/.ssh/id_ed25519.pub` — copy the whole line.
6. First login (use your **Windows** username and password):
   ```bash
   ssh WINDOWSUSER@gaming-pc
   ```
   You should land at the Ubuntu prompt. Type `exit` to leave.
7. Passwordless logins — on the PC, paste your public key into
   `C:\ProgramData\ssh\administrators_authorized_keys` (create the file with
   Notepad run as administrator; this special path is a Windows quirk — for
   admin accounts, `~/.ssh/authorized_keys` is IGNORED). Then in admin
   PowerShell fix its permissions:
   ```powershell
   icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
   ```
8. Test from the Mac: `ssh WINDOWSUSER@gaming-pc` should now log in with no
   password, straight into Ubuntu.

Sources:
- Microsoft OpenSSH Server doc (updated 2026): https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse
- The pattern's origin, still current: Hanselman, "THE EASY WAY how to SSH
  into Bash and WSL2" — https://www.hanselman.com/blog/the-easy-way-how-to-ssh-into-bash-and-wsl2-on-windows-10-from-an-external-machine
- 2026 walkthrough of exactly this stack (Tailscale + Windows OpenSSH +
  WSL2): https://benjijang.com/posts/2026/03/tailscale-wsl2-ssh/
- Mac key generation (GitHub Docs): https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

## Part G — Copy the project + data to the PC (20 min)

From the **Mac** (all one-time; re-runs only sync changes):
```bash
# code (excludes junk); trailing slashes matter
rsync -avz --progress \
  --exclude .venv --exclude runs --exclude data --exclude node_modules \
  ~/ScalpTrader/ WINDOWSUSER@gaming-pc:~/ScalpTrader/

# corpus + manifest (several GB — LAN will take a while)
rsync -avz --progress ~/ScalpTrader/data/corpus/manifest.csv \
  WINDOWSUSER@gaming-pc:~/ScalpTrader/data/corpus/
rsync -avz --progress ~/ScalpTrader/data/corpus/1s/ \
  WINDOWSUSER@gaming-pc:~/ScalpTrader/data/corpus/1s/
```
On the **PC** (inside Ubuntu), build its Python environment:
```bash
cd ~/ScalpTrader
sudo apt install -y python3.12-venv
python3.12 -m venv .venv
.venv/bin/pip install -e ./research   # or: pip install pandas pyarrow scikit-learn joblib pytest
# then the ROCm torch install from Part D goes into THIS venv:
#   follow the AMD pip commands using .venv/bin/pip
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"  # must be True
```

## Part H — First training run (the payoff)

From the Mac (or on the PC directly):
```bash
ssh WINDOWSUSER@gaming-pc
cd ~/ScalpTrader
nohup .venv/bin/python research/scripts/train_tcn.py \
  --barrier-mode vol --vol-target-mult 1.0 --vol-stop-mult 0.5 --timeout 120 \
  --test-start-date 2025-06-27 --train-quality-limit 451 \
  --val-start-date 2025-01-01 --device auto \
  --window 240 --channels 64 --blocks 4 --epochs 30 --patience 5 \
  --batch 512 --neg-frac 0.12 > ~/tcn_run.log 2>&1 &
tail -f ~/tcn_run.log        # watch it train; Ctrl-C stops watching, not training
```
When done, from the Mac pull the artifact back:
```bash
rsync -avz WINDOWSUSER@gaming-pc:~/ScalpTrader/runs/tcn/ ~/ScalpTrader/runs/tcn/
```
Artifacts are device-portable by design (`model.pt` is loaded with
`map_location="cpu"`), so the Mac sim-gates and serves them unchanged.

Once `ssh WINDOWSUSER@gaming-pc` works from the Mac, tell Claude — training
commands can then be dispatched to the PC inside the normal improvement
workflows (bigger corpus, wider searches) while the Mac stays free.

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `rocminfo` empty / errors in WSL | Windows AMD driver too old (need ≥26.1.1, want 26.2.2+); then `wsl --update` in PowerShell and `wsl --shutdown`, reopen Ubuntu |
| `torch.cuda.is_available()` → False | Wrong torch wheel (CPU build). Reinstall per Part D inside `.venv`; confirm `torch.version.hip` is not None |
| `ssh` → "Connection refused" | `Start-Service sshd` on the PC; check Tailscale is connected on both (tray/menu-bar icons) |
| `ssh` asks for a password despite key | The key must be in `C:\ProgramData\ssh\administrators_authorized_keys` with the `icacls` permissions from Part F step 7 |
| Landed in Windows CMD, not Ubuntu | The DefaultShell registry step (Part F step 3) missing, or run `wsl` manually |
| Training killed / OOM in WSL | WSL defaults to half your RAM; create `C:\Users\YOU\.wslconfig` with `[wsl2]` `memory=24GB`, then `wsl --shutdown` |
| Ubuntu 26.04 temptation | Don't — AMD's support matrix pins the 9070 XT to Ubuntu 24.04/22.04 for now |

## Fallback: dual-boot native Ubuntu (only if WSL2 disappoints)

Native Linux is slightly faster and the most battle-tested ROCm path, at the
cost of real partitioning risk. If ever needed:
- Article (2025, covers Secure Boot + backups): https://www.tomshardware.com/software/linux/how-to-dual-boot-linux-and-windows-on-any-pc
- Video (Ubuntu 24.04 + Windows 11, Learn Linux TV): https://www.youtube.com/watch?v=qypfkDx_Qnc
- Then install ROCm per: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html

## Verified facts this guide rests on (as of 2026-07-19)

- RX 9070 XT (gfx1201) officially supported by ROCm on native Linux (Ubuntu
  24.04.4 / 22.04.5) AND under WSL2 (Ubuntu 24.04 / 22.04 guests), per AMD's
  Radeon-track docs at ROCm 7.2.1 with PyTorch 2.9.1 production support.
- WSL2 host requirement: AMD Adrenalin 26.1.1+, 26.2.2+ recommended (ROCDXG).
- pytorch.org stable channel ships Linux ROCm 7.2 wheels (`torch 2.13`);
  it still ships NO Windows-native ROCm — AMD's separate Windows preview
  track exists but is not beginner-recommended (Triton "in progress").
- PyTorch on ROCm reuses the `torch.cuda` API; `torch.version.hip` non-None
  identifies the ROCm build — which is why ScalpTrader's `--device auto`
  works unmodified.
