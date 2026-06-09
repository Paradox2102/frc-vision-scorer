# SAM 3.1 Tracking Backend

SAM 3.1 is available as an experimental alternative to the default SAM 2.1
tracker. The rest of the field-tracking pipeline is shared between backends.

## Setup

1. Request access to `facebook/sam3.1` on Hugging Face.
2. Authenticate on the inference host with `hf auth login`.
3. Install the optional dependency set:

   ```bash
   pip install -e '.[sam3]'
   ```

The SAM3 package requires NumPy `<2`, so this install may downgrade an existing
NumPy 2.x environment. A separate deployment environment is recommended while
the backend remains experimental.

## Run

Select **SAM 3.1** in the web upload form, or set it as the process default:

```bash
TRACKING_BACKEND=sam3 python web/app.py
```

The checkpoint downloads from Hugging Face on first use. To use a pre-downloaded
checkpoint instead:

```bash
SAM3_CHECKPOINT=/models/sam3.1_multiplex.pt TRACKING_BACKEND=sam3 python web/app.py
```

Optional H100 optimizations:

```bash
SAM3_COMPILE=1 SAM3_USE_FA3=1 TRACKING_BACKEND=sam3 python web/app.py
```

`SAM3_USE_FA3=1` requires the optional Flash Attention 3 installation described
by Meta. Leave it disabled until that dependency is installed on the target host.
