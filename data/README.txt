Dataset - License Plate Deblurring
==================================

This folder contains the input data used in the project.

Structure
---------
data/
    synthetic/
        sharp/      30 sharp synthetic license plates (PNG, 128x384)
        blurred/    30 corresponding motion-blurred plates (PNG, 128x384)
        restored/   outputs produced by the restoration pipeline

    real/
        car_real.jpeg                 real car image (reference)
        car_blurred.jpeg              blurred real car image (input)
        restored/car_restored.jpeg    restored real image (output)

Notes
-----
- Synthetic data is generated/used by the notebook at code/License_Plate_Deblurring.ipynb.
- The notebook demonstrates a full pipeline from data loading to restoration and evaluation.
