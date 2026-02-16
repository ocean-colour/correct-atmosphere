# Notebook 06, TOA from OSOAA

## Cell 6 of the 06_osoaa_toa_simulation.ipynb Notebook in correct-atmosphere/nb/ is executing but none of the OSOAA calculations are running successfully.  Please explore.  If you need to run python, be sure to use the "ocean14" environment in conda.

## Modify 06_osoaa_toa_simulation.ipynb to also read in the Advanced outputs from LUM_Advanced.txt and add the radiance components to the output JSON file.

# Notebook 04, Full correction

## Step 4 in the 04_full_correction.ipynb Notebook in correct-atmosphere/nb/ crashes on a shape error. Please explore.  If you need to run python, be sure to use the "ocean14" environment in conda.

# Notebook 07, Correction on OSOAA output

## Generate a Notebook 07_correction_on_osoaa_output.ipynb that corrects the OSOAA output using the correct_atmosphere package. See 04_full_correction.ipynb for an example. You will need to load in the OSOAA output nb/osoaa_toa_dataset.json file for the TOA radiances, the viewing geometry, and the ancillary data.

## Turn the primary methods into a module called correction_on_osoaa_output.py and put it in the dev/ folder.  Where possible use methods from module correct-atmosphere/osoaa.py

# Module from Notebook 06

## Please turn the 06_osoaa_toa_simulation.ipynb Notebook in correct-atmosphere/nb/ into a module that can be imported into other notebooks.  Call that module osoaa_toa_simulation.py. and put it in the dev/ folder that I just created.

## I have moved that module into correct-atmosphere/osssa.py Please update the Notebook to use the new module.