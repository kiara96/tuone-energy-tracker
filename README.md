# Tuone: Real-Time Information on Clean Technology Projects
## Overview
We are building an application, Tuone, that provides real-time information on projects related to the manufacturing and deployment of clean technologies.

### Micro-Applications
1. Scrape News Articles and Information
Tuone scrapes news articles and relevant information from sources in the field, such as electrive.com.

2. Extract Project-Specific Information
Tuone extracts key information related to projects, including investment details, country, timeline, technologies, and more.

3. Generate Project Summaries
Tuone generates summary overviews for each project, providing essential information along with related sources.

## Breakdown of Each Component
data/

This directory will store the dataset and any other data-related files.


models/

This directory will store trained models and checkpoints.

src/

__init__.py: Makes Python treat the directories as containing packages.

config.py: Contains configuration settings or constants used across the project.

data_preparation.py: Handles data loading, preprocessing, and preparation for training.

model.py: Contains the model class and any related functionality.

tokenizer.py: Handles tokenization and any preprocessing that is specific to the tokenizer.

train.py: Orchestrates the training and evaluation process.

requirements.txt

Lists all Python libraries that the project depends on.


README.md
Describes the project, its purpose, and basic instructions on how to set it up and run it.


.gitignore
Specifies intentionally untracked files that Git should ignore.